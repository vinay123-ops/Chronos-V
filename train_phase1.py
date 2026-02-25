import torch
import torch.nn as nn
import torch.nn.functional as F
import os, cv2, math
import numpy as np
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from mamba_ssm import Mamba

# ==========================================================
# 1. CORE CHRONOS-V COMPONENTS
# ==========================================================
class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class CLIPDirector(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", d_model=256):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.clip = CLIPTextModel.from_pretrained(model_name)
        for param in self.clip.parameters(): param.requires_grad = False
        self.text_projector = nn.Sequential(nn.Linear(self.clip.config.hidden_size, d_model), nn.LayerNorm(d_model), nn.GELU())
    def forward(self, text_prompts):
        device = next(self.parameters()).device
        inputs = self.tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = self.clip(**inputs)
        return self.text_projector(outputs.pooler_output)

class WaveletRouter(nn.Module):
    def __init__(self, in_channels, threshold=0.1):
        super().__init__()
        self.threshold = threshold
        self.haar_weights = nn.Parameter(torch.zeros(in_channels * 4, 1, 2, 2, dtype=torch.float32), requires_grad=False)
        ll, hl, lh, hh = [torch.tensor([[1.,1.],[1.,1.]])/4., torch.tensor([[-1.,-1.],[1.,1.]])/4., 
                          torch.tensor([[-1.,1.],[-1.,1.]])/4., torch.tensor([[1.,-1.],[-1.,1.]])/4.]
        for c in range(in_channels):
            for i, f in enumerate([ll, hl, lh, hh]): self.haar_weights[c * 4 + i, 0] = f
    def forward(self, x):
        B, C, T, H, W = x.shape
        x_s = x.transpose(1, 2).reshape(B * T, C, H, W)
        dwt = F.conv2d(x_s, self.haar_weights.to(x.device), stride=2, groups=C)
        energy = dwt.view(B * T, C, 4, H//2, W//2)[:, :, 1:, :, :].abs().mean(dim=(1, 2))
        mask = (energy > self.threshold).float().view(B * T, 1, H//2, W//2)
        return F.interpolate(mask, size=(H, W), mode='nearest').view(B, T, H, W)

class Apply3DRoPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dim_per_axis = 86 
    def _apply_1d_rope(self, x, pos, dim):
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        angles = pos.unsqueeze(-1) * freqs
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        x_rotated[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
        return x_rotated
    def forward(self, x_3d):
        B, T, H, W, C = x_3d.shape
        x_pad = F.pad(x_3d, (0, 2)) 
        x_t, x_h, x_w = x_pad.split(self.dim_per_axis, dim=-1)
        pos_t = torch.arange(T, device=x_3d.device).view(T, 1, 1).expand(T, H, W)
        pos_h = torch.arange(H, device=x_3d.device).view(1, H, 1).expand(T, H, W)
        pos_w = torch.arange(W, device=x_3d.device).view(1, 1, W).expand(T, H, W)
        roped = torch.cat([self._apply_1d_rope(x_t, pos_t, self.dim_per_axis), 
                           self._apply_1d_rope(x_h, pos_h, self.dim_per_axis), 
                           self._apply_1d_rope(x_w, pos_w, self.dim_per_axis)], dim=-1)
        return roped[..., :256]

class ComplexMamba3D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.rope = Apply3DRoPE(d_model)
        self.norm = nn.LayerNorm(d_model) 
        self.m_t = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.m_h = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.m_w = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.fusion = nn.Linear(d_model * 3, d_model)
    def forward(self, x_3d):
        x = self.rope(x_3d) 
        x = self.norm(x)
        B, T, H, W, C = x.shape
        x_t = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        t_out = self.m_t(x_t).view(B, H, W, T, C).permute(0, 3, 1, 2, 4)
        x_h = x.permute(0, 1, 3, 2, 4).reshape(B * T * W, H, C)
        h_out = self.m_h(x_h).view(B, T, W, H, C).permute(0, 1, 3, 2, 4)
        x_w = x.reshape(B * T * H, W, C)
        w_out = self.m_w(x_w).view(B, T, H, W, C)
        return self.fusion(torch.cat([t_out, h_out, w_out], dim=-1))

# ==========================================================
# 2. MASTER ARCHITECTURE (With AdaLN)
# ==========================================================
class ChronosVAdvanced(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.director = CLIPDirector(d_model=d_model) 
        self.backbone = ComplexMamba3D(d_model=d_model)
        self.router = WaveletRouter(in_channels=d_model)
        
        # Mamba needs channel dimension LAST (B, T, H, W, C)
        self.vae_projector = nn.Linear(4, d_model)
        self.output_projector = nn.Linear(d_model, 4)
        
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbeddings(d_model), nn.Linear(d_model, d_model), nn.SiLU())
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 2))
        
        self.router_gate = nn.Parameter(torch.ones(1)) 

    def forward(self, x_vae_latents, prompt_texts, t):
        # 1. Project channels: (B, T, H, W, 4) -> (B, T, H, W, 256)
        x = self.vae_projector(x_vae_latents) 
        
        # 2. AdaLN Time + Text Fusion
        t_emb = self.time_mlp(t.view(-1))
        text_blueprint = self.director(prompt_texts) 
        cond_emb = t_emb + text_blueprint
        
        scale, shift = self.adaln_modulation(cond_emb).chunk(2, dim=-1)
        scale, shift = scale.view(x.shape[0], 1, 1, 1, -1), shift.view(x.shape[0], 1, 1, 1, -1)
        x = x * (1 + scale) + shift
        
        # 3. Backbone
        velocity = self.backbone(x)
        
        # 4. Routing (Needs B, C, T, H, W)
        mask = self.router(x.permute(0, 4, 1, 2, 3))
        routed_velocity = velocity + (velocity * mask.unsqueeze(-1) * self.router_gate)
        
        return self.output_projector(routed_velocity)

def save_video(tensor_list, filename):
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 8, (256, 256))
    for frame in tensor_list: 
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

# ==========================================================
# 3. ALL-IN-ONE PIPELINE
# ==========================================================
def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🚀 Booting Ultimate Chronos-V Mamba Engine...")
    vid_path = "training_videos/sample_walk.mp4"
    if not os.path.exists(vid_path):
        print(f"❌ Error: '{vid_path}' not found.")
        return

    vae = AutoencoderKL.from_pretrained("./models/sdxl-vae", local_files_only=True, torch_dtype=torch.float32).to(device)
    
    cap = cv2.VideoCapture(vid_path)
    frames = []
    while len(frames) < 8:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (256, 256)))
    cap.release()
    
    # 2D VAE input
    video_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        latents = vae.encode(video_tensor).latent_dist.sample() 
        # SD Scaling + Channel LAST format for Mamba -> (1, 8, 32, 32, 4)
        x_1_target = (latents * 0.18215).unsqueeze(0).permute(0, 2, 3, 4, 1) 
        
    print(f"✅ Created 5D Target Latents for Mamba: {x_1_target.shape}")

    model = ChronosVAdvanced().to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    prompt = ["A cinematic shot of a person walking across the frame"]

    print(f"📊 Training Sequence Initiated. Overfitting to video...")
    for epoch in range(1001):
        optimizer.zero_grad()
        
        x_1 = x_1_target
        x_0 = torch.randn_like(x_1) 
        t = torch.rand(1, 1, 1, 1, 1).to(device).float()
        
        x_t = (1.0 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0
        
        pred_velocity = model(x_t, prompt, t) 
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 100 == 0:
            print(f"📈 Epoch {epoch:4d} | Mamba Flow Loss: {loss.item():.6f}")

    print("\n🌊 Training Complete. Generating via 30-Step Euler ODE...")
    model.eval()
    
    x_current = torch.randn_like(x_1_target)
    steps = 30
    
    with torch.no_grad():
        for i in range(steps):
            t_val = torch.full((1, 1, 1, 1, 1), i / steps).to(device).float()
            v_pred = model(x_current, prompt, t_val)
            x_current = x_current + (v_pred * (1.0 / steps))
            
        print("✨ Decoding Latents...")
        x_current = x_current / 0.18215 
        
        # Mamba out is (B, T, H, W, C) -> VAE needs (B*T, C, H, W)
        x_vae_input = x_current.squeeze(0).permute(0, 3, 1, 2) 
        
        frames_out = []
        for i in range(x_vae_input.shape[0]):
            single_frame_latent = x_vae_input[i].unsqueeze(0) 
            decoded = vae.decode(single_frame_latent).sample
            
            # THE FLAWLESS DECODING MATH
            img = (decoded / 2 + 0.5) * 255
            img = img.clamp(0, 255).byte()
            img = img[0].permute(1, 2, 0).cpu().numpy()
            frames_out.append(img)
            
        save_video(frames_out, "chronos_mamba_output.mp4")
        print("🎬 Done! Check 'chronos_mamba_output.mp4'.")

if __name__ == "__main__":
    main()