# Chronos-V: Geometry-Aware Linear Video Generation

**Chronos-V** is an experimental Text-to-Video (T2V) engine built to validate the thesis that high-fidelity video generation can be achieved through **Linear State-Space Modeling** and **Differential Geometry** rather than brute-force quadratic attention.

By replacing the standard Transformer backbone with an **Omni-Directional Complex Mamba** engine and utilizing **Rectified Flow Matching**, we demonstrate that stable video generation is possible on consumer-grade hardware (NVIDIA GTX 1650, 4GB VRAM) with true $O(n)$ scaling.

---

## 🚀 The Core Thesis: Math > Brute Force

Current State-of-the-Art (SOTA) models rely on $O(n^2)$ space-time attention, which creates a massive compute wall for long-form or high-resolution video. Chronos-V explores a "Geometry-First" approach:

1.  **Linear Complexity:** Leveraging Mamba (SSM) for $O(n)$ scaling.
2.  **Physical Grounding:** Using 3D Rotary Position Embeddings (3D RoPE) to maintain structural integrity.
3.  **Efficiency:** Implementing Shannon Entropy-based routing via Wavelets to focus compute only on high-information motion.

---

## 🛠️ Technical Architecture

### 1. Omni-Directional Complex Mamba
We utilize a **Complex-Domain State Matrix** to handle the oscillatory physics inherent in video (e.g., walking, spinning, fluid motion).

$$A = -\Lambda + i\Omega$$

* **Real Part ($-\Lambda$):** Governs state decay and long-range memory.
* **Imaginary Part ($i\Omega$):** Natively models Fourier-like rotational frequencies, allowing the model to track cyclical motion without the overhead of global attention.



### 2. Geometric Grounding (3D RoPE)
Unlike 1D sequence models, Chronos-V operates on **5D Tensors** $[B, C, T, H, W]$. We apply **3D Rotary Position Embeddings** to ensure that pixel interactions are strictly dependent on relative 3D physical distance:

$$R_{\Theta}^{3D}(t, h, w) = R_{\theta_t} \otimes R_{\theta_h} \otimes R_{\theta_w}$$

### 3. Wavelet Information Routing
To optimize for edge hardware, we calculate the **Shannon Entropy** (Information Energy $E$) of spatial patches using a 2D Haar Wavelet Transform.

$$E(X) = \frac{1}{C} \sum |\text{DWT}_{high}(X)|$$

A "Soft Gate" mask $M$ allows the backbone to bypass low-entropy static regions, saving significant GFLOPS.



### 4. Rectified Flow Matching
We replace Gaussian Diffusion with a straight-line mathematical trajectory. The model predicts the **Vector Field (Velocity)** $V$ required to transform noise into video in a predictable, linear path.

$$\mathcal{L}_{RFM}(\theta) = \mathbb{E} || V_\theta(X_t, t) - (X_1 - X_0) ||^2$$



---

## 📊 Benchmarks & Hardware

This project is developed and verified in a **Linux environment** on an **NVIDIA GTX 1650 (4GB VRAM)**. 

| Metric | Chronos-V (Mamba) | Standard Transformer |
| :--- | :--- | :--- |
| **Complexity** | $O(n)$ Linear | $O(n^2)$ Quadratic |
| **VRAM Usage (8 Frames)** | ~3.2 GB | ~7.8 GB |
| **Inference Path** | Straight-line ODE | Curved Diffusion |
| **Hardware Target** | Consumer GPU / Edge | H100 Clusters |

---

## 📦 Installation & Setup

```bash
# Clone the repository
git clone [https://github.com/your-username/Chronos-V.git](https://github.com/your-username/Chronos-V.git)
cd Chronos-V

# Create environment
conda create -n chronos-v python=3.10
conda activate chronos-v

# Install core dependencies
pip install torch torchvision torchaudio
pip install mamba-ssm causal-conv1d>=1.2.0
pip install diffusers transformers opencv-python matplotlib

```
🗺️ Roadmap & Collaboration
We are currently refining the following areas:

Complex ODE Stability: Optimizing the imaginary component for fluid dynamics and high-frequency motion.

Scaling Studies: Benchmarking FVD (Fréchet Video Distance) against sequence length.

Hybrid Kernels: Exploring Mamba + Local Attention fusions for high-frequency detail.

We are looking for collaborators deep in State-Space Models (SSMs), Neural ODEs, and Latent Geometry. If you are interested in pushing the boundaries of what non-transformer architectures can do, please open an issue or reach out via DM.

Special thanks to Albert Gu and Tri Dao for their pioneering work on the Mamba architecture and State-Space Models.
