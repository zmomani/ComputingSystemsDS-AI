---
week: 13
title: "Understanding Diffusion Models"
source: https://albanna-tutorials.com/diffusion.html
---
# Understanding Diffusion Models

> Course page by Dr. Abedal-Kareem Al-Banna (Al-banna Tutorials). Code samples are PyTorch, as on the source page.

## 1. Introduction to Diffusion Models

Diffusion models are generative models that gained wide attention from 2020 onward. They create new samples by **gradually adding noise to training data, then learning to reverse that process**.

**Why they matter**
- State-of-the-art image quality.
- No adversarial training (unlike GANs).
- Comparatively stable training.
- Highly controllable generation.
- They power well-known tools such as DALL-E 2 and Stable Diffusion.

**Timeline**
| Year | Milestone |
|---|---|
| 2015 | Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* — the first diffusion model, inspired by non-equilibrium thermodynamics |
| 2020 | Ho et al., *Denoising Diffusion Probabilistic Models* (DDPM) — the foundation for modern implementations |
| 2021 | *Improved Denoising Diffusion Probabilistic Models* — better sampling techniques |
| 2022 | Stability AI releases Stable Diffusion, bringing diffusion models to the mainstream |

> **Key insight.** Diffusion models gradually add noise to data, then learn to reverse that process to generate realistic, diverse samples.

## 2. Fundamentals

Diffusion models destroy structure in data with a **forward process**, then learn to restore it with a **reverse process**.

- **Forward diffusion** — adds noise over many steps until the data becomes pure Gaussian noise.
- **Reverse diffusion** — starts from noise and removes it step by step to produce a clean sample resembling the training data.

**Key concepts**
- **Markov chain** — each step depends only on the previous one, which keeps the math tractable:

  q(x₁:T | x₀) := ∏ₜ₌₁ᵀ q(xₜ | xₜ₋₁)

- **Gaussian transitions** — each forward step adds Gaussian noise according to a **variance schedule βₜ**:

  q(xₜ | xₜ₋₁) = 𝒩(xₜ; √(1 − βₜ)·xₜ₋₁, βₜ·I)

- **Neural network** — any architecture whose input and output dimensions match can work; U-Nets are most common for images. The network is usually trained to predict (a) the added noise (most common), (b) the original data directly, or (c) the mean and variance of the posterior.
- **Timesteps** — T is a key hyperparameter; more steps usually improve quality but slow generation.

**Noise schedules**
| Schedule | Formula | Notes |
|---|---|---|
| Linear | βₜ = β₁ + (β_T − β₁) · (t−1)/(T−1) | Used in the original DDPM paper; can destroy most information around the halfway point |
| Cosine | βₜ = 1 − cos²(((t/T + s)/(1 + s)) · π/2) / cos²((s/(1 + s)) · π/2) | Introduced by OpenAI; preserves more information early, allows as few as ~50 steps |
| Quadratic | βₜ = β₁ + (β_T − β₁) · ((t−1)/(T−1))² | Adds noise faster later; can help preserve important structure early |

## 3. Forward Diffusion Process

The forward process gradually adds Gaussian noise until the data becomes pure noise.

1. **Start with original data** — sample x₀ from the real data distribution.
2. **Define the noise schedule** — choose β₁ … β_T (linear, cosine, or quadratic).
3. **Apply noise iteratively** — at each step mix the previous sample with fresh Gaussian noise, scaled by the schedule:

   xₜ = √(1 − βₜ)·xₜ₋₁ + √βₜ·εₜ,  εₜ ~ 𝒩(0, I)

4. **Sample efficiently** — any xₜ can be computed directly from x₀ using the cumulative product ᾱₜ = ∏ₛ₌₁ᵗ (1 − βₛ):

   q(xₜ | x₀) = 𝒩(xₜ; √ᾱₜ·x₀, (1 − ᾱₜ)·I)
   xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε

   This shortcut is essential for training.
5. **End result** — after T steps, x_T is approximately standard Gaussian noise; all original structure is gone.

> **Important insight.** The forward process is never used to generate images. It only defines the mathematical framework for learning the reverse process, which does the generating.

```python
def forward_diffusion_sample(x_0, t, device):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it at timestep t
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]

    # Forward diffusion formula: mean + variance
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
```

## 4. Reverse Diffusion Process

The reverse process is what gets learned during training and used during generation: it turns noise into structured data.

**The challenge.** Unlike the simple forward process, the reverse process is complex and must be learned from data. The model learns to predict and remove noise step by step.

**Mathematical formulation**

  p_θ(x₀:T) := p(x_T) · ∏ₜ₌₁ᵀ p_θ(xₜ₋₁ | xₜ)
  p_θ(xₜ₋₁ | xₜ) = 𝒩(xₜ₋₁; μ_θ(xₜ, t), Σ_θ(xₜ, t))

- Each transition's variance is often fixed to match the forward schedule, so the network mainly predicts the mean.
- Predicting the **noise component** ε_θ(xₜ, t) ≈ ε, rather than the mean directly, gives more stable training and better results.
- The denoising update subtracts the predicted-noise contribution and adds a small random term (except at the final step):

  xₜ₋₁ = (1/√αₜ) · (xₜ − ((1 − αₜ)/√(1 − ᾱₜ)) · ε_θ(xₜ, t)) + σₜ·z,  z ~ 𝒩(0, I), αₜ = 1 − βₜ

**Sampling steps**
1. Start with random Gaussian noise x_T.
2. For t = T down to 1: predict the noise with the network, then compute the less-noisy sample with the denoising formula.
3. Return x₀ as the generated output.

```python
def sample(model, n_samples, device, image_size):
    """
    Samples n_samples new images from the model
    """
    model.eval()
    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(n_samples, 3, image_size, image_size).to(device)

        for i in reversed(range(1, timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = model(x, t)

            # Get alpha values for this timestep
            alpha = alphas[t][:, None, None, None]
            alpha_hat = alphas_cumprod[t][:, None, None, None]
            beta = betas[t][:, None, None, None]

            # Only add noise if we're not at the last step
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Compute the denoised x_0 at this step
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

    # Rescale to [0, 1] for images
    return x.clamp(-1, 1).add(1).div(2)
```

**Advanced sampling techniques**
- **DDIM sampling** — uses non-Markovian paths to skip steps while maintaining quality.
- **Classifier guidance** — a separate classifier steers generation toward desired attributes.
- **Classifier-free guidance** — balances conditional and unconditional predictions to improve quality and prompt adherence; the guidance scale controls how strongly the output follows the condition.

## 5. Neural Network Architecture

The network predicts noise at each timestep. The most common choice for images is the **U-Net**:
- **Downsampling path** — captures context through convolutional layers.
- **Upsampling path** — rebuilds spatial detail.
- **Skip connections** — pass information directly from encoder to decoder.
- **Attention layers** — many implementations add self-attention at certain resolutions.

**Key architectural elements**
- **ResNet blocks** — the U-Net backbone: group normalization, SiLU activation, 3×3 convolutions, and a skip connection adding input to output.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.act(self.norm1(self.conv1(x)))
        time_emb = self.act(self.time_mlp(t))
        h = h + time_emb[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)
```

- **Self-attention blocks** — capture long-range dependencies, as in transformers; often placed in the middle of the U-Net where feature maps are medium resolution.
- **Timestep embeddings** — convert t into a high-dimensional sinusoidal embedding, pass it through an MLP, and inject it into each residual block.

```python
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
```

**Conditional generation** — the model can be conditioned on:
- **Class labels** — converted to embeddings and combined with timestep embeddings.
- **Text prompts** — encoded by a pre-trained text encoder such as CLIP, then injected into the model.
- **Other images** — encoded and used as extra conditioning, enabling image-to-image translation.

**Advanced architectures**
- **Latent diffusion models** — run diffusion in a smaller latent space rather than pixel space, greatly reducing compute. Stable Diffusion works on 4×64×64 latents rather than 3×512×512 images.
- **Cascaded diffusion** — chains models at increasing resolutions for very high-resolution output; DALL-E 2 uses this to reach 1024×1024.
- **Score-based models** — model the gradient of the log-density instead of predicting noise; under certain conditions mathematically equivalent to diffusion models.

> **Architectural flexibility.** The only hard requirement is matching input and output dimensions, which makes the architecture easy to adapt.

## 6. Training Diffusion Models

Training teaches the network to predict the noise added during the forward process.

**Objective.** The full objective is a variational bound on the negative log-likelihood, one KL term per timestep:

  L_vlb = L₀ + L₁ + … + L_{T−1} + L_T

In practice the **simplified objective** is used — mean squared error between the true and predicted noise at a random timestep:

  L_simple = 𝔼_{t, x₀, ε} [ ‖ε − ε_θ(xₜ, t)‖² ]

**Training algorithm**
1. Sample a batch of training data x₀.
2. Sample a random timestep t for each example.
3. Sample Gaussian noise ε.
4. Create noisy samples xₜ using the direct sampling formula.
5. Predict the noise with the network.
6. Compute the loss between predicted and true noise.
7. Update parameters with gradient descent.

```python
def train_step(model, x_0, optimizer):
    batch_size = x_0.shape[0]

    # Sample random timesteps
    t = torch.randint(1, timesteps, (batch_size,), device=x_0.device).long()

    # Sample random noise
    noise = torch.randn_like(x_0)

    # Get noisy samples
    x_t = get_noisy_samples(x_0, t, noise)

    # Predict noise
    predicted_noise = model(x_t, t)

    # Calculate loss
    loss = F.mse_loss(predicted_noise, noise)

    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

**Hyperparameters**
| Hyperparameter | Typical value |
|---|---|
| Timesteps (T) | ~1000 for training; fewer for sampling |
| Noise schedule | Linear, cosine, or quadratic |
| Batch size | 32 – 256 |
| Learning rate | ~1e-4 to 2e-5 |
| EMA decay | ~0.9999 |

**Optimization techniques**
- Careful weight initialization.
- Gradient clipping to prevent exploding gradients.
- Cosine learning-rate decay.
- Mixed-precision training for speed on modern GPUs.
- Exponential moving average (EMA) of model weights, which often improves results.

> **Training stability.** Diffusion models are generally more stable than GANs: they avoid mode collapse and converge more consistently.

**Conditional training**
- **Class-conditional** — feed the class label with the noisy image and timestep; convert labels to learned embeddings, combine with timestep embeddings, inject throughout the model.
- **Text-conditional** — encode prompts with a pre-trained encoder (CLIP), process through extra layers, inject via **cross-attention** so the model learns to match descriptions.
- **Classifier-free guidance** — train one model for both conditional and unconditional generation by randomly dropping the condition (c → ∅) during training. At sampling time extrapolate from the unconditional toward the conditional prediction with guidance weight w > 1:

  ε_θ^CFG(xₜ, c, t) = ε_θ(xₜ, ∅, t) + w · (ε_θ(xₜ, c, t) − ε_θ(xₜ, ∅, t))

## 7. Applications

Covered areas: text-to-image generation, image editing, video, audio, 3D content, and scientific uses such as drug discovery.

**Text-to-image walkthrough**
1. The user enters a text prompt.
2. A text encoder (typically CLIP) turns the prompt into embeddings.
3. The diffusion model uses those embeddings to condition generation.
4. Starting from random noise, the model denoises into an image matching the prompt.
5. Classifier-free guidance improves quality and prompt adherence.

**Model comparison**
| Model | Architecture | Key features | Strengths | Limitations |
|---|---|---|---|---|
| DALL-E 2 | CLIP + diffusion | Cascaded diffusion models; CLIP image embeddings | Strong photorealism and composition | Closed source, limited customization |
| Stable Diffusion | Latent diffusion | Works in compressed latent space; open source | Efficient, customizable, large community of extensions | Can be less coherent than DALL-E 2 |
| Midjourney | Proprietary diffusion | Focus on artistic quality; Discord interface | Strong aesthetics and artistic styles | Less control; Discord-only |
| Google Imagen | Cascaded diffusion | T5 text encoder; super-resolution diffusion | Strong text alignment, high resolution | Limited public access |

**Future directions**
- Multimodal models unifying text, image, audio, and video.
- Faster sampling with fewer steps.
- More precise control over generated content.
- Greater compute efficiency and accessibility.
- Specialized models for scientific, medical, and industrial domains.

## 8. Advanced Topics

- **Sampling** — DDIM (fewer steps, deterministic), DPM-Solver (higher-order ODE solvers), ancestral sampling (controlled randomness).
- **Latent diffusion** — a VAE compresses images, diffusion runs in latent space, a decoder reconstructs the image.
- **Control** — ControlNet (spatial conditioning), textual inversion (learning new concepts), LoRA (small trainable adapters for lightweight fine-tuning).
- **Cascaded models** — a low-resolution base model followed by upsampling models.
- **Video diffusion** — 3D U-Nets, transformers, latent video diffusion; challenges are cost and temporal consistency.
- **Theory** — links to score-based models and SDEs, and to variational inference / hierarchical VAEs.
- **Research frontiers** — efficiency, quality–speed trade-offs, multimodal generation, ethics.

## 9. Learning Resources

**Research papers**
| Category | Title | Authors | Year |
|---|---|---|---|
| Foundational | Deep Unsupervised Learning using Nonequilibrium Thermodynamics | Sohl-Dickstein et al. | 2015 |
| Core DDPM | Denoising Diffusion Probabilistic Models | Ho et al. | 2020 |
| Improvements | Improved Denoising Diffusion Probabilistic Models | Nichol & Dhariwal | 2021 |
| Latent space | High-Resolution Image Synthesis with Latent Diffusion Models | Rombach et al. | 2022 |
| Guidance | Classifier-Free Diffusion Guidance | Ho & Salimans | 2022 |

**Tutorials and courses**
- *Understanding Diffusion Models: A Unified Perspective* — Calvin Luo (mathematical background)
- *Diffusion Models from Scratch* — Hugging Face (code walkthrough)
- *Diffusion Models | Paper Explanation* — Yannic Kilcher (video)
- *Diffusion Models: A Practical Guide* — Google AI (interactive)
- *Generative AI with Diffusion Models* (full course)

**Code and implementation**
- `denoising-diffusion-pytorch` — Phil Wang (PyTorch)
- Stable Diffusion GitHub repository (production code)
- `score_sde` — Yang Song (JAX)
- Hugging Face *Diffusion Models Tutorial* (simplified demo)
- Colab notebooks for experimentation

**Learning path**
1. **Foundations** — probability, Markov chains, generative models, Gaussians, KL divergence.
2. **Core concepts** — forward and reverse diffusion; how noise is added and removed.
3. **Architecture** — U-Net design, timestep embedding, conditioning.
4. **Training and sampling** — objectives and sampling procedures; a small PyTorch/TensorFlow implementation.
5. **Advanced topics** — latent diffusion, classifier-free guidance, accelerated sampling.
6. **Applications and projects** — text-to-image generation or inpainting.

## 10. Conclusion

Diffusion models are stable to train, produce high-quality and controllable output, are theoretically grounded, and are architecturally flexible. Continued growth is expected across modalities and domains.

## Key concepts worth simulating

- **Forward diffusion**: an image (or 2-D point cloud) progressively noised over T steps with slider for t; show xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε and the closed-form jump to any t.
- **Noise schedules**: plot βₜ, αₜ and ᾱₜ for linear vs cosine vs quadratic; show signal-to-noise ratio √ᾱₜ/√(1−ᾱₜ) collapsing at different rates.
- **Reverse denoising**: animate the sampling loop from x_T to x₀ on a toy 2-D dataset (e.g. two moons / Swiss roll) with a small MLP ε_θ; toggle the σₜ·z term to see stochastic vs deterministic paths.
- **Noise-prediction training**: live loss curve of ‖ε − ε_θ(xₜ, t)‖² as the toy model trains; show predicted vs true noise vectors.
- **Sinusoidal timestep embedding**: heat-map of the embedding matrix over t and dimension; show how nearby timesteps get similar vectors.
- **Classifier-free guidance**: on a class-conditional toy dataset, slide w from 0 to 7 and watch samples sharpen toward the class mode (and over-saturate at high w).
- **DDIM step skipping**: compare 1000-step ancestral sampling with 50-step DDIM on the same noise seed.
- **Latent diffusion**: encoder → low-dim latent → diffusion → decoder, with pixel count 3×512×512 vs 4×64×64 shown as a compute ratio.

## Test code

```python
# 1. Noise schedules and the closed-form forward sample (numpy only)
import numpy as np

T = 1000
betas_linear = np.linspace(1e-4, 0.02, T)
s = 0.008
tt = np.arange(T + 1) / T
f = np.cos((tt + s) / (1 + s) * np.pi / 2) ** 2
alphas_bar_cos = f / f[0]
betas_cosine = np.clip(1 - alphas_bar_cos[1:] / alphas_bar_cos[:-1], 0, 0.999)

def alpha_bar(betas):
    return np.cumprod(1 - betas)

for name, b in [("linear", betas_linear), ("cosine", betas_cosine)]:
    ab = alpha_bar(b)
    print(f"{name:7s} alpha_bar at t=250/500/750: {ab[249]:.3f} {ab[499]:.3f} {ab[749]:.3f}")

# forward sample of a scalar x0 = 1.0 at t = 500 under the linear schedule
ab = alpha_bar(betas_linear)
x0, t = 1.0, 500
eps = np.random.randn()
x_t = np.sqrt(ab[t]) * x0 + np.sqrt(1 - ab[t]) * eps
print("x_t =", round(x_t, 3))
```

```python
# 2. Train a tiny DDPM on a 2-D toy dataset and sample from it (PyTorch)
import math, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
T = 200
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, 0)

def make_data(n=2048):           # two Gaussian blobs
    c = torch.randint(0, 2, (n, 1)).float() * 4 - 2
    return torch.cat([c, torch.zeros(n, 1)], 1) + 0.3 * torch.randn(n, 2)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half).float() / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)

class EpsNet(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + d, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
        self.d = d
    def forward(self, x, t):
        return self.net(torch.cat([x, timestep_embedding(t, self.d)], -1))

model = EpsNet(); opt = torch.optim.Adam(model.parameters(), 1e-3)
x0_all = make_data()
for step in range(1500):
    x0 = x0_all[torch.randint(0, len(x0_all), (256,))]
    t = torch.randint(0, T, (256,))
    noise = torch.randn_like(x0)
    ab = alphas_cumprod[t][:, None]
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    loss = F.mse_loss(model(x_t, t), noise)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(f"step {step} loss {loss.item():.4f}")

@torch.no_grad()
def sample(n=500):
    x = torch.randn(n, 2)
    for i in reversed(range(T)):
        t = torch.full((n,), i)
        eps = model(x, t)
        a, ab, b = alphas[i], alphas_cumprod[i], betas[i]
        z = torch.randn_like(x) if i > 0 else 0
        x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt() + b.sqrt() * z
    return x

xs = sample()
print("sampled x-means by cluster:", xs[xs[:, 0] < 0, 0].mean().item(), xs[xs[:, 0] > 0, 0].mean().item())
```

```python
# 3. Sinusoidal timestep embeddings: nearby timesteps are close, distant ones are not
import math, torch

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half).float() / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)

e = timestep_embedding(torch.tensor([10, 11, 500]), 32)
cos = torch.nn.functional.cosine_similarity
print("sim(10,11) =", round(cos(e[0:1], e[1:2]).item(), 3))
print("sim(10,500)=", round(cos(e[0:1], e[2:3]).item(), 3))
```

```python
# 4. Classifier-free guidance arithmetic on noise predictions
import torch

eps_uncond = torch.tensor([0.10, -0.20])   # eps_theta(x_t, None, t)
eps_cond   = torch.tensor([0.40,  0.10])   # eps_theta(x_t, c, t)
for w in [0.0, 1.0, 3.0, 7.5]:
    eps_cfg = eps_uncond + w * (eps_cond - eps_uncond)
    print(f"w={w:<4} eps_cfg={eps_cfg.tolist()}")
```
