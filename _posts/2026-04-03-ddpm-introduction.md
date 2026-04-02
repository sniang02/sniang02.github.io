---
layout: post
title: a post with formatting and links
date: 2015-03-15 16:40:16
description: march & april, looking forward to summer
tags: formatting links
categories: sample-posts
---

Denoising Diffusion Probabilistic Models (DDPMs) have revolutionised generative modelling, producing state-of-the-art results in image synthesis, audio generation, and beyond. In this post, we build the theory from scratch and implement a DDPM on the MNIST dataset.

---

## 1. The Core Idea

A diffusion model works in two phases:

- **Forward process**: progressively add Gaussian noise to a data sample $$x_0$$ over $$T$$ steps until it becomes pure noise $$x_T \sim \mathcal{N}(0, I)$$.
- **Reverse process**: learn to denoise step by step, going from $$x_T$$ back to $$x_0$$, effectively learning to generate data.

The key insight is that if we can learn to reverse the noise process, we can **generate new samples** by starting from random noise and denoising iteratively.

---

## 2. The Forward Process

Given a data point $$x_0 \sim q(x_0)$$, the forward process gradually adds Gaussian noise:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t I\right)
$$

where $$\{\beta_t\}_{t=1}^T$$ is a **variance schedule** — a sequence of small positive constants.

### 2.1 Closed-form Sampling at Any Timestep

Using $$\alpha_t = 1 - \beta_t$$ and $$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$, we can sample $$x_t$$ directly from $$x_0$$:

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t) I\right)
$$

which means:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

This is extremely useful for training: we can corrupt any sample $$x_0$$ to any noise level $$t$$ in a **single step**.

---

## 3. The Reverse Process

The reverse process is defined as:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\right)
$$

where $$\mu_\theta$$ and $$\Sigma_\theta$$ are parameterised by a neural network.

### 3.1 What Does the Network Actually Predict?

Rather than predicting $$\mu_\theta$$ directly, Ho et al. (2020) showed it is more effective to train a network $$\varepsilon_\theta$$ to **predict the noise** $$\varepsilon$$ added at step $$t$$. The mean is then recovered as:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right)
$$

---

## 4. The Training Objective

The full ELBO simplifies to a remarkably clean objective:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon} \left[ \left\| \varepsilon - \varepsilon_\theta\left(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon,\, t\right) \right\|^2 \right]
$$

In words: **given a noisy image at step $$t$$, predict the noise that was added**.

This is simply a **noise prediction regression problem** — elegant and easy to optimise.

---

## 5. Sampling (Inference)

Once trained, we generate new samples with the following iterative procedure:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right) + \sqrt{\beta_t}\, z
$$

where $$z \sim \mathcal{N}(0, I)$$ if $$t > 1$$, else $$z = 0$$.

Starting from $$x_T \sim \mathcal{N}(0, I)$$, we iteratively apply this update for $$t = T, T-1, \ldots, 1$$ to obtain a generated sample $$x_0$$.

---

## 6. PyTorch Implementation on MNIST

### 6.1 Noise Schedule

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear variance schedule from Ho et al. (2020)."""
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    """Precompute all quantities needed for training and sampling."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # For reverse process
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance,
    }


T = 1000
betas = linear_beta_schedule(T)
schedule = precompute_schedule(betas)
```

### 6.2 Forward Process (Adding Noise)

```python
def extract(a, t, x_shape):
    """Extract values from a 1D tensor at indices t."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_0, t, schedule, noise=None):
    """Sample x_t from x_0 at timestep t (forward process)."""
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = extract(
        schedule['sqrt_alphas_cumprod'], t, x_0.shape
    )
    sqrt_one_minus_alphas_cumprod_t = extract(
        schedule['sqrt_one_minus_alphas_cumprod'], t, x_0.shape
    )

    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
```

### 6.3 The U-Net Denoiser

We use a simplified U-Net as the noise prediction network $$\varepsilon_\theta$$. The timestep $$t$$ is injected via sinusoidal embeddings.

```python
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

    def forward(self, x, t_emb):
        h = self.norm1(F.relu(self.conv1(x)))
        h = h + self.time_mlp(F.relu(t_emb))[:, :, None, None]
        h = self.norm2(F.relu(self.conv2(h)))
        return h


class SimpleUNet(nn.Module):
    """Lightweight U-Net for MNIST (28x28 images)."""

    def __init__(self, in_channels=1, time_emb_dim=32):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Encoder
        self.enc1 = Block(in_channels, 32, time_emb_dim)
        self.enc2 = Block(32, 64, time_emb_dim)
        self.enc3 = Block(64, 128, time_emb_dim)

        # Bottleneck
        self.bottleneck = Block(128, 128, time_emb_dim)

        # Decoder
        self.dec3 = Block(256, 64, time_emb_dim)
        self.dec2 = Block(128, 32, time_emb_dim)
        self.dec1 = Block(64, 32, time_emb_dim)

        self.out = nn.Conv2d(32, in_channels, 1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)

        # Bottleneck
        b = self.bottleneck(self.pool(e3), t_emb)

        # Decoder with skip connections
        d3 = self.dec3(
            torch.cat([self.up(b), e3], dim=1), t_emb
        )
        d2 = self.dec2(
            torch.cat([self.up(d3), e2], dim=1), t_emb
        )
        d1 = self.dec1(
            torch.cat([self.up(d2), e1], dim=1), t_emb
        )

        return self.out(d1)
```

### 6.4 Training

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 2) - 1)  # Scale to [-1, 1]
])
dataset = datasets.MNIST('./data', train=True, download=True,
                          transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Move schedule to device
for key in schedule:
    schedule[key] = schedule[key].to(device)


def p_losses(model, x_0, t, schedule):
    """Compute the simplified DDPM loss."""
    noise = torch.randn_like(x_0)
    x_noisy = q_sample(x_0, t, schedule, noise=noise)
    predicted_noise = model(x_noisy, t)
    return F.mse_loss(noise, predicted_noise)


# Training loop
for epoch in range(1, 51):
    model.train()
    total_loss = 0.0
    for batch, _ in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Sample random timesteps
        t = torch.randint(0, T, (batch.shape[0],), device=device).long()

        loss = p_losses(model, batch, t, schedule)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        avg = total_loss / len(loader)
        print(f"Epoch {epoch:3d} | Loss: {avg:.4f}")
```

### 6.5 Generating Samples

```python
@torch.no_grad()
def p_sample(model, x, t, t_index, schedule):
    """One step of the reverse process."""
    betas_t = extract(schedule['betas'], t, x.shape)
    sqrt_one_minus_t = extract(
        schedule['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = extract(
        1.0 / torch.sqrt(schedule['alphas']), t, x.shape
    )

    # Predict mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_var_t = extract(
            schedule['posterior_variance'], t, x.shape
        )
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise


@torch.no_grad()
def sample(model, schedule, image_size=28, n_samples=16, channels=1):
    """Full reverse process: generate samples from noise."""
    device = next(model.parameters()).device
    shape = (n_samples, channels, image_size, image_size)

    # Start from pure noise
    img = torch.randn(shape, device=device)

    for i in reversed(range(T)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, schedule)

    # Rescale to [0, 1]
    img = (img.clamp(-1, 1) + 1) / 2
    return img


# Generate 16 samples
model.eval()
generated = sample(model, schedule, n_samples=16)
print(f"Generated samples shape: {generated.shape}")
# torch.Size([16, 1, 28, 28])
```

---

## 7. Summary

| | VAE | DDPM |
|---|---|---|
| **Latent space** | Explicit, low-dimensional | Implicit (noise levels) |
| **Training** | ELBO (reconstruction + KL) | Noise prediction (MSE) |
| **Sampling** | Single forward pass | $$T$$ iterative steps |
| **Sample quality** | Good | Excellent |
| **Speed** | Fast | Slow ($$T$$ steps) |

DDPMs produce higher quality samples than VAEs but at the cost of a slower sampling procedure ($$T = 1000$$ steps by default). Recent work (DDIM, score-based models) has significantly reduced this cost.

---

## 8. Connection to My Research

In my paper presented at **JDS 2024**, we extended the DDPM framework by introducing a **conditional version** that accounts for cluster membership of images. The model assumes images are distributed into $$Q$$ clusters, and the denoising network is conditioned on cluster assignments. Inference is performed using a variational EM-type algorithm.

This shows how diffusion models can be turned into **clustering tools** — not just generative models.

---

## 9. Going Further

- **DDIM** (Song et al., 2020): deterministic sampling in far fewer steps.
- **Score-based models**: a unified view of diffusion through score matching.
- **Classifier-free guidance**: condition generation on labels without a separate classifier.
- **Latent Diffusion Models (LDM)**: run diffusion in the latent space of a VAE — the architecture behind Stable Diffusion.

---

*Questions or comments? Feel free to reach out by email.*
