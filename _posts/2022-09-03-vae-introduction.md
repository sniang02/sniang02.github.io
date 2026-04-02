---
layout: post
title: "An Introduction to Variational Autoencoders (VAE)"
date: 2026-04-03 10:00:00+0100
description: "A gentle introduction to Variational Autoencoders: theory, probabilistic foundations, and PyTorch implementation."
tags: [deep-learning, generative-models, VAE, PyTorch]
categories: [machine-learning]
giscus_comments: false
related_posts: true
toc:
  beginning: true
---

Variational Autoencoders (VAEs) are one of the most elegant frameworks in deep generative modelling. They combine probabilistic reasoning with neural networks to learn compact latent representations of data — and to generate new samples from those representations.

In this post, we build up the intuition, derive the mathematical objective, and implement a VAE from scratch in PyTorch.

---

## 1. The Big Picture

Given a dataset $$\mathcal{X} = \{x_1, \ldots, x_n\}$$, we want to learn a model that:

1. **Compresses** each observation $$x$$ into a low-dimensional latent variable $$z$$,
2. **Generates** new realistic samples by decoding $$z$$ back into data space.

A classical autoencoder does this deterministically. A VAE does it **probabilistically**: instead of mapping $$x$$ to a single point $$z$$, it maps $$x$$ to a **distribution** over $$z$$.

---

## 2. Probabilistic Setup

We assume the following generative model:

$$
z \sim p(z) = \mathcal{N}(0, I_d)
$$

$$
x \mid z \sim p_\theta(x \mid z)
$$

where $$\theta$$ are the parameters of a neural network (the **decoder**).

Our goal is to maximise the **marginal log-likelihood**:

$$
\log p_\theta(x) = \log \int p_\theta(x \mid z) \, p(z) \, dz
$$

This integral is **intractable** in general. We need an approximation.

---

## 3. The Evidence Lower Bound (ELBO)

We introduce an **approximate posterior** $$q_\phi(z \mid x)$$ — another neural network called the **encoder** — and use Jensen's inequality:

$$
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \mathrm{KL}\left( q_\phi(z \mid x) \,\|\, p(z) \right)
$$

This lower bound is called the **ELBO** (Evidence Lower BOund). It decomposes into:

- **Reconstruction term**: $$\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right]$$ — how well the decoder reconstructs $$x$$ from $$z$$.
- **Regularisation term**: $$-\mathrm{KL}\left( q_\phi(z \mid x) \,\|\, p(z) \right)$$ — how close the approximate posterior is to the prior.

We **maximise** the ELBO jointly over $$\theta$$ and $$\phi$$.

---

## 4. The Reparameterisation Trick

The encoder outputs a Gaussian approximate posterior:

$$
q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \, \mathrm{diag}(\sigma^2_\phi(x)))
$$

To backpropagate through the sampling step $$z \sim q_\phi(z \mid x)$$, we use the **reparameterisation trick**:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

This separates the stochastic part ($$\varepsilon$$) from the learnable parameters, making gradient computation possible.

---

## 5. The KL Divergence in Closed Form

When both distributions are Gaussian, the KL term has a closed-form expression. For a diagonal Gaussian approximate posterior with parameters $$\mu \in \mathbb{R}^d$$ and $$\sigma^2 \in \mathbb{R}^d$$:

$$
\mathrm{KL}\left( \mathcal{N}(\mu, \mathrm{diag}(\sigma^2)) \,\|\, \mathcal{N}(0, I) \right) = \frac{1}{2} \sum_{j=1}^{d} \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)
$$

This is efficient to compute and differentiable — no Monte Carlo approximation needed for this term.

---

## 6. PyTorch Implementation

Let us implement a VAE on the MNIST dataset.

### 6.1 Model Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
```

### 6.2 The ELBO Loss

```python
def elbo_loss(x_recon, x, mu, logvar):
    # Reconstruction term (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(
        x_recon, x.view(-1, 784), reduction='sum'
    )

    # KL divergence term (closed form)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss
```

### 6.3 Training Loop

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True,
                                transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Model & optimiser
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
def train(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()

        x_recon, mu, logvar = model(data)
        loss = elbo_loss(x_recon, data, mu, logvar)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader.dataset)
    print(f'Epoch {epoch}: Average loss = {avg_loss:.4f}')

for epoch in range(1, 21):
    train(model, train_loader, optimizer, epoch)
```

### 6.4 Generating New Samples

```python
def generate_samples(model, n_samples=64):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 20).to(device)
        samples = model.decode(z)
    return samples.view(n_samples, 1, 28, 28)
```

---

## 7. Summary

Here is a concise recap of the VAE framework:

| Component | Role | Parametrised by |
|---|---|---|
| Encoder $$q_\phi(z \mid x)$$ | Maps data to latent distribution | $$\phi$$ (neural network) |
| Decoder $$p_\theta(x \mid z)$$ | Maps latent code to data | $$\theta$$ (neural network) |
| Prior $$p(z)$$ | Regularises the latent space | Fixed: $$\mathcal{N}(0, I)$$ |
| ELBO | Tractable objective to maximise | — |

The key insight is that the VAE learns a **structured latent space** where interpolation and generation are meaningful — unlike classical autoencoders.

---

## 8. Going Further

VAEs are the foundation of many powerful models, including:

- **$$\beta$$-VAE**: adds a weight on the KL term to enforce disentanglement.
- **Conditional VAE (CVAE)**: conditions generation on a label or context.
- **Graph VAE (VGAE)**: extends VAEs to graph-structured data — the basis of my own research on network clustering.

I will cover these extensions in future posts. Stay tuned!

---

*If you have questions or comments, feel free to reach out by email.*
