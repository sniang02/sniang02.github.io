---
layout: post
title: "The Complete Mathematical Background for Artificial Intelligence"
date: 2026-04-03 14:00:00+0100
description: "A comprehensive reference covering linear algebra, analysis, probability theory, statistics, Bayesian methods, and Markov chains — everything you need for modern AI and machine learning."
tags: [mathematics, linear-algebra, probability, statistics, Bayesian, Markov-chains, AI]
categories: [mathematics]
giscus_comments: false
related_posts: true
toc:
  beginning: true
---

This post is a self-contained mathematical reference for artificial intelligence and machine learning. It covers everything from the foundations of linear algebra and analysis to Bayesian inference and Markov chain theory. It is intended as a reminder of the key results and intuitions accumulated through years of study — from undergraduate courses to doctoral research.

---

# Part I — Linear Algebra

## 1. Vector Spaces

A **vector space** $$V$$ over a field $$\mathbb{F}$$ (usually $$\mathbb{R}$$ or $$\mathbb{C}$$) is a set equipped with addition and scalar multiplication satisfying the usual axioms (associativity, commutativity, distributivity, identity, inverse).

**Subspace**: a subset $$W \subseteq V$$ closed under addition and scalar multiplication.

**Span**: $$\text{span}(v_1, \ldots, v_k) = \{ \sum_{i=1}^k \lambda_i v_i : \lambda_i \in \mathbb{F} \}$$.

**Linear independence**: $$\{v_1, \ldots, v_k\}$$ is linearly independent if $$\sum_{i=1}^k \lambda_i v_i = 0 \Rightarrow \lambda_i = 0$$ for all $$i$$.

**Basis**: a linearly independent spanning set. All bases of a finite-dimensional space have the same cardinality, called the **dimension** $$\dim(V)$$.

---

## 2. Matrices

A matrix $$A \in \mathbb{R}^{m \times n}$$ represents a linear map $$A : \mathbb{R}^n \to \mathbb{R}^m$$.

### 2.1 Key Operations

- **Transpose**: $$(A^\top)_{ij} = A_{ji}$$
- **Trace**: $$\text{tr}(A) = \sum_i A_{ii}$$ (square matrices only)
- **Matrix product**: $$(AB)_{ij} = \sum_k A_{ik} B_{kj}$$
- **Frobenius norm**: $$\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(A^\top A)}$$

### 2.2 Special Matrices

| Name | Property |
|------|----------|
| Symmetric | $$A = A^\top$$ |
| Orthogonal | $$A^\top A = I$$ |
| Positive definite | $$x^\top A x > 0$$ for all $$x \neq 0$$ |
| Positive semi-definite | $$x^\top A x \geq 0$$ for all $$x$$ |
| Diagonal | $$A_{ij} = 0$$ for $$i \neq j$$ |

### 2.3 Rank, Kernel, Image

For $$A \in \mathbb{R}^{m \times n}$$:
- **Kernel** (null space): $$\ker(A) = \{x \in \mathbb{R}^n : Ax = 0\}$$
- **Image** (column space): $$\text{Im}(A) = \{Ax : x \in \mathbb{R}^n\}$$
- **Rank-nullity theorem**: $$\text{rank}(A) + \text{nullity}(A) = n$$

---

## 3. Eigenvalues and Eigenvectors

A scalar $$\lambda \in \mathbb{C}$$ is an **eigenvalue** of $$A \in \mathbb{R}^{n \times n}$$ if there exists a non-zero vector $$v$$ such that:

$$
Av = \lambda v
$$

The set of all eigenvalues is the **spectrum** $$\sigma(A)$$. Eigenvalues satisfy the **characteristic polynomial**:

$$
\det(A - \lambda I) = 0
$$

**Key properties**:
- $$\text{tr}(A) = \sum_i \lambda_i$$
- $$\det(A) = \prod_i \lambda_i$$
- Symmetric matrices have real eigenvalues
- Symmetric matrices have orthogonal eigenvectors corresponding to distinct eigenvalues

---

## 4. Matrix Decompositions

### 4.1 Eigendecomposition

If $$A \in \mathbb{R}^{n \times n}$$ is diagonalisable:

$$
A = P \Lambda P^{-1}
$$

where $$P$$ contains eigenvectors as columns and $$\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$$.

For symmetric matrices: $$A = Q \Lambda Q^\top$$ with $$Q$$ orthogonal (**spectral theorem**).

### 4.2 Singular Value Decomposition (SVD)

For any $$A \in \mathbb{R}^{m \times n}$$:

$$
A = U \Sigma V^\top
$$

where:
- $$U \in \mathbb{R}^{m \times m}$$ is orthogonal (left singular vectors)
- $$\Sigma \in \mathbb{R}^{m \times n}$$ is diagonal with non-negative entries $$\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$$ (singular values)
- $$V \in \mathbb{R}^{n \times n}$$ is orthogonal (right singular vectors)

**Applications in ML**:
- **PCA**: the principal components are the right singular vectors of the centred data matrix
- **Low-rank approximation**: the best rank-$$k$$ approximation of $$A$$ is $$A_k = U_k \Sigma_k V_k^\top$$ (Eckart-Young theorem)
- **Pseudo-inverse**: $$A^+ = V \Sigma^+ U^\top$$

### 4.3 Cholesky Decomposition

For a symmetric positive definite matrix $$A$$:

$$
A = LL^\top
$$

where $$L$$ is lower triangular. Essential for efficient sampling from multivariate Gaussians.

### 4.4 QR Decomposition

$$
A = QR
$$

with $$Q$$ orthogonal and $$R$$ upper triangular. Used for solving least squares problems.

---

## 5. Norms and Inner Products

An **inner product** on $$V$$ satisfies: symmetry, linearity, and positive definiteness.

The standard inner product on $$\mathbb{R}^n$$: $$\langle x, y \rangle = x^\top y = \sum_i x_i y_i$$.

A **norm** satisfies: non-negativity, homogeneity, triangle inequality.

Common norms:
- $$\ell_1$$: $$\|x\|_1 = \sum_i |x_i|$$
- $$\ell_2$$: $$\|x\|_2 = \sqrt{\sum_i x_i^2}$$
- $$\ell_\infty$$: $$\|x\|_\infty = \max_i |x_i|$$
- $$\ell_p$$: $$\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$$

**Cauchy-Schwarz inequality**: $$|\langle x, y \rangle| \leq \|x\| \cdot \|y\|$$

---

## 6. Quadratic Forms and Positive Definiteness

A **quadratic form** is $$f(x) = x^\top A x$$ for symmetric $$A$$.

$$A$$ is **positive definite** ($$A \succ 0$$) iff:
1. All eigenvalues are strictly positive
2. All leading principal minors are positive (Sylvester's criterion)
3. $$A = B^\top B$$ for some non-singular $$B$$

This is central in probability: the covariance matrix of a multivariate Gaussian is always positive semi-definite.

---

# Part II — Analysis and Calculus

## 7. Differential Calculus

### 7.1 Derivatives and Gradients

For $$f : \mathbb{R}^n \to \mathbb{R}$$, the **gradient** is:

$$
\nabla f(x) = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)^\top \in \mathbb{R}^n
$$

The gradient points in the direction of steepest ascent.

### 7.2 Jacobian

For $$f : \mathbb{R}^n \to \mathbb{R}^m$$, the **Jacobian** is:

$$
J_f(x) = \frac{\partial f}{\partial x} \in \mathbb{R}^{m \times n}, \quad (J_f)_{ij} = \frac{\partial f_i}{\partial x_j}
$$

### 7.3 Hessian

For $$f : \mathbb{R}^n \to \mathbb{R}$$, the **Hessian** is:

$$
H_f(x) = \nabla^2 f(x) \in \mathbb{R}^{n \times n}, \quad (H_f)_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

The Hessian captures the local curvature:
- $$H \succ 0$$: local minimum
- $$H \prec 0$$: local maximum
- $$H$$ indefinite: saddle point

### 7.4 Chain Rule

For $$h = f \circ g$$:

$$
\frac{\partial h}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}
$$

In matrix form: $$J_h = J_f \cdot J_g$$. This is the foundation of **backpropagation**.

### 7.5 Taylor Expansion

For $$f : \mathbb{R}^n \to \mathbb{R}$$:

$$
f(x + \delta) \approx f(x) + \nabla f(x)^\top \delta + \frac{1}{2} \delta^\top H_f(x) \delta + O(\|\delta\|^3)
$$

---

## 8. Optimisation

### 8.1 Gradient Descent

To minimise $$f$$, update:

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

where $$\eta > 0$$ is the **learning rate**.

**Stochastic Gradient Descent (SGD)**: use a mini-batch $$\mathcal{B} \subset \{1, \ldots, n\}$$:

$$
x_{t+1} = x_t - \eta \cdot \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla f_i(x_t)
$$

### 8.2 Momentum and Adam

**Momentum**:
$$
v_{t+1} = \beta v_t + \nabla f(x_t), \quad x_{t+1} = x_t - \eta v_{t+1}
$$

**Adam** (Kingma & Ba, 2015):
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
x_{t+1} = x_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

with bias-corrected estimates $$\hat{m}_t = m_t / (1 - \beta_1^t)$$ and $$\hat{v}_t = v_t / (1-\beta_2^t)$$.

### 8.3 Convexity

A function $$f$$ is **convex** if:

$$
f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y) \quad \forall \lambda \in [0,1]
$$

For differentiable $$f$$: $$f$$ is convex iff $$H_f(x) \succeq 0$$ everywhere.

For convex functions, **any local minimum is a global minimum**.

### 8.4 Lagrange Multipliers

To minimise $$f(x)$$ subject to $$g(x) = 0$$, introduce the **Lagrangian**:

$$
\mathcal{L}(x, \lambda) = f(x) + \lambda^\top g(x)
$$

At optimality: $$\nabla_x \mathcal{L} = 0$$ and $$\nabla_\lambda \mathcal{L} = 0$$.

For inequality constraints $$g(x) \leq 0$$: **KKT conditions**.

---

## 9. Integration

### 9.1 Riemann and Lebesgue Integrals

The **Lebesgue integral** extends the Riemann integral to a much broader class of functions and is the foundation of modern probability theory.

### 9.2 Change of Variables

For a bijective differentiable map $$y = g(x)$$:

$$
\int_A f(g(x)) |\det J_g(x)| \, dx = \int_{g(A)} f(y) \, dy
$$

The **Jacobian determinant** $$|\det J_g|$$ accounts for the volume distortion. This is fundamental in normalising flows and change-of-variable formulas for probability densities.

### 9.3 Fubini's Theorem

For measurable $$f : \mathbb{R}^m \times \mathbb{R}^n \to \mathbb{R}$$:

$$
\int_{\mathbb{R}^{m+n}} f(x, y) \, d(x,y) = \int_{\mathbb{R}^m} \left( \int_{\mathbb{R}^n} f(x,y) \, dy \right) dx
$$

This justifies swapping the order of integration — essential for marginalisation in probability.

---

# Part III — Probability Theory

## 10. Probability Spaces

A **probability space** is a triple $$(\Omega, \mathcal{F}, \mathbb{P})$$ where:
- $$\Omega$$ is the sample space
- $$\mathcal{F}$$ is a $$\sigma$$-algebra of events
- $$\mathbb{P} : \mathcal{F} \to [0,1]$$ is a probability measure satisfying $$\mathbb{P}(\Omega) = 1$$

**Kolmogorov axioms**:
1. $$\mathbb{P}(A) \geq 0$$ for all $$A \in \mathcal{F}$$
2. $$\mathbb{P}(\Omega) = 1$$
3. Countable additivity: for pairwise disjoint events, $$\mathbb{P}(\bigcup_i A_i) = \sum_i \mathbb{P}(A_i)$$

---

## 11. Random Variables

A **random variable** is a measurable function $$X : \Omega \to \mathbb{R}$$.

### 11.1 Distribution and Density

The **cumulative distribution function (CDF)**:

$$
F_X(x) = \mathbb{P}(X \leq x)
$$

If $$X$$ is continuous, it admits a **probability density function (pdf)** $$f_X$$ such that:

$$
F_X(x) = \int_{-\infty}^x f_X(t) \, dt, \quad \int_{-\infty}^{+\infty} f_X(x) \, dx = 1
$$

### 11.2 Expectation and Variance

$$
\mathbb{E}[X] = \int x \, f_X(x) \, dx
$$

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**Linearity of expectation**: $$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$.

### 11.3 Covariance and Correlation

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

$$
\rho(X, Y) = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}} \in [-1, 1]
$$

For a random vector $$X \in \mathbb{R}^d$$, the **covariance matrix** is:

$$
\Sigma = \text{Cov}(X) = \mathbb{E}[(X - \mu)(X - \mu)^\top] \in \mathbb{R}^{d \times d}
$$

$$\Sigma$$ is always symmetric positive semi-definite.

---

## 12. Common Distributions

### 12.1 Discrete Distributions

**Bernoulli** $$\text{Ber}(p)$$: $$\mathbb{P}(X=1) = p$$, $$\mathbb{P}(X=0) = 1-p$$

$$\mathbb{E}[X] = p, \quad \text{Var}(X) = p(1-p)$$

**Binomial** $$\mathcal{B}(n, p)$$: $$\mathbb{P}(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$

$$\mathbb{E}[X] = np, \quad \text{Var}(X) = np(1-p)$$

**Poisson** $$\mathcal{P}(\lambda)$$: $$\mathbb{P}(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

$$\mathbb{E}[X] = \text{Var}(X) = \lambda$$

**Categorical** $$\text{Cat}(\pi)$$ with $$\pi \in \Delta_K$$: $$\mathbb{P}(X=k) = \pi_k$$

**Multinomial** $$\text{Mult}(n, \pi)$$: generalisation of Binomial to $$K$$ categories.

### 12.2 Continuous Distributions

**Uniform** $$\mathcal{U}(a,b)$$: $$f(x) = \frac{1}{b-a}$$ on $$[a,b]$$

**Gaussian** $$\mathcal{N}(\mu, \sigma^2)$$:

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

$$\mathbb{E}[X] = \mu, \quad \text{Var}(X) = \sigma^2$$

**Exponential** $$\text{Exp}(\lambda)$$: $$f(x) = \lambda e^{-\lambda x}$$ for $$x \geq 0$$

$$\mathbb{E}[X] = 1/\lambda, \quad \text{Var}(X) = 1/\lambda^2$$

**Beta** $$\text{Beta}(\alpha, \beta)$$:

$$
f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \quad x \in [0,1]
$$

**Dirichlet** $$\text{Dir}(\alpha)$$ with $$\alpha \in \mathbb{R}_{>0}^K$$: generalisation of Beta to simplices. The natural conjugate prior for the Categorical/Multinomial.

**Gamma** $$\Gamma(\alpha, \beta)$$:

$$
f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0
$$

**Laplace** $$\text{Lap}(\mu, b)$$: $$f(x) = \frac{1}{2b} e^{-|x-\mu|/b}$$. Heavy-tailed; promotes sparsity (Lasso prior).

---

## 13. Multivariate Gaussian Distribution

The **multivariate Gaussian** $$\mathcal{N}(\mu, \Sigma)$$ with $$\mu \in \mathbb{R}^d$$, $$\Sigma \succ 0$$:

$$
f(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)
$$

### Key properties

**Marginalisation**: if $$X = (X_1, X_2)$$ is jointly Gaussian, then $$X_1$$ and $$X_2$$ are marginally Gaussian.

**Conditioning**: the conditional distribution $$X_1 \mid X_2 = x_2$$ is also Gaussian:

$$
X_1 \mid X_2 = x_2 \sim \mathcal{N}\left(\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2),\, \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\right)
$$

**Linear transformations**: if $$X \sim \mathcal{N}(\mu, \Sigma)$$, then $$AX + b \sim \mathcal{N}(A\mu + b, A\Sigma A^\top)$$.

**Sampling via Cholesky**: to sample $$X \sim \mathcal{N}(\mu, \Sigma)$$, compute $$L = \text{chol}(\Sigma)$$ and set $$X = \mu + L\varepsilon$$ with $$\varepsilon \sim \mathcal{N}(0, I)$$.

---

## 14. Convergence of Random Variables

Let $$X_1, X_2, \ldots$$ be a sequence of random variables.

### 14.1 Almost Sure Convergence

$$X_n \xrightarrow{a.s.} X$$ if $$\mathbb{P}(\lim_{n \to \infty} X_n = X) = 1$$

### 14.2 Convergence in Probability

$$X_n \xrightarrow{p} X$$ if for all $$\varepsilon > 0$$: $$\lim_{n \to \infty} \mathbb{P}(|X_n - X| > \varepsilon) = 0$$

### 14.3 Convergence in Distribution

$$X_n \xrightarrow{d} X$$ if $$F_{X_n}(x) \to F_X(x)$$ at all continuity points of $$F_X$$.

### 14.4 Law of Large Numbers (LLN)

**Weak LLN**: for i.i.d. $$X_i$$ with $$\mathbb{E}[X_i] = \mu$$:

$$
\frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{p} \mu
$$

**Strong LLN**: the convergence holds almost surely.

### 14.5 Central Limit Theorem (CLT)

For i.i.d. $$X_i$$ with $$\mathbb{E}[X_i] = \mu$$, $$\text{Var}(X_i) = \sigma^2 < \infty$$:

$$
\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0,1)
$$

The CLT justifies the ubiquity of Gaussian distributions in statistics and machine learning.

---

## 15. Conditional Probability and Independence

### 15.1 Conditional Probability

$$
\mathbb{P}(A \mid B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}, \quad \mathbb{P}(B) > 0
$$

### 15.2 Independence

Events $$A$$ and $$B$$ are **independent** if $$\mathbb{P}(A \cap B) = \mathbb{P}(A)\mathbb{P}(B)$$.

Random variables $$X$$ and $$Y$$ are independent if their joint density factorises: $$f_{X,Y}(x,y) = f_X(x) f_Y(y)$$.

### 15.3 Conditional Independence

$$X \perp\!\!\!\perp Y \mid Z$$ means $$f_{X,Y|Z}(x,y|z) = f_{X|Z}(x|z) f_{Y|Z}(y|z)$$.

This is foundational in graphical models and latent variable models.

### 15.4 Total Probability and Bayes

**Law of total probability**:

$$
\mathbb{P}(A) = \sum_i \mathbb{P}(A \mid B_i) \mathbb{P}(B_i)
$$

**Bayes' theorem**:

$$
\mathbb{P}(B \mid A) = \frac{\mathbb{P}(A \mid B) \mathbb{P}(B)}{\mathbb{P}(A)}
$$

For continuous variables:

$$
p(\theta \mid x) = \frac{p(x \mid \theta) p(\theta)}{p(x)} = \frac{p(x \mid \theta) p(\theta)}{\int p(x \mid \theta) p(\theta) \, d\theta}
$$

---

# Part IV — Information Theory

## 16. Entropy

The **Shannon entropy** of a discrete distribution $$p$$:

$$
H(X) = -\sum_x p(x) \log p(x)
$$

For a continuous distribution (**differential entropy**):

$$
h(X) = -\int f(x) \log f(x) \, dx
$$

Entropy measures the average uncertainty. The Gaussian maximises differential entropy among all distributions with fixed variance.

---

## 17. KL Divergence

The **Kullback-Leibler (KL) divergence** from $$q$$ to $$p$$:

$$
\mathrm{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

**Properties**:
- $$\mathrm{KL}(p \| q) \geq 0$$ with equality iff $$p = q$$ (**Gibbs' inequality**)
- **Asymmetric**: $$\mathrm{KL}(p \| q) \neq \mathrm{KL}(q \| p)$$ in general
- Not a metric distance

The KL divergence is central to variational inference, VAEs, and information-theoretic analyses of learning algorithms.

### KL Between Two Gaussians

$$
\mathrm{KL}\left(\mathcal{N}(\mu_1, \Sigma_1) \| \mathcal{N}(\mu_2, \Sigma_2)\right) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_1 - \mu_2)^\top \Sigma_2^{-1}(\mu_1 - \mu_2)\right]
$$

For the VAE, with $$\Sigma_1 = \text{diag}(\sigma^2)$$ and $$\Sigma_2 = I$$:

$$
\mathrm{KL}\left(\mathcal{N}(\mu, \text{diag}(\sigma^2)) \| \mathcal{N}(0, I)\right) = \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)
$$

---

## 18. Mutual Information

$$
I(X; Y) = \mathrm{KL}(p_{X,Y} \| p_X \otimes p_Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)
$$

Mutual information measures the amount of information shared between two variables. It is symmetric and equals zero iff $$X \perp\!\!\!\perp Y$$.

---

# Part V — Statistics

## 19. Statistical Inference

We observe data $$x_1, \ldots, x_n$$ i.i.d. from an unknown distribution $$p_\theta$$ and want to estimate $$\theta$$.

### 19.1 Point Estimation

An **estimator** $$\hat{\theta} = \hat{\theta}(x_1, \ldots, x_n)$$ is a function of the data.

- **Bias**: $$\text{bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$$
- **Variance**: $$\text{Var}(\hat{\theta})$$
- **Mean Squared Error**: $$\text{MSE}(\hat{\theta}) = \text{bias}^2 + \text{Var}(\hat{\theta})$$
- **Consistency**: $$\hat{\theta} \xrightarrow{p} \theta$$ as $$n \to \infty$$

### 19.2 Maximum Likelihood Estimation (MLE)

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \, \mathcal{L}(\theta; x) = \arg\max_\theta \sum_{i=1}^n \log p_\theta(x_i)
$$

**Properties of MLE** (under regularity conditions):
- **Consistent**: $$\hat{\theta}_{\text{MLE}} \xrightarrow{p} \theta^*$$
- **Asymptotically normal**: $$\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta^*) \xrightarrow{d} \mathcal{N}(0, I(\theta^*)^{-1})$$
- **Asymptotically efficient**: achieves the Cramér-Rao lower bound

### 19.3 Fisher Information

The **Fisher information** matrix:

$$
I(\theta) = \mathbb{E}\left[\nabla_\theta \log p_\theta(X) \, \nabla_\theta \log p_\theta(X)^\top\right] = -\mathbb{E}\left[\nabla_\theta^2 \log p_\theta(X)\right]
$$

**Cramér-Rao bound**: for any unbiased estimator $$\hat{\theta}$$:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

---

## 20. Exponential Families

A distribution belongs to the **exponential family** if its density can be written as:

$$
p_\theta(x) = h(x) \exp\left(\eta(\theta)^\top T(x) - A(\theta)\right)
$$

where:
- $$T(x)$$: sufficient statistic
- $$\eta(\theta)$$: natural parameters
- $$A(\theta)$$: log-partition function (ensures normalisation)
- $$h(x)$$: base measure

**Examples**: Gaussian, Bernoulli, Poisson, Gamma, Beta, Dirichlet — all exponential family members.

**Key property**: the MLE for exponential families sets the expected sufficient statistics equal to the empirical sufficient statistics:

$$
\mathbb{E}_{\hat{\theta}}[T(X)] = \frac{1}{n}\sum_{i=1}^n T(x_i)
$$

This property is exploited extensively in the **EM algorithm**.

---

## 21. Hypothesis Testing

We test a **null hypothesis** $$H_0$$ against an **alternative** $$H_1$$.

- **Type I error** ($$\alpha$$): rejecting $$H_0$$ when it is true (false positive)
- **Type II error** ($$\beta$$): accepting $$H_0$$ when $$H_1$$ is true (false negative)
- **Power**: $$1 - \beta$$, the probability of correctly rejecting $$H_0$$

**p-value**: the probability of observing a test statistic as extreme as the one observed, under $$H_0$$. Reject $$H_0$$ if p-value $$< \alpha$$.

---

## 22. Confidence Intervals

A $$95\%$$ **confidence interval** $$[L(X), U(X)]$$ satisfies:

$$
\mathbb{P}(L(X) \leq \theta \leq U(X)) = 0.95
$$

For a Gaussian mean with known variance: $$\bar{x} \pm 1.96 \sigma/\sqrt{n}$$.

---

## 23. Linear Regression

Given $$(x_i, y_i)_{i=1}^n$$ with $$x_i \in \mathbb{R}^p$$, $$y_i \in \mathbb{R}$$:

$$
y = X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

**Ordinary Least Squares (OLS)**:

$$
\hat{\beta} = (X^\top X)^{-1} X^\top y
$$

**Gauss-Markov theorem**: OLS is the Best Linear Unbiased Estimator (BLUE).

**Ridge regression** (L2 regularisation):

$$
\hat{\beta}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
$$

**Lasso** (L1 regularisation): promotes sparsity in $$\hat{\beta}$$.

---

# Part VI — Bayesian Statistics

## 24. The Bayesian Framework

Bayesian statistics treats all unknown quantities as random variables with probability distributions.

- **Prior**: $$p(\theta)$$ — beliefs about $$\theta$$ before seeing data
- **Likelihood**: $$p(x \mid \theta)$$ — the data-generating process
- **Posterior**: $$p(\theta \mid x) \propto p(x \mid \theta) p(\theta)$$ — updated beliefs
- **Marginal likelihood** (evidence): $$p(x) = \int p(x \mid \theta) p(\theta) \, d\theta$$

### Bayesian Inference Steps

1. Choose prior $$p(\theta)$$
2. Specify likelihood $$p(x \mid \theta)$$
3. Compute posterior $$p(\theta \mid x)$$
4. Make predictions: $$p(x_{\text{new}} \mid x) = \int p(x_{\text{new}} \mid \theta) p(\theta \mid x) \, d\theta$$

---

## 25. Conjugate Priors

A prior $$p(\theta)$$ is **conjugate** to the likelihood $$p(x \mid \theta)$$ if the posterior $$p(\theta \mid x)$$ belongs to the same distributional family as the prior.

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Bernoulli$$(\theta)$$ | Beta$$(\alpha, \beta)$$ | Beta$$(\alpha + n_1, \beta + n_0)$$ |
| Poisson$$(\lambda)$$ | Gamma$$(\alpha, \beta)$$ | Gamma$$(\alpha + \sum x_i, \beta + n)$$ |
| Gaussian$$(\mu, \sigma^2)$$ | Gaussian$$(\mu_0, \sigma_0^2)$$ | Gaussian (closed form) |
| Categorical$$(\pi)$$ | Dirichlet$$(\alpha)$$ | Dirichlet$$(\alpha + n)$$ |
| Gaussian$$(\mu, \Sigma)$$ | Wishart | Wishart |

Conjugate priors allow **closed-form Bayesian updates** — essential for tractable inference.

---

## 26. Bayesian Linear Regression

Model: $$y = X\beta + \varepsilon$$, $$\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$, prior $$\beta \sim \mathcal{N}(0, \tau^2 I)$$.

Posterior:

$$
p(\beta \mid X, y) = \mathcal{N}\left(\hat{\mu}, \hat{\Sigma}\right)
$$

where:

$$
\hat{\Sigma} = \left(\frac{1}{\sigma^2} X^\top X + \frac{1}{\tau^2} I\right)^{-1}, \quad \hat{\mu} = \frac{1}{\sigma^2} \hat{\Sigma} X^\top y
$$

The MAP estimate (maximum a posteriori) of $$\beta$$ is exactly **ridge regression** with $$\lambda = \sigma^2/\tau^2$$.

---

## 27. MAP Estimation

The **Maximum A Posteriori** estimator:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \, p(\theta \mid x) = \arg\max_\theta \left[\log p(x \mid \theta) + \log p(\theta)\right]
$$

MAP = MLE + regularisation term. The prior acts as a regulariser:
- Gaussian prior $$\to$$ L2 (Ridge) regularisation
- Laplace prior $$\to$$ L1 (Lasso) regularisation

---

## 28. Variational Inference

When the posterior $$p(\theta \mid x)$$ is intractable, **variational inference** approximates it with a simpler distribution $$q_\phi(\theta)$$ from a tractable family $$\mathcal{Q}$$:

$$
q^*_\phi = \arg\min_{q \in \mathcal{Q}} \mathrm{KL}(q(\theta) \| p(\theta \mid x))
$$

Minimising the KL divergence is equivalent to maximising the **ELBO**:

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(x \mid \theta)] - \mathrm{KL}(q_\phi(\theta) \| p(\theta))
$$

since $$\log p(x) = \mathcal{L}(\phi) + \mathrm{KL}(q_\phi \| p(\cdot \mid x))$$.

**Mean-field approximation**: assumes $$q(\theta) = \prod_i q_i(\theta_i)$$.

This is the backbone of the **VAE** and most modern deep generative models.

---

## 29. Expectation-Maximisation (EM) Algorithm

The EM algorithm maximises the marginal likelihood $$p_\theta(x) = \int p_\theta(x, z) \, dz$$ when latent variables $$z$$ make direct optimisation difficult.

### 29.1 The Two Steps

**E-step**: compute the expected complete-data log-likelihood under the current parameters $$\theta^{(t)}$$:

$$
Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{z \mid x, \theta^{(t)}}[\log p_\theta(x, z)]
$$

**M-step**: maximise $$Q$$ with respect to $$\theta$$:

$$
\theta^{(t+1)} = \arg\max_\theta \, Q(\theta \mid \theta^{(t)})
$$

### 29.2 Convergence

EM **monotonically increases** the log-likelihood: $$\log p_{\theta^{(t+1)}}(x) \geq \log p_{\theta^{(t)}}(x)$$.

It converges to a local maximum (not necessarily global).

### 29.3 Example: Gaussian Mixture Model (GMM)

Model: $$x_i \mid z_i \sim \mathcal{N}(\mu_{z_i}, \Sigma_{z_i})$$, $$z_i \sim \text{Cat}(\pi)$$.

**E-step** (soft assignment):

$$
r_{ik} = \mathbb{P}(z_i = k \mid x_i, \theta^{(t)}) = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_{j} \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}
$$

**M-step** (parameter update):

$$
\pi_k^{(t+1)} = \frac{\sum_i r_{ik}}{n}, \quad \mu_k^{(t+1)} = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}, \quad \Sigma_k^{(t+1)} = \frac{\sum_i r_{ik}(x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i r_{ik}}
$$

---

# Part VII — Markov Chains and Stochastic Processes

## 30. Stochastic Processes

A **stochastic process** is a collection of random variables $$\{X_t\}_{t \in \mathcal{T}}$$ indexed by time $$\mathcal{T}$$.

- **Discrete time**: $$\mathcal{T} = \{0, 1, 2, \ldots\}$$
- **Continuous time**: $$\mathcal{T} = [0, \infty)$$

---

## 31. Markov Chains (Discrete Time)

A stochastic process $$(X_n)_{n \geq 0}$$ taking values in a countable state space $$\mathcal{S}$$ is a **Markov chain** if it satisfies the **Markov property**:

$$
\mathbb{P}(X_{n+1} = j \mid X_0, \ldots, X_n) = \mathbb{P}(X_{n+1} = j \mid X_n)
$$

The future depends on the present only, not the past — the chain is **memoryless**.

### 31.1 Transition Matrix

For a **homogeneous** Markov chain, the transition probabilities are time-invariant:

$$
P_{ij} = \mathbb{P}(X_{n+1} = j \mid X_n = i)
$$

The matrix $$P \in [0,1]^{|\mathcal{S}| \times |\mathcal{S}|}$$ satisfies $$\sum_j P_{ij} = 1$$ for all $$i$$ (row stochastic).

**$$n$$-step transition probabilities**: $$(P^n)_{ij} = \mathbb{P}(X_n = j \mid X_0 = i)$$.

### 31.2 Classification of States

- **Accessible**: $$j$$ is accessible from $$i$$ ($$i \to j$$) if $$\exists n \geq 0 : (P^n)_{ij} > 0$$
- **Communicating**: $$i \leftrightarrow j$$ if $$i \to j$$ and $$j \to i$$
- **Irreducible**: all states communicate
- **Recurrent**: $$\mathbb{P}(T_i < \infty \mid X_0 = i) = 1$$ (returns to $$i$$ with probability 1)
- **Transient**: $$\mathbb{P}(T_i < \infty \mid X_0 = i) < 1$$
- **Positive recurrent**: expected return time $$\mathbb{E}[T_i \mid X_0 = i] < \infty$$
- **Null recurrent**: expected return time is infinite
- **Periodic**: $$d(i) = \gcd\{n \geq 1 : (P^n)_{ii} > 0\} > 1$$
- **Aperiodic**: $$d(i) = 1$$

---

## 32. Stationary Distribution

A distribution $$\pi$$ is **stationary** (invariant) for $$P$$ if:

$$
\pi^\top P = \pi^\top, \quad \text{i.e.,} \quad \pi_j = \sum_i \pi_i P_{ij}
$$

**Existence and uniqueness** (Perron-Frobenius theorem): an irreducible, positive recurrent Markov chain has a unique stationary distribution $$\pi$$.

Moreover, $$\pi_i = 1 / \mathbb{E}[T_i \mid X_0 = i]$$.

---

## 33. Ergodicity and Convergence

An **ergodic** Markov chain is irreducible, positive recurrent, and aperiodic.

**Ergodic theorem**: for an ergodic chain,

$$
\frac{1}{n}\sum_{t=0}^{n-1} f(X_t) \xrightarrow{a.s.} \sum_i f(i) \pi_i = \mathbb{E}_\pi[f]
$$

**Convergence to stationarity**: for an ergodic chain,

$$
\|P^n(i, \cdot) - \pi\|_{\text{TV}} \to 0 \quad \text{as } n \to \infty
$$

where $$\|\cdot\|_{\text{TV}}$$ is the total variation distance.

---

## 34. Detailed Balance (Reversibility)

A Markov chain with stationary distribution $$\pi$$ satisfies **detailed balance** if:

$$
\pi_i P_{ij} = \pi_j P_{ji} \quad \forall i, j
$$

A chain satisfying detailed balance is **reversible** — it looks the same forwards and backwards in time.

Detailed balance is a sufficient (but not necessary) condition for $$\pi$$ to be stationary.

---

## 35. Markov Chain Monte Carlo (MCMC)

MCMC methods construct a Markov chain whose stationary distribution is the target distribution $$\pi$$. Running the chain and collecting samples (after a burn-in period) gives approximate samples from $$\pi$$.

### 35.1 Metropolis-Hastings Algorithm

To sample from $$\pi(x) \propto f(x)$$:

1. At state $$x$$, propose $$x' \sim q(x' \mid x)$$
2. Compute acceptance ratio: $$\alpha = \min\left(1, \frac{f(x') q(x \mid x')}{f(x) q(x' \mid x)}\right)$$
3. Accept: set $$X_{t+1} = x'$$ with probability $$\alpha$$; else set $$X_{t+1} = x$$

The resulting chain satisfies detailed balance with respect to $$\pi$$.

### 35.2 Gibbs Sampling

For a joint distribution $$p(x_1, \ldots, x_d)$$, iteratively sample:

$$
X_j^{(t+1)} \sim p(x_j \mid x_1^{(t+1)}, \ldots, x_{j-1}^{(t+1)}, x_{j+1}^{(t)}, \ldots, x_d^{(t)})
$$

Gibbs sampling is a special case of Metropolis-Hastings with acceptance rate 1. It is widely used in Bayesian inference when full conditionals are tractable.

---

## 36. Hidden Markov Models (HMM)

A **Hidden Markov Model** is a Markov chain $$(Z_t)$$ whose states are **unobserved**, with observations $$(X_t)$$ conditionally independent given the hidden states:

$$
Z_{t+1} \mid Z_t \sim P_{Z_t, \cdot}, \quad X_t \mid Z_t \sim p_{\text{obs}}(\cdot \mid Z_t)
$$

**Three fundamental problems**:
1. **Evaluation**: compute $$p(x_1, \ldots, x_T \mid \lambda)$$ — **Forward algorithm** (dynamic programming)
2. **Decoding**: find $$\arg\max_{z} \mathbb{P}(Z = z \mid X = x)$$ — **Viterbi algorithm**
3. **Learning**: estimate parameters $$\lambda = (P, p_{\text{obs}}, \pi)$$ from data — **Baum-Welch algorithm** (a special case of EM)

HMMs are widely used in speech recognition, bioinformatics, and financial modelling.

---

## 37. Continuous-Time Markov Chains

A continuous-time Markov chain $$(X_t)_{t \geq 0}$$ satisfies:

$$
\mathbb{P}(X_{t+s} = j \mid X_u, u \leq t) = \mathbb{P}(X_{t+s} = j \mid X_t)
$$

The chain is characterised by an **infinitesimal generator** (rate matrix) $$Q$$:

$$
Q_{ij} \geq 0 \text{ for } i \neq j, \quad Q_{ii} = -\sum_{j \neq i} Q_{ij}
$$

The transition matrix satisfies: $$P(t) = e^{Qt}$$.

**Stationary distribution**: $$\pi Q = 0$$ with $$\sum_i \pi_i = 1$$.

---

## 38. Brownian Motion and Diffusion Processes

**Brownian motion** (Wiener process) $$W_t$$ satisfies:
1. $$W_0 = 0$$
2. Independent increments
3. $$W_t - W_s \sim \mathcal{N}(0, t-s)$$ for $$t > s$$
4. Continuous paths almost surely

A **diffusion process** is defined by a stochastic differential equation (SDE):

$$
dX_t = \mu(X_t, t) \, dt + \sigma(X_t, t) \, dW_t
$$

where $$\mu$$ is the **drift** and $$\sigma$$ is the **diffusion coefficient**.

**Itô's lemma**: for $$f \in C^2$$:

$$
df(X_t) = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{\sigma^2}{2}\frac{\partial^2 f}{\partial x^2}\right)dt + \sigma \frac{\partial f}{\partial x} dW_t
$$

Diffusion processes are the continuous-time limit of random walks and underlie modern **score-based generative models** and **diffusion models**.

---

# Part VIII — Deep Learning Foundations

## 39. Neural Networks

A **neural network** is a composition of affine transformations and non-linear activations:

$$
f(x) = \sigma_L(W_L \sigma_{L-1}(\cdots \sigma_1(W_1 x + b_1)\cdots) + b_L)
$$

**Universal approximation theorem**: a single hidden layer network with enough neurons can approximate any continuous function on a compact set.

### Common Activation Functions

| Name | Formula | Use |
|------|---------|-----|
| ReLU | $$\max(0, x)$$ | Hidden layers (default) |
| Sigmoid | $$1/(1+e^{-x})$$ | Binary output |
| Softmax | $$e^{x_k}/\sum_j e^{x_j}$$ | Categorical output |
| Tanh | $$(e^x - e^{-x})/(e^x + e^{-x})$$ | Hidden layers |
| GELU | $$x \Phi(x)$$ | Transformers |

---

## 40. Backpropagation

Backpropagation is the efficient application of the **chain rule** to compute gradients of the loss $$\mathcal{L}$$ with respect to all parameters.

For a layer $$h = \sigma(Wx + b)$$:

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial h}{\partial W}, \quad \frac{\partial \mathcal{L}}{\partial x} = W^\top \frac{\partial \mathcal{L}}{\partial h}
$$

The computational complexity is $$O(|\text{parameters}|)$$ per sample.

---

## 41. Regularisation

**L2 (weight decay)**: adds $$\lambda \|W\|_F^2$$ to the loss. Corresponds to a Gaussian prior on weights.

**L1**: adds $$\lambda \|W\|_1$$. Promotes sparsity. Corresponds to a Laplace prior.

**Dropout**: randomly zero out neurons with probability $$p$$ during training. Acts as ensemble averaging.

**Batch normalisation**: normalise activations within a mini-batch:

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta
$$

---

## 42. Probabilistic Perspective on Deep Learning

### 42.1 MLE as Minimising Cross-Entropy

Maximising the log-likelihood for a categorical model:

$$
\hat{\theta} = \arg\max_\theta \sum_i \log p_\theta(y_i \mid x_i)
$$

is equivalent to minimising the **cross-entropy loss**:

$$
\mathcal{L} = -\frac{1}{n}\sum_i \sum_k y_{ik} \log \hat{p}_{ik}
$$

### 42.2 MSE as MLE Under Gaussian Noise

Minimising the mean squared error:

$$
\mathcal{L} = \frac{1}{n}\sum_i \|y_i - f_\theta(x_i)\|^2
$$

is equivalent to MLE under the model $$y_i = f_\theta(x_i) + \varepsilon_i$$, $$\varepsilon_i \sim \mathcal{N}(0, \sigma^2 I)$$.

---

# Summary Table

| Area | Key Tools in ML |
|------|----------------|
| **Linear Algebra** | PCA, SVD, eigendecomposition, covariance matrices |
| **Calculus** | Backpropagation, gradient descent, Taylor expansions |
| **Probability** | Distributions, expectations, CLT, LLN |
| **Information Theory** | KL divergence, entropy, mutual information, ELBO |
| **Statistics** | MLE, MAP, Fisher information, regression |
| **Bayesian Methods** | Posterior inference, conjugate priors, variational inference |
| **EM Algorithm** | GMM, HMM, latent variable models |
| **Markov Chains** | MCMC, ergodicity, stationary distributions |
| **Stochastic Processes** | Brownian motion, SDEs, diffusion models |
| **Deep Learning** | Neural networks, backpropagation, regularisation |

---

*This post is a living reference — I will continue updating it as my research evolves. Feel free to reach out for questions or discussions.*
