# Optimization Methods for Adversarial Attacks

This document provides the mathematical formulation of the optimization methods used for generating adversarial examples.

## Problem Formulation

For an image classification model $f: \mathbb{R}^d \rightarrow \mathbb{R}^k$ that outputs class probabilities, adversarial attacks aim to find a perturbation $\delta$ that changes the prediction while remaining imperceptible. This can be formulated as:

$$\min_{\delta} \mathcal{L}(f(x + \delta), y) \quad \text{subject to} \quad \|\delta\|_p \leq \epsilon$$

Where:
- $x$ is the original image
- $y$ is the target label (original label for untargeted attacks)
- $\mathcal{L}$ is a loss function (cross-entropy or margin loss)
- $\|\delta\|_p$ is the $L_p$ norm of the perturbation
- $\epsilon$ is the maximum perturbation size

## 1. Projected Gradient Descent (PGD)

PGD is a first-order method that iteratively takes steps in the gradient direction and projects back onto the constraint set.

### Algorithm

Initialize $x_0 = x$ (or with small random noise), then iterate:

$$x_{t+1} = \Pi_{\mathcal{B}_\epsilon(x)}\left(x_t + \alpha_t \nabla_x \mathcal{L}(f(x_t), y)\right)$$

Where:
- $\Pi_{\mathcal{B}_\epsilon(x)}$ is the projection onto the $\epsilon$-ball around $x$
- $\alpha_t$ is the step size at iteration $t$

### Step Size Schedules

- Constant: $\alpha_t = \alpha_0$
- Diminishing: $\alpha_t = \frac{\alpha_0}{\sqrt{t+1}}$

### Convergence

For convex objectives with Lipschitz-continuous gradients and diminishing step sizes, PGD converges to the global optimum. For non-convex objectives (as in deep learning), PGD converges to a local optimum with rate $O(1/\sqrt{T})$ for the averaged iterates.

## 2. Conjugate Gradient (CG)

CG improves upon steepest descent by using conjugate directions that produce more efficient traversal of the optimization landscape.

### Algorithm

Initialize $x_0 = x$, compute $g_0 = \nabla_x \mathcal{L}(f(x_0), y)$, and set $d_0 = -g_0$. Then iterate:

1. Line search: Find $\alpha_t$ that approximately minimizes $\mathcal{L}(f(x_t + \alpha_t d_t), y)$
2. Update: $x_{t+1} = \Pi_{\mathcal{B}_\epsilon(x)}\left(x_t + \alpha_t d_t\right)$
3. Compute gradient: $g_{t+1} = \nabla_x \mathcal{L}(f(x_{t+1}), y)$
4. Compute $\beta$ using one of:
   - Fletcher-Reeves: $\beta_t = \frac{\|g_{t+1}\|^2}{\|g_t\|^2}$
   - Polak-Ribière: $\beta_t = \max\left(0, \frac{g_{t+1}^T(g_{t+1} - g_t)}{\|g_t\|^2}\right)$
5. Update direction: $d_{t+1} = -g_{t+1} + \beta_t d_t$
6. Periodically reset: If $t \bmod m = 0$, set $d_{t+1} = -g_{t+1}$

### Line Search

We use backtracking line search with the Armijo condition:

$$\mathcal{L}(f(x_t + \alpha_t d_t), y) \leq \mathcal{L}(f(x_t), y) + c_1 \alpha_t g_t^T d_t$$

### Convergence

For quadratic objectives, CG converges in at most $n$ iterations (where $n$ is the problem dimension). For non-quadratic objectives, CG with periodic restarts still offers superlinear convergence rates under certain conditions.

## 3. Limited-memory BFGS (L-BFGS)

L-BFGS approximates the inverse Hessian matrix using a limited history of positions and gradients, offering quasi-Newton acceleration without storing the full Hessian.

### Algorithm

Initialize $x_0 = x$ and iterate:

1. Compute search direction using two-loop recursion to implicitly apply $H_k \approx [\nabla^2 \mathcal{L}(f(x_k), y)]^{-1}$:
   - $q = g_k$
   - For $i = k-1, k-2, ..., k-m$:
     - $\alpha_i = \rho_i s_i^T q$
     - $q = q - \alpha_i y_i$
   - $r = H_0^k q$ (with $H_0^k = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}} I$)
   - For $i = k-m, ..., k-1$:
     - $\beta = \rho_i y_i^T r$
     - $r = r + s_i(\alpha_i - \beta)$
   - Set direction $d_k = -r$
2. Line search: Find $\alpha_k$ satisfying Wolfe conditions
3. Update: $x_{k+1} = \Pi_{\mathcal{B}_\epsilon(x)}\left(x_k + \alpha_k d_k\right)$
4. Store history:
   - $s_k = x_{k+1} - x_k$
   - $y_k = g_{k+1} - g_k$
   - $\rho_k = \frac{1}{y_k^T s_k}$

Where:
- $s_i$ is the position difference at iteration $i$
- $y_i$ is the gradient difference at iteration $i$
- $\rho_i = \frac{1}{y_i^T s_i}$
- $m$ is the history size

### Wolfe Conditions

1. Sufficient decrease (Armijo): $\mathcal{L}(f(x_k + \alpha_k d_k), y) \leq \mathcal{L}(f(x_k), y) + c_1 \alpha_k g_k^T d_k$
2. Curvature: $|g_{k+1}^T d_k| \leq c_2 |g_k^T d_k|$

Where $0 < c_1 < c_2 < 1$ are constants (typically $c_1 = 10^{-4}$, $c_2 = 0.9$).

### Convergence

L-BFGS exhibits superlinear convergence rates for strongly convex functions with Lipschitz continuous Hessians. For non-convex problems, it generally converges faster than first-order methods.

## Comparative Analysis

Each method offers different trade-offs:

- **PGD**: Simplest to implement and most robust, but slower convergence
- **CG**: Better convergence than PGD and moderate memory requirements
- **L-BFGS**: Fastest convergence but more complex and potentially less stable for highly non-convex problems

The choice between methods depends on factors such as problem size, available computational resources, and the need for precision versus speed.

## References

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.
2. Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer Science & Business Media.
3. Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients for solving linear systems. Journal of Research of the National Bureau of Standards, 49(6), 409-436.
4. Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. Mathematical Programming, 45(1-3), 503-528.
