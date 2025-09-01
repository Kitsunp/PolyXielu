# Advanced Activation Functions: XIELUPoly and XIELUPolyNorm

This repository provides PyTorch implementations of two novel activation functions, `XIELUPoly` and `XIELUPolyNorm`. These functions are designed by combining the gradient-centric **xIELU** activation with the expressive power of **Polynomial Composition (PolyCom)**, drawing inspiration from recent advancements in activation function design for large language models.

The core idea is to leverage the stable and adaptive gradient properties of xIELU as a base and enhance its representational capacity by applying polynomial transformations, as proposed in the PolyCom framework.

## Core Concepts

Before diving into the implementations, it's essential to understand the building blocks.

### 1. The xIELU Activation Function

**xIELU (Expanded Integral of the Exponential Linear Unit)** is a trainable, piecewise activation function designed by focusing on its gradient properties first and then deriving the function via integration. It combines two key features:

1.  A **linearly increasing gradient** for positive inputs, similar to `ReLU²`, which allows for effective learning from large activation values.
2.  A **trainable, negative-valued gradient** for negative inputs, inspired by `xSiLU`, which mitigates the "dying ReLU" problem and allows for more complex data modeling.

The mathematical formulation of xIELU is:

$$
\text{xIELU}(x) = \begin{cases}
\alpha_p x^2 + \beta x & \text{if } x > 0 \\
\alpha_n (e^x - 1) - \alpha_n x + \beta x & \text{if } x \le 0
\end{cases}
$$

Where:
- $\alpha_p > 0$ and $\alpha_n > \beta$ are trainable scalar parameters that control the function's nonlinearity.
- $\beta$ is a fixed scalar (typically 0.5) that ensures gradient continuity at $x=0$.

#### Numerical Stability: Taylor Patch

For inputs very close to zero ($x \to 0^-$), the term `(e^x - 1)` can suffer from catastrophic cancellation, leading to numerical instability. To address this, our implementation uses a third-order Taylor series approximation of `(e^x - 1)` in a small interval `[-ε, 0]`:

$$
e^x - 1 \approx x + \frac{x^2}{2} + \frac{x^3}{6}
$$

This makes the practical implementation a three-piece function, ensuring stable training.

### 2. Polynomial Composition (PolyCom)

**PolyCom** is a framework for enhancing activation functions by composing them with polynomials. This allows a network to learn higher-order interactions within the data. The original paper introduces two main types:

#### PolyCom Type I
A polynomial is applied to the *output* of a base activation function, $u = \sigma(x)$.

$$
\text{PolyCom-I}(x) = P(u) = \sum_{i=0}^{r} a_i u^i = a_0 + a_1 u + a_2 u^2 + \dots + a_r u^r
$$

where $a_i$ are learnable coefficients.

#### PolyCom Type II
A weighted sum of a function $\rho$ applied to the *powers of the input*. **PolyNorm** is a specific, powerful instance of this type.

$$
\text{PolyCom-II}(x) = \sum_{i=0}^{r} a_i \rho(x^i)
$$

The **PolyNorm** variant uses RMS normalization for $\rho$:

$$
\text{Norm}(z) = z \cdot \left( \mathbb{E}[z^2] + \epsilon \right)^{-1/2}
$$

## Implemented Activation Functions

Our work combines these concepts to create two powerful activation functions.

### 1. `XIELUPoly` (A PolyCom Type I Implementation)

This module implements a **PolyCom Type I** activation where the base function is `xIELU`. It applies a 3rd-order polynomial to the output of xIELU.

First, the base activation is computed:
$$
u = \text{xIELU}(x)
$$

Then, a cubic polynomial with learnable coefficients $[a_0, a_1, a_2, a_3]$ is applied to $u$:

$$
\text{XIELUPoly}(x) = a_0 + a_1 u + a_2 u^2 + a_3 u^3
$$

For computational efficiency, the implementation uses **Horner's method** to evaluate the polynomial, which minimizes multiplications:

$$
\text{XIELUPoly}(x) = a_0 + u \cdot (a_1 + u \cdot (a_2 + u \cdot a_3))
$$

### 2. `XIELUPolyNorm` (A PolyNorm Implementation)

This module first computes the base activation $u = \text{xIELU}(x)$ and then applies the **PolyNorm** transformation. This can be seen as a specialized variant of PolyCom Type II applied in activation space.

The steps are as follows:

1.  Compute the base activation:
    $$
    u = \text{xIELU}(x)
    $$

2.  Calculate the powers of the base activation: $u^1, u^2, u^3$.

3.  Normalize each power using RMS Normalization:
    $$
    \text{norm_u}^k = \text{Norm}(u^k) = u^k \cdot \left( \mathbb{E}[(u^k)^2] + \epsilon \right)^{-1/2}
    $$

4.  Compute the final output as a weighted sum of the normalized powers, plus a bias term:
    $$
    \text{XIELUPolyNorm}(x) = w_2 \cdot \text{norm_u}^1 + w_1 \cdot \text{norm_u}^2 + w_0 \cdot \text{norm_u}^3 + b
    $$
    *(Note: The code uses `weight[0]` for the cubic term, `weight[1]` for quadratic, and `weight[2]` for linear, which is a common convention in the paper.)*

## How to Use

```python
import torch
import torch.nn as nn
from Polynorm import XIELUPoly, XIELUPolyNorm

# Example usage
input_tensor = torch.randn(16, 128, 512) # (Batch, Sequence, Dimension)

# Initialize XIELUPoly (PolyCom Type I)
xielu_poly_activation = XIELUPoly()
output_poly = xielu_poly_activation(input_tensor)

# Initialize XIELUPolyNorm
xielu_polynorm_activation = XIELUPolyNorm()
output_polynorm = xielu_polynorm_activation(input_tensor)

print("XIELUPoly output shape:", output_poly.shape)
print("XIELUPolyNorm output shape:", output_polynorm.shape)
```

```References
These implementations are based on the ideas presented in the following papers:
Zhuo, Z., Wang, Y., Zeng, Y., et al. (2024). POLYNOMIAL COMPOSITION ACTIVATIONS: UNLEASHING THE DYNAMICS OF LARGE LANGUAGE MODELS. arXiv:2411.03884.
Huang, A. H., & Schlag, I. (2025). Deriving Activation Functions Using Integration. arXiv:2411.13010.
