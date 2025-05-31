import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from scipy.optimize import minimize

# -----------------------------
# User Input
# -----------------------------
# Get problem dimensions and parameters from user
p = int(input("Enter value for p: "))  # Number of input features
q = int(input("Enter value for q: "))  # Number of output features
m = int(float(input("Enter value for m: ")))  # Number of data samples
K = int(float(input("Enter value for K: ")))  # Number of F applications

a = float(input("Enter value for a: "))  # Parameter a
b = float(input("Enter value for b: "))  # Parameter b

# -----------------------------
# Data Loading
# -----------------------------
# Load data matrix D from file (shape: (p+q, m))
d_np = np.loadtxt("data.txt")
if d_np.shape != (p + q, m):
    raise ValueError(f"data.txt must have shape ({p + q}, {m}), but got {d_np.shape}")
print("Matrix D loaded from data.txt:")
print(d_np)
D = jnp.array(d_np)

# -----------------------------
# Error Function Definition
# -----------------------------
def error_fn(theta_flat, D, p, q, m, K):
    """
    Compute the sum of squared errors for the dynamical system defined by block matrix F.
    Args:
        theta_flat: Flattened (q*p,) parameter vector for theta.
        D: Data matrix (p+q, m).
        p, q, m, K: Problem dimensions and number of F applications.
    Returns:
        Scalar error value (sum of squared differences).
    """
    theta = theta_flat.reshape((q, p))
    # Construct block matrix F
    F = jnp.zeros((p + q, p + q))
    F = F.at[:p, :p].set(jnp.eye(p))
    F = F.at[p:p+q, :p].set(theta)
    error = 0.0
    for i in range(m):
        # Initialize w with first p rows from D
        w = jnp.zeros((p + q, 1))
        w = w.at[:p, 0].set(D[:p, i])
        # Apply F K times
        for _ in range(K):
            w = F @ w
        # Target output is last q rows of D
        y = D[p:p+q, i].reshape((q, 1))
        diff = y - w[p:p+q, :]
        error += jnp.sum(diff ** 2)
    return error

# Wrapper for JAX grad
error_fn_jax = lambda theta_flat: error_fn(theta_flat, D, p, q, m, K)

# -----------------------------
# Theta Initialization
# -----------------------------
# Initialize theta: first entry is a-1, second is b-1, rest are zero
# (flattened for optimizer)
theta_init = np.zeros((q, p), dtype=np.float32)
if theta_init.size > 0:
    theta_init.flat[0] = a - 1
if theta_init.size > 1:
    theta_init.flat[1] = b - 1
x0 = theta_init.flatten()

# -----------------------------
# Optimization
# -----------------------------
# Use L-BFGS-B optimizer from scipy with JAX gradients
grad_fn = jax.jit(jax.grad(error_fn_jax))

result = minimize(
    fun=lambda x: np.array(error_fn_jax(x), dtype=np.float64),
    x0=x0,
    jac=lambda x: np.array(grad_fn(x), dtype=np.float64),
    method='L-BFGS-B',
    options={'maxiter': 100}
)

# -----------------------------
# Results
# -----------------------------
theta_opt = result.x.reshape((q, p))
min_error = error_fn(theta_opt.flatten(), D, p, q, m, K)

print("\nOptimized theta (reshaped to q x p):")
print(theta_opt)
print("Minimum error:", float(min_error))
