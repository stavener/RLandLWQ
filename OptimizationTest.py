import numpy as np
import torch
import json

# -----------------------------
# User Input or File Input
# -----------------------------
def get_inputs():
    """
    Prompt user for input or load from a JSON file.
    Returns: dict with keys p, q, m, K, a, b
    """
    use_file = input("Read input from file? (y/n): ").strip().lower()
    if use_file == 'y':
        filename = input("Enter input JSON filename (default: 'input_params.json'): ").strip() or 'input_params.json'
        with open(filename, 'r') as f:
            params = json.load(f)
        for k in ['p', 'q', 'm', 'K', 'a', 'b']:
            if k not in params:
                raise ValueError(f"Missing key '{k}' in input file.")
        return params
    else:
        p = int(input("Enter value for p: "))
        q = int(input("Enter value for q: "))
        m = int(float(input("Enter value for m: ")))
        K = int(float(input("Enter value for K: ")))
        a = float(input("Enter value for a: "))
        b = float(input("Enter value for b: "))
        return dict(p=p, q=q, m=m, K=K, a=a, b=b)

params = get_inputs()
p, q, m, K, a, b = params['p'], params['q'], params['m'], params['K'], params['a'], params['b']

# -----------------------------
# Data Loading
# -----------------------------
D_np = np.loadtxt("data.txt")
if D_np.shape != (p + q, m):
    raise ValueError(f"data.txt must have shape ({p + q}, {m}), but got {D_np.shape}")
print("Matrix D loaded from data.txt:")
print(D_np)
D = torch.tensor(D_np, dtype=torch.float32)

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
    theta = theta_flat.view(q, p)
    F = torch.zeros((p + q, p + q), dtype=torch.float32)
    F[:p, :p] = torch.eye(p)
    F[p:p+q, :p] = theta
    error = 0.0
    for i in range(m):
        w = torch.zeros((p + q, 1), dtype=torch.float32)
        w[:p, 0] = D[:p, i]
        for _ in range(K):
            w = F @ w
        y = D[p:p+q, i].reshape((q, 1))
        diff = y - w[p:p+q, :]
        error += torch.sum(diff ** 2)
    return error

# -----------------------------
# Theta Initialization
# -----------------------------
theta_init = torch.zeros((q, p), dtype=torch.float32, requires_grad=True)
with torch.no_grad():
    if theta_init.numel() > 0:
        theta_init.view(-1)[0] = a - 1
    if theta_init.numel() > 1:
        theta_init.view(-1)[1] = b - 1

# -----------------------------
# Optimization
# -----------------------------
optimizer = torch.optim.LBFGS([theta_init], max_iter=100)

def closure():
    optimizer.zero_grad()
    loss = error_fn(theta_init, D, p, q, m, K)
    loss.backward()
    return loss

optimizer.step(closure)

theta_opt = theta_init.detach().numpy()
min_error = error_fn(torch.tensor(theta_opt, dtype=torch.float32), D, p, q, m, K).item()

print("\nOptimized theta (reshaped to q x p):")
print(theta_opt)
print("Minimum error:", min_error)