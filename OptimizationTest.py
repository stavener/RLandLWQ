import numpy as np
import torch

# Let the user define variables p, q, m, K, a, b
p = int(input("Enter value for p: "))
q = int(input("Enter value for q: "))
m = int(float(input("Enter value for m: ")))
K = int(float(input("Enter value for K: ")))
a = float(input("Enter value for a: "))
b = float(input("Enter value for b: "))

# Load D from file
D_np = np.loadtxt("data.txt")
if D_np.shape != (p + q, m):
    raise ValueError(f"data.txt must have shape ({p + q}, {m}), but got {D_np.shape}")
print("Matrix D:")
print(D_np)
D = torch.tensor(D_np, dtype=torch.float32)

def error_fn(theta_flat, D, p, q, m, K):
    theta = theta_flat.view(q, p)
    # Build block matrix F
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

# Initialize theta so that the first entry is a-1, the second entry is b-1, all others zero
theta_init = torch.empty((q, p), dtype=torch.float32, requires_grad=True)
with torch.no_grad():
    theta_init.zero_()
    if theta_init.numel() > 0:
        theta_init.view(-1)[0] = a + 1
    if theta_init.numel() > 1:
        theta_init.view(-1)[1] = b + 2

# Use LBFGS optimizer
optimizer = torch.optim.LBFGS([theta_init], max_iter=100)

def closure():
    optimizer.zero_grad()
    loss = error_fn(theta_init, D, p, q, m, K)
    loss.backward()
    return loss

optimizer.step(closure)

theta_opt = theta_init.detach().numpy()
min_error = error_fn(torch.tensor(theta_opt, dtype=torch.float32), D, p, q, m, K).item()

print("Optimized theta:")
print(theta_opt)
print("Minimum error:", min_error)