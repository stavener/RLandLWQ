import torch
import numpy as np

# Get user input
x = float(input("Enter value for x: "))
y = float(input("Enter value for y: "))

# Convert to tensors
x_tensor = torch.tensor([[x]])  # shape (1,1)
y_tensor = torch.tensor([[y]])  # shape (1,1)

# Let user define the shape of alpha
p = int(input("Enter number of rows for alpha: "))
q = int(input("Enter number of columns for alpha: "))

# Initialize alpha as a matrix (requires_grad=True for optimization)
alpha = torch.ones((p, q), requires_grad=True)

# Set up optimizer
optimizer = torch.optim.SGD([alpha], lr=0.1)

# Training loop
for i in range(1000):
    optimizer.zero_grad()
    # Example: prediction = alpha^2 @ x_tensor (matrix multiplication)
    prediction = (alpha ** 2) @ x_tensor  # shape (p,1) if x_tensor is (q,1)
    # For demonstration, use only the first row if shapes don't match
    if prediction.shape != y_tensor.shape:
        prediction = prediction[:y_tensor.shape[0], :]
    loss = torch.sum((y_tensor - prediction) ** 2)
    loss.backward()
    optimizer.step()

# Print the result
print(f"Optimal alpha matrix:\n{alpha.detach().numpy()}")
print(f"Least squares error: {loss.item()}")
