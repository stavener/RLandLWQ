import torch

# Get user input
x = float(input("Enter value for x: "))
y = float(input("Enter value for y: "))

# Convert to tensors
x_tensor = torch.tensor(x)
y_tensor = torch.tensor(y)

# Initialize alpha (requires_grad=True for optimization)
alpha = torch.tensor(1.0, requires_grad=True)

# Set up optimizer
optimizer = torch.optim.SGD([alpha], lr=0.1)

# Training loop
for i in range(1000):
    optimizer.zero_grad()
    prediction = alpha**2 * x_tensor
    loss = (y_tensor - prediction) ** 2
    loss.backward()
    optimizer.step()

# Print the result
print(f"Optimal alpha: {alpha.item()}")
print(f"Least squares error: {loss.item()}")