import jax
import jax.numpy as jnp

# Get user input
x = float(input("Enter value for x: "))
y = float(input("Enter value for y: "))

# Define the loss function
def loss_fn(alpha, x, y):
    prediction = alpha**2 * x
    return (y - prediction) ** 2

# Initialize alpha
alpha = 1.0

# Set learning rate and number of steps
lr = 0.1
steps = 1000

# Gradient function
grad_fn = jax.grad(loss_fn)

# Gradient descent loop
for i in range(steps):
    grad = grad_fn(alpha, x, y)
    alpha -= lr * grad

# Print the result
print(f"Optimal alpha: {alpha}")
print(f"Least squares error: {loss_fn(alpha, x, y)}")