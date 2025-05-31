import numpy as np

def create_xy_vector(m):
    """
    Create a vector of size m, first m/2 labeled x, second m/2 labeled y.
    Returns the vector and the labels.
    """
    if m % 2 != 0:
        raise ValueError("m must be even.")
    x = np.random.rand(m // 2)
    y = np.random.rand(m // 2)
    vec = np.concatenate([x, y])
    labels = [f"x{i+1}" for i in range(m // 2)] + [f"y{i+1}" for i in range(m // 2)]
    return vec, labels

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a vector of size m with x and y components.")
    parser.add_argument('--m', type=int, required=True, help='Size of the vector (must be even)')
    args = parser.parse_args()
    vec, labels = create_xy_vector(args.m)
    for label, value in zip(labels, vec):
        print(f"{label}: {value:.4f}")
