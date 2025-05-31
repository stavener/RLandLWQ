import numpy as np

def create_block_matrix(m, k):
    """
    Create a (m*k) x (m*k) block matrix with m*m blocks of size k*k.
    Top-left block is identity, all subdiagonal blocks are random [0,1], others are zero.
    """
    size = m * k
    mat = np.zeros((size, size))
    for i in range(m):
        for j in range(m):
            row_start, row_end = i*k, (i+1)*k
            col_start, col_end = j*k, (j+1)*k
            if i == 0 and j == 0:
                mat[row_start:row_end, col_start:col_end] = np.eye(k)
            elif i > j and i - j == 1:
                mat[row_start:row_end, col_start:col_end] = np.random.rand(k, k)
            # else: already zero
    return mat

def create_xy_vector(m):
    """
    Create a vector of size m composed of two elements of size m/2.
    The first m/2 components are labeled x, the second m/2 are labeled y.
    Returns (vec, x, y)
    """
    assert m % 2 == 0, "m must be even"
    x = np.random.rand(m//2)
    y = np.random.rand(m//2)
    vec = np.concatenate([x, y])
    return vec, x, y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create block matrix and xy vector.")
    parser.add_argument('--m', type=int, required=True, help='Number of blocks per row/column (must be even for xy vector)')
    parser.add_argument('--k', type=int, required=True, help='Size of each block')
    args = parser.parse_args()
    mat = create_block_matrix(args.m, args.k)
    print("Block matrix:\n", mat)
    if args.m % 2 == 0:
        vec, x, y = create_xy_vector(args.m)
        print("\nCombined vector (first m/2 = x, second m/2 = y):\n", vec)
        print("x:", x)
        print("y:", y)
    else:
        print("\nCannot create xy vector: m must be even.")
