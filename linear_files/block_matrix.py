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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a block matrix as described.")
    parser.add_argument('--m', type=int, required=True, help='Number of blocks per row/column')
    parser.add_argument('--k', type=int, required=True, help='Size of each block')
    args = parser.parse_args()
    mat = create_block_matrix(args.m, args.k)
    np.set_printoptions(precision=3, suppress=True)
    print(mat)
