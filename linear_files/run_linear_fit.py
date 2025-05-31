from linear_data_utils import (
    generate_linear_data,
    least_squares_fit,
    mlp_fit,
    plot_all,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate linear data with noise and plot it.")
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--sigma', type=float, default=0.1, help='Standard deviation of noise')
    parser.add_argument('--filename', type=str, default="lindata", help='Output filename (no extension)')
    args = parser.parse_args()
    generate_linear_data(args.n_samples, args.sigma, args.filename)
    least_squares_fit(args.filename)
    mlp_fit(args.filename)
    plot_all(args.filename, "lsqdata", "MLPdata", "all_fits.png")