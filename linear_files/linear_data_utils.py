import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def generate_linear_data(n_samples, sigma, filename="lindata"):
    x = np.random.uniform(-1, 1, n_samples)
    noise = np.random.normal(0, sigma, n_samples)
    y = x + noise
    data = np.column_stack((x, y))
    np.savetxt(filename, data, header="x y", comments='')
    plt.figure()
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Linear data: y = x + N(0, {sigma}^2)")
    plt.grid(True)
    plt.savefig(filename + ".png")
    plt.close()

def least_squares_fit(datafile="lindata", out_file="lsqdata"):
    data = np.loadtxt(datafile, skiprows=1)
    x, y = data[:, 0], data[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    x_fit = np.linspace(-1, 1, 100)
    y_fit = a * x_fit + b
    np.savetxt(out_file, np.column_stack((x_fit, y_fit)), header="x_fit y_fit", comments='')
    plt.figure()
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.plot(x_fit, y_fit, 'r-', label=f"LSQ fit: y={a:.3f}x+{b:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Least Squares Fit")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_file + ".png")
    plt.close()

#class CustomMLPRegressor(MLPRegressor):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)

def mlp_fit(datafile="lindata", out_file="MLPdata"):
    data = np.loadtxt(datafile, skiprows=1)
    x, y = data[:, 0], data[:, 1]
    x = x.reshape(-1, 1)
    mlp = MLPRegressor(hidden_layer_sizes=(32, 32), activation='relu', max_iter=5000, random_state=0)
    mlp.fit(x, y)
    x_fit = np.linspace(-1, 1, 100).reshape(-1, 1)
    y_fit = mlp.predict(x_fit)
    np.savetxt(out_file, np.column_stack((x_fit.flatten(), y_fit)), header="x_fit y_fit", comments='')
    plt.figure()
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.plot(x_fit, y_fit, 'g-', label="MLP fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("MLP Fit")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_file + ".png")
    plt.close()

def plot_all(datafile="lindata", lsqfile="lsqdata", mlpfile="MLPdata", out_file="all_fits.png"):
    data = np.loadtxt(datafile, skiprows=1)
    x, y = data[:, 0], data[:, 1]
    lsq = np.loadtxt(lsqfile, skiprows=1)
    x_lsq, y_lsq = lsq[:, 0], lsq[:, 1]
    mlp = np.loadtxt(mlpfile, skiprows=1)
    x_mlp, y_mlp = mlp[:, 0], mlp[:, 1]
    plt.figure()
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.plot(x_lsq, y_lsq, 'r-', label="Least Squares Fit")
    plt.plot(x_mlp, y_mlp, 'g-', label="MLP Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data, Least Squares, and MLP Fits")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_file)
    plt.close()