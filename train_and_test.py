"""
Train and test a reinforcement learning agent (using stable-baselines3) on a linear regression environment.
The environment is defined in linear_regression_env.py.
Results (plots and metrics) will be saved in the 'results/' directory.

You can set the number of training steps with the --steps argument, e.g.:
    python train_and_test.py --steps 10000
If not specified, defaults to 5000 steps.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from linear_regression_env import LinearRegressionEnv
import argparse

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_agent(total_timesteps, learning_rate):
    env = LinearRegressionEnv()
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate)
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(RESULTS_DIR, "ppo_linear_regression"))
    return model

def test_agent(model, n_test=100):
    env = LinearRegressionEnv(n_samples=n_test)
    obs, _ = env.reset()
    y_trues, y_preds = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        y_trues.append(info["y_true"])
        y_preds.append(info["y_pred"])
    mse = np.mean((np.array(y_trues) - np.array(y_preds)) ** 2)
    print(f"Test MSE: {mse:.4f}")
    plt.figure()
    plt.scatter(y_trues, y_preds, alpha=0.7)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("Linear Regression RL Agent Predictions")
    plt.plot([min(y_trues), max(y_trues)], [min(y_trues), max(y_trues)], 'r--')
    plt.savefig(os.path.join(RESULTS_DIR, "test_scatter.png"))
    plt.close()
    with open(os.path.join(RESULTS_DIR, "test_mse.txt"), "w") as f:
        f.write(f"Test MSE: {mse:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description="Train and test RL agent for linear regression.")
    parser.add_argument('--steps', type=int, default=5000, help='Number of training steps (default: 5000)')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate for PPO optimizer (default: 0.0003)')
    args = parser.parse_args()
    model = train_agent(args.steps, args.lr)
    test_agent(model)

if __name__ == "__main__":
    main()
