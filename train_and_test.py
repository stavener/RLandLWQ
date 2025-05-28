"""
Train and test a reinforcement learning agent (using stable-baselines3) on a linear regression environment.
The environment is defined in linear_regression_env.py.
Results (plots and metrics) will be saved in the 'results/' directory.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from linear_regression_env import LinearRegressionEnv

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_agent():
    env = LinearRegressionEnv()
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
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
    model = train_agent()
    test_agent(model)

if __name__ == "__main__":
    main()
