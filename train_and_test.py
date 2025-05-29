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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class LinearAndNonlinearExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that outputs both linear and nonlinear features.
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.linear = nn.Linear(observation_space.shape[0], 1)
        self.nonlinear = nn.Sequential(
            nn.Linear(observation_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, features_dim-1),
            nn.ReLU()
        )

    def forward(self, observations):
        linear_out = self.linear(observations)
        nonlinear_out = self.nonlinear(observations)
        # Concatenate linear and nonlinear outputs
        return th.cat([linear_out, nonlinear_out], dim=1)

class LinearAndNonlinearPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
            features_extractor_class=LinearAndNonlinearExtractor,
            features_extractor_kwargs={"features_dim": 64}
        )

def train_agent(total_timesteps, learning_rate, use_custom_policy=False):
    env = LinearRegressionEnv()
    check_env(env)
    policy = LinearAndNonlinearPolicy if use_custom_policy else "MlpPolicy"
    model = PPO(policy, env, verbose=1, learning_rate=learning_rate)
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
    parser.add_argument('--custom_policy', action='store_true', help='Use custom actor with both linear and nonlinear heads')
    args = parser.parse_args()
    model = train_agent(args.steps, args.lr, use_custom_policy=args.custom_policy)
    test_agent(model)

if __name__ == "__main__":
    main()
