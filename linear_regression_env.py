import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LinearRegressionEnv(gym.Env):
    """
    Gymnasium environment for a simple linear regression problem.
    Observation: x (float)
    Action: predicted y (float)
    Reward: negative mean squared error between predicted y and true y
    """
    def __init__(self, w=2.0, b=1.0, noise_std=0.1, n_samples=100):
        super().__init__()
        self.w = w
        self.b = b
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)
        self.xs = np.random.uniform(-10, 10, self.n_samples)
        self.ys = self.w * self.xs + self.b + np.random.normal(0, self.noise_std, self.n_samples)
        self.idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.xs = np.random.uniform(-10, 10, self.n_samples)
        self.ys = self.w * self.xs + self.b + np.random.normal(0, self.noise_std, self.n_samples)
        self.idx = 0
        return np.array([self.xs[self.idx]], dtype=np.float32), {}

    def step(self, action):
        x = self.xs[self.idx]
        y_true = self.ys[self.idx]
        y_pred = action[0]
        reward = -((y_pred - y_true) ** 2)
        self.idx += 1
        done = self.idx >= self.n_samples
        if not done:
            obs = np.array([self.xs[self.idx]], dtype=np.float32)
        else:
            obs = np.zeros(1, dtype=np.float32)
        info = {"y_true": y_true, "y_pred": y_pred}
        return obs, reward, done, False, info
