# Tutorial: Supervised Learning as Reinforcement Learning (Linear Regression)

## 1. Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 2. Training and Testing

Run the following command to train the RL agent and test its performance:

```bash
python train_and_test.py
```

- This will train a PPO agent (from stable-baselines3) on a custom gymnasium environment that mimics a linear regression problem.
- After training, the agent will be tested on new data.
- Results (test MSE and a scatter plot of predictions vs. true values) will be saved in the `results/` directory.

## 3. Output

- `results/test_mse.txt`: Contains the mean squared error on the test set.
- `results/test_scatter.png`: Scatter plot of true vs. predicted values.

## 4. Customization

- You can modify the environment parameters (e.g., slope, intercept, noise) in `linear_regression_env.py`.
- Adjust training steps or model parameters in `train_and_test.py` as needed.

## 5. Notes

- Make sure to activate the virtual environment before running the code.
- All code files are commented for clarity.
