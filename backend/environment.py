import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FinancialPortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, feature_tensor, returns_matrix, tickers, lookback_window=5, rolling_reward_window=20):
        super(FinancialPortfolioEnv, self).__init__()
        
        self.feature_tensor = feature_tensor        # Shape: (Total_Days, Lookback, Num_Assets * Num_Features)
        self.returns_matrix = returns_matrix        # Shape: (Total_Days, Num_Assets)
        self.tickers = tickers
        self.num_assets = len(tickers)
        self.lookback_window = lookback_window
        self.rolling_reward_window = rolling_reward_window
        
        self.max_steps = len(returns_matrix) - 1
        self.current_step = 0
        self.historical_returns = []
        self.last_weights = np.ones(self.num_assets, dtype=np.float32) / self.num_assets

        # Action Space: Raw unbounded logits for portfolio rebalancing
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32
        )

        # Observation Space: Flattened engineered feature matrix + previous portfolio weights vector
        feature_dim = self.feature_tensor.shape[1] * self.feature_tensor.shape[2]
        total_obs_dim = feature_dim + self.num_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )

    def _get_obs(self):
        market_features = self.feature_tensor[self.current_step].flatten()
        observation = np.concatenate([market_features, self.last_weights]).astype(np.float32)
        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.historical_returns = [0.0] * self.rolling_reward_window
        self.last_weights = np.ones(self.num_assets, dtype=np.float32) / self.num_assets
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Enforce continuous probability simplex via numerical-safe Softmax
        exp_action = np.exp(action - np.max(action))
        current_weights = exp_action / np.sum(exp_action)
        
        assert np.isclose(np.sum(current_weights), 1.0, atol=1e-5), "Simplex constraint failure."

        # Compute portfolio return for the current step
        step_asset_returns = self.returns_matrix[self.current_step]
        portfolio_return = float(np.dot(current_weights, step_asset_returns))
        self.historical_returns.append(portfolio_return)

        if len(self.historical_returns) > self.rolling_reward_window:
            self.historical_returns.pop(0)

        # Rolling 20-Day Annualized Sharpe Ratio Reward Shaping
        returns_window = np.array(self.historical_returns)
        mean_ret = np.mean(returns_window)
        std_ret = np.std(returns_window)

        if std_ret > 1e-6:
            reward = float((mean_ret / std_ret) * np.sqrt(252))
        else:
            reward = float(mean_ret)

        self.last_weights = current_weights
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "weights": current_weights,
            "step_return": portfolio_return,
            "sharpe": reward
        }

        next_obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return next_obs, reward, terminated, truncated, info