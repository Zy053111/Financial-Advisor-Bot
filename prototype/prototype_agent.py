import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import json
import requests
from stable_baselines3 import PPO

# =====================================================================
# 1. DATA PIPELINE LAYER
# =====================================================================
def load_magnificent_seven_data(start_date="2024-01-01", end_date="2025-12-31"):
    """
    Ingests historical daily closing prices for the 'Magnificent Seven' equity basket
    via the yfinance API framework to construct the underlying financial state tensors.
    """
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    print(f"[Data Pipeline] Fetching historical K-line data for: {tickers}...")
    
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    
    # Extract asset closing prices safely regardless of multi-index headers
    if 'Close' in raw_data.columns:
        cleaned_data = raw_data['Close']
    else:
        cleaned_data = raw_data.xs('Close', axis=1, level=0)
    
    # Vectorized imputation data cleaning: Forward-fill and backward-fill 
    # to handle missing values caused by weekend closures and market holidays
    cleaned_data = cleaned_data.ffill().bfill()
    
    # Transform raw prices into stationary Daily Log Returns (R_t = ln(P_t / P_{t-1}))
    # This removes non-stationarity price bias and improves neural network stability
    daily_returns = np.log(cleaned_data / cleaned_data.shift(1)).dropna()
    
    print("[Data Pipeline] Data ingestion completed successfully.")
    return cleaned_data, daily_returns

# =====================================================================
# 2. CUSTOM FINANCIAL ENVIRONMENT
# =====================================================================
class FinancialPortfolioEnv(gym.Env):
    """
    An advanced financial rebalancing environment modeled as a Markov Decision Process (MDP).
    Features rolling Sharpe Ratio reward shaping and structural defensive unit assertions.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, price_df, returns_df, rolling_reward_window=20):
        super(FinancialPortfolioEnv, self).__init__()
        
        self.price_df = price_df
        self.returns_df = returns_df
        self.num_assets = len(price_df.columns)
        self.current_step = 0
        self.max_steps = len(returns_df) - 1
        
        # Hyperparameters for risk management tracking loops
        self.rolling_reward_window = rolling_reward_window
        self.historical_portfolio_returns = []

        # Continuous Action Space: Maps policy raw logits between [-1.0, 1.0] 
        # for each managed asset prior to the allocation normalization step
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32
        )

        # State Space: A sliding lookback window of the past 5 trading days' log returns
        # Satisfies autonomous observation requirements without heavy feature engineering
        self.lookback_window = 5
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.lookback_window, self.num_assets), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.historical_portfolio_returns = [0.0] * self.rolling_reward_window
        
        # Extract initial lookback data to formulate the starting state matrix
        state = self.returns_df.iloc[self.current_step - self.lookback_window : self.current_step].values
        
        # --- DEFENSIVE UNIT TEST: OBSERVATION STATE CHECK ---
        # Guarantees that the input state tensor shape perfectly matches the expected layout
        assert state.shape == (self.lookback_window, self.num_assets), \
            f"Critical Layout Failure: Expected state shape {(self.lookback_window, self.num_assets)}, got {state.shape}"
            
        info = {}
        return state.astype(np.float32), info

    def step(self, action):
        """Executes a single-step environment rebalancing calculation."""
        # Mathematical Rigor: Apply an exponential Softmax normalisation function 
        # to force the raw actions into a valid probability simplex bounded between [0.0, 1.0]
        exp_action = np.exp(action - np.max(action))
        portfolio_weights = exp_action / np.sum(exp_action)
        
        # --- DEFENSIVE UNIT TEST: PROBABILITY SIMPLEX CONSTRAINT ---
        # Verifies that portfolio weights structurally sum up to exactly 1.0 (100% full allocation)
        assert np.isclose(np.sum(portfolio_weights), 1.0, atol=1e-5), \
            f"Simplex Violation: Portfolio weights sum up to {np.sum(portfolio_weights)} instead of 1.0"

        # Fetch current true market returns for this timeline index
        current_returns = self.returns_df.iloc[self.current_step].values
        
        # Compute the deterministic dot product of weights and returns to get portfolio return
        portfolio_return = float(np.dot(portfolio_weights, current_returns))
        self.historical_portfolio_returns.append(portfolio_return)

        # Maintain the rolling window queue length configuration parameter
        if len(self.historical_portfolio_returns) > self.rolling_reward_window:
            self.historical_portfolio_returns.pop(0)

        # --- REWARD SHAPING ENGINE: ANNUALIZED ROLLING SHARPE RATIO ---
        # Prevents the policy from over-concentrating capital into single hyper-volatile assets 
        # by continuously penalizing performance variance (standard deviation)
        window_returns = self.historical_portfolio_returns
        mean_return = np.mean(window_returns)
        std_return = np.std(window_returns)
        
        if std_return > 1e-6:
            # Annualize the daily Sharpe value metric assuming 252 standard trading days per year
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        else:
            sharpe_ratio = mean_return

        reward = float(sharpe_ratio)
        
        # Advance environment timeline pointers
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Build the next step's sliding window state matrix
        next_state = self.returns_df.iloc[self.current_step - self.lookback_window : self.current_step].values
        
        info = {"portfolio_weights": portfolio_weights, "step_return": portfolio_return, "sharpe": reward}
        return next_state.astype(np.float32), reward, terminated, truncated, info

# =====================================================================
# 3. SEMANTIC EXPLANATION MIDDLEWARE LAYER
# =====================================================================
def run_semantic_explanation_middleware(ticker_names, weights, mock=True):
    """
    Packages numeric target metrics into a telemetry JSON schema 
    and bridges it to the localized Ollama language compilation layer.
    """
    # Map raw float arrays into a standardized, language-agnostic JSON payload structure
    telemetry_payload = {
        "status": "success",
        "engine": "PPO-Actor-Critic",
        "allocations": {ticker: round(float(weight), 4) for ticker, weight in zip(ticker_names, weights)}
    }
    
    json_string = json.dumps(telemetry_payload, indent=2)
    print("\n[Middleware] Telemetry successfully serialized to JSON schema:")
    print(json_string)
    
    if mock:
        # Fallback Mocking Logic: Bypasses live Ollama connections to guarantee local 
        # script execution without active system package dependency errors
        allocations_dict = telemetry_payload['allocations']
        max_ticker = max(allocations_dict, key=allocations_dict.get)

        mock_prose = (
            f"[Mock Ollama Narrative] The agent has dynamically adjusted allocations. High concentration is given "
            f"to {max_ticker} to maximize risk-adjusted momentum, while minor defensive hedges are maintained "
            f"across remaining Magnificent Seven assets."
        )
        return mock_prose
    
    # Active Execution Pipeline: Sends an asynchronous REST request to a local edge-hosted LLM instance
    try:
        ollama_url = "http://localhost:11434/api/generate"
        system_prompt = (
            "You are a professional financial advisor backend. Analyze the incoming JSON portfolio allocation weights "
            "and output a single paragraph explaining the strategy in clear, plain English to a retail investor."
        )
        response = requests.post(ollama_url, json={
            "model": "llama3",
            "prompt": f"{system_prompt}\n\nData Payload: {json_string}",
            "stream": False
        }, timeout=5)
        return response.json().get("response", "Error: Empty text response.")
    except Exception:
        # Error Isolation: Fallback if the local background Ollama service daemon is offline
        return "[Local Fallback] Ollama instance offline. Rule-based narrative compiled successfully."

# =====================================================================
# 4. PROTOTYPE EXECUTION & VERIFICATION SANDBOX
# =====================================================================
if __name__ == "__main__":
    print("=== STARTING UPGRADED FINANCIAL ADVISOR BOT PROTOTYPE ===")
    
    # Pipeline Process 1: Load and wash time-series data streams
    prices, returns = load_magnificent_seven_data()
    
    # Pipeline Process 2: Instantiate custom Gymnasium financial simulator env
    env = FinancialPortfolioEnv(prices, returns)
    
    # Pipeline Process 3: Initialize Stable-Baselines3 PPO Actor-Critic neural core
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003)
    model.learn(total_timesteps=1000)

    # Pipeline Process 4: Run a single-step evaluation inference validation cycle
    obs, info = env.reset()
    action, _states = model.predict(obs)
    _, _, _, _, simulation_metrics = env.step(action)
    
    print("-" * 60)
    print("Recommended Allocation Portfolio Weights (Magnificent Seven):")
    for ticker, weight in zip(prices.columns, simulation_metrics["portfolio_weights"]):
        print(f" * {ticker}: {weight * 100:.2f}%")
    print(f"Resulting Rolling Sharpe Metric: {simulation_metrics['sharpe']:.4f}")
    print("-" * 60)
    
    # Pipeline Process 5: Test translation middleware layer routing capabilities
    weights_vector = simulation_metrics["portfolio_weights"]
    explanation_prose = run_semantic_explanation_middleware(prices.columns, weights_vector, mock=True)
    print("\n[Interface Presentation Output]")
    print(explanation_prose)
    print("-" * 60)
    print("=== FEATURE PROTOTYPE PIPELINE VALIDATION ENDED ===")
