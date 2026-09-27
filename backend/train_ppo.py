import os
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from environment import FinancialPortfolioEnv

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
LOOKBACK = 5

# Resolve script-relative directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100.0  # Normalized between 0.0 and 1.0

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return (macd - signal_line) / series  # Price-normalized MACD difference

def build_data_pipeline():
    print(f"[Pipeline] Fetching data for {TICKERS} from {START_DATE} to {END_DATE}...")
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    
    close_prices = raw['Close'] if 'Close' in raw.columns else raw.xs('Close', axis=1, level=0)
    close_prices = close_prices.ffill().bfill()
    
    # 1. Log Returns
    log_returns = np.log(close_prices / close_prices.shift(1))
    
    # 2. RSI & MACD Calculation
    rsi_frames = pd.DataFrame(index=close_prices.index)
    macd_frames = pd.DataFrame(index=close_prices.index)
    
    for ticker in TICKERS:
        rsi_frames[ticker] = calculate_rsi(close_prices[ticker])
        macd_frames[ticker] = calculate_macd(close_prices[ticker])
        
    combined_df = pd.concat([log_returns, rsi_frames, macd_frames], axis=1).dropna()
    valid_indices = combined_df.index
    returns_matrix = log_returns.loc[valid_indices].values

    # 3. Stack into (Samples, Lookback, Features) Tensor
    num_samples = len(valid_indices) - LOOKBACK
    features_per_day = len(TICKERS) * 3  # returns + rsi + macd
    tensor = np.zeros((num_samples, LOOKBACK, features_per_day), dtype=np.float32)

    for i in range(num_samples):
        window_data = combined_df.iloc[i : i + LOOKBACK].values
        tensor[i] = window_data

    aligned_returns = returns_matrix[LOOKBACK:]
    print(f"[Pipeline] Processed {num_samples} time steps with observation tensor shape {tensor.shape}.")
    return tensor, aligned_returns

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    feature_tensor, returns_matrix = build_data_pipeline()
    env = FinancialPortfolioEnv(feature_tensor, returns_matrix, TICKERS, lookback_window=LOOKBACK)

    print("[DRL Core] Initializing Proximal Policy Optimization (PPO)...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        clip_range=0.2,
        gamma=0.99,
        verbose=1
    )

    print("[DRL Core] Training agent across 50,000 timesteps...")
    model.learn(total_timesteps=50000)

    model_path = os.path.join(SAVE_DIR, "ppo_mag7_agent.zip")
    model.save(model_path)
    print(f"[DRL Core] Policy network saved successfully to: {model_path}")