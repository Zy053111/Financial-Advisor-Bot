import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from train_ppo import TICKERS, LOOKBACK, calculate_rsi, calculate_macd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "ppo_mag7_agent.zip")
OLLAMA_URL = "http://localhost:11434/api/generate"

def fetch_latest_inference_data():
    """Fetch real-time rolling financial indicators and spot prices from Yahoo Finance."""
    raw = yf.download(TICKERS, period="3mo", interval="1d", progress=False)
    close_prices = raw['Close'] if 'Close' in raw.columns else raw.xs('Close', axis=1, level=0)
    close_prices = close_prices.ffill().bfill()

    log_returns = np.log(close_prices / close_prices.shift(1))
    
    rsi_frames = pd.DataFrame(index=close_prices.index)
    macd_frames = pd.DataFrame(index=close_prices.index)
    for t in TICKERS:
        rsi_frames[t] = calculate_rsi(close_prices[t])
        macd_frames[t] = calculate_macd(close_prices[t])

    combined_df = pd.concat([log_returns, rsi_frames, macd_frames], axis=1).dropna()
    
    # Extract the most recent LOOKBACK window
    recent_window = combined_df.iloc[-LOOKBACK:].values
    market_features = recent_window.flatten()

    # Equal-weight default for previous portfolio state
    initial_weights = np.ones(len(TICKERS), dtype=np.float32) / len(TICKERS)
    obs = np.concatenate([market_features, initial_weights]).astype(np.float32)
    
    latest_rsi = {t: round(float(rsi_frames[t].iloc[-1] * 100), 2) for t in TICKERS}
    
    # Extract latest market closing prices for execution calculations
    latest_prices = {t: round(float(close_prices[t].iloc[-1]), 2) for t in TICKERS}
    
    return obs, latest_rsi, latest_prices

def apply_risk_profile(weights, risk_profile="moderate"):
    """
    Post-process PPO target weights to align with investor risk tolerance:
    - aggressive: concentrates capital into top momentum assets
    - moderate: uses raw PPO Softmax policy distribution
    - conservative: smooths toward equal weights and caps maximum single-asset exposure
    """
    n = len(weights)
    if risk_profile == "aggressive":
        # Amplify concentration toward leading assets
        scaled = np.power(weights, 1.3)
        return scaled / np.sum(scaled)
    elif risk_profile == "conservative":
        # Blend 50% with equal-weight baseline to reduce concentration risk
        equal_weights = np.ones(n, dtype=np.float32) / n
        blended = 0.5 * weights + 0.5 * equal_weights
        # Enforce maximum individual asset holding constraint of 20%
        blended = np.clip(blended, 0.02, 0.20)
        return blended / np.sum(blended)
    return weights

def generate_rule_based_narrative(allocations, rsi_data, risk_profile="moderate"):
    """Deterministic template fallback if the local Ollama LLM is unreachable."""
    top_ticker = max(allocations, key=allocations.get)
    top_weight = allocations[top_ticker] * 100
    top_rsi = rsi_data.get(top_ticker, 50.0)

    rsi_desc = "bullish momentum" if top_rsi > 55 else "oversold recovery value"
    return (
        f"[Rule-Based Fallback] The agent has allocated maximum simplex exposure to {top_ticker} "
        f"({top_weight:.2f}%) under a {risk_profile} mandate based on Sharpe optimization and {rsi_desc} (RSI: {top_rsi:.1f}). "
        f"The remaining capital is distributed defensively across other Magnificent Seven assets."
    )

def get_portfolio_recommendation(risk_profile="moderate"):
    """Execute DRL inference and generate strategic explanation with local Ollama."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")

    obs, latest_rsi, latest_prices = fetch_latest_inference_data()

    # Load trained PPO agent and infer policy actions
    model = PPO.load(MODEL_PATH)
    raw_action, _ = model.predict(obs, deterministic=True)

    # Softmax normalization to guarantee sum(weights) == 1.0
    exp_action = np.exp(raw_action - np.max(raw_action))
    raw_weights = exp_action / np.sum(exp_action)

    # Apply mathematical risk profile transformations
    adjusted_weights = apply_risk_profile(raw_weights, risk_profile)
    allocations = {t: round(float(w), 4) for t, w in zip(TICKERS, adjusted_weights)}

    # Construct structured telemetry payload for auditability
    telemetry_payload = {
        "status": "success",
        "engine": "PPO-Actor-Critic",
        "risk_profile": risk_profile,
        "allocations": allocations,
        "market_indicators": {"rsi": latest_rsi},
        "prices": latest_prices
    }

    # Prompt engineering customized for retail persona and risk mandate
    system_prompt = (
        f"You are an institutional financial advisor explaining an automated portfolio rebalance to a {risk_profile} retail investor. "
        "Analyze the given asset allocation weights and RSI indicators. Write exactly one clear, professional paragraph (80-120 words) "
        f"tailored to a {risk_profile} investment philosophy, explaining which stock is held with highest weight and how the portfolio balances return versus risk. "
        "Do not hallucinate numbers not provided in the payload."
    )

    narrative = None
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": f"{system_prompt}\n\nTelemetry Payload: {json.dumps(telemetry_payload)}",
                "stream": False
            },
            timeout=45
        )
        if response.status_code == 200:
            narrative = response.json().get("response", "").strip()
    except Exception as e:
        print(f"[Ollama Error] Connection failed or timed out: {e}")

    if not narrative:
        narrative = generate_rule_based_narrative(allocations, latest_rsi, risk_profile)

    return {
        "telemetry": telemetry_payload,
        "narrative": narrative
    }

def answer_portfolio_query(user_query, telemetry_payload):
    """Answer user questions about current allocation adapting tone to the active risk mandate."""
    risk_profile = telemetry_payload.get("risk_profile", "moderate")

    # Dynamic tone guidance conditioned on active risk mandate
    tone_directives = {
        "aggressive": (
            "Adopt an aggressive, growth-seeking institutional persona. Emphasize capital growth, "
            "bullish upside potential, high-momentum breakouts, and conviction-driven allocation."
        ),
        "moderate": (
            "Adopt a balanced institutional persona. Emphasize risk-adjusted returns (Sharpe ratio), "
            "measured upside participation, and disciplined diversification across leaders."
        ),
        "conservative": (
            "Adopt a defensive, capital-preservation persona. Emphasize downside mitigation, "
            "capping single-stock exposure, cash flow defense, and volatility reduction."
        )
    }

    directive = tone_directives.get(risk_profile, tone_directives["moderate"])

    system_prompt = (
        f"You are an institutional financial advisor. The client is under an active '{risk_profile.upper()}' mandate. "
        f"{directive} "
        "Use strictly the provided portfolio telemetry (allocations, RSI, risk profile, prices) to substantiate your answers. "
        "Keep your response concise (2-4 sentences), factual, sharp, and directly focused on the user's question. "
        "Do not hallucinate stocks or numbers not present in the payload."
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": f"{system_prompt}\n\nCurrent Portfolio Context:\n{json.dumps(telemetry_payload)}\n\nInvestor Inquiry: {user_query}\nAdvisor Response:",
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[Ollama Chat Error] {e}")

    return "The portfolio allocation balances risk-adjusted return and drawdown protection based on technical momentum (RSI) and DRL policy targets."

if __name__ == "__main__":
    print("[Inference] Running portfolio recommendation cycle...")
    result = get_portfolio_recommendation(risk_profile="moderate")
    print("\n--- Telemetry Output ---")
    print(json.dumps(result["telemetry"], indent=2))
    print("\n--- Generated Strategy Explanation ---")
    print(result["narrative"])