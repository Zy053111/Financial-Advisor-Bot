import os
import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import FinancialPortfolioEnv
from train_ppo import TICKERS, START_DATE, END_DATE, LOOKBACK

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
FULL_MODEL_PATH = os.path.join(SAVE_DIR, "ppo_mag7_agent.zip")
ABLATION_MODEL_PATH = os.path.join(SAVE_DIR, "ppo_no_indicators_agent.zip")
TEST_START = "2025-01-01"
TEST_END = "2026-08-30"

# 1. 构建仅含收益率（剥离 RSI 和 MACD）的消融数据流水线
def build_ablation_pipeline():
    print(f"[Ablation Pipeline] Fetching raw data from {START_DATE} to {END_DATE} (Returns only)...")
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)
    close_prices = raw['Close'] if 'Close' in raw.columns else raw.xs('Close', axis=1, level=0)
    close_prices = close_prices.ffill().bfill()

    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    valid_indices = log_returns.index
    returns_matrix = log_returns.values

    # 特征数只有 7 (仅价格收益率，无 RSI 和 MACD)
    num_samples = len(valid_indices) - LOOKBACK
    features_per_day = len(TICKERS)
    tensor = np.zeros((num_samples, LOOKBACK, features_per_day), dtype=np.float32)

    for i in range(num_samples):
        window_data = log_returns.iloc[i : i + LOOKBACK].values
        tensor[i] = window_data

    aligned_returns = returns_matrix[LOOKBACK:]
    return tensor, aligned_returns

# 2. 训练消融模型
def train_ablation_model():
    tensor, aligned_returns = build_ablation_pipeline()
    env = FinancialPortfolioEnv(tensor, aligned_returns, TICKERS, lookback_window=LOOKBACK)

    print("[Ablation Train] Training PPO without technical indicators (50,000 steps)...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        clip_range=0.2,
        gamma=0.99,
        verbose=0
    )
    model.learn(total_timesteps=50000)
    model.save(ABLATION_MODEL_PATH)
    print(f"[Ablation Train] Ablated model saved to: {ABLATION_MODEL_PATH}")

# 3. 运行回测策略
class AblationPPOStrategy(bt.Strategy):
    params = (
        ('model_path', None),
        ('feature_df', None),
        ('lookback', LOOKBACK),
        ('tickers', TICKERS),
    )

    def __init__(self):
        self.model = PPO.load(self.p.model_path)
        self.tickers = self.p.tickers
        self.feature_df = self.p.feature_df
        self.last_weights = np.ones(len(self.tickers), dtype=np.float32) / len(self.tickers)
        self.history_values = []
        self.dates = []

    def next(self):
        dt = self.datas[0].datetime.date(0)
        dt_str = dt.strftime('%Y-%m-%d')

        self.history_values.append(self.broker.getvalue())
        self.dates.append(dt)

        if dt_str not in self.feature_df.index.strftime('%Y-%m-%d'):
            return

        idx = self.feature_df.index.get_indexer([pd.Timestamp(dt_str)], method='pad')[0]
        if idx < self.p.lookback:
            return

        window_data = self.feature_df.iloc[idx - self.p.lookback + 1 : idx + 1].values
        market_features = window_data.flatten()
        obs = np.concatenate([market_features, self.last_weights]).astype(np.float32)

        raw_action, _ = self.model.predict(obs, deterministic=True)
        exp_action = np.exp(raw_action - np.max(raw_action))
        target_weights = exp_action / np.sum(exp_action)

        for i, ticker in enumerate(self.tickers):
            data_feed = self.getdatabyname(ticker)
            self.order_target_percent(data_feed, target=float(target_weights[i]))

        self.last_weights = target_weights

def evaluate_strategy(model_path, feature_df, raw_data):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0010)
    cerebro.broker.set_slippage_perc(perc=0.0005)

    for ticker in TICKERS:
        ticker_df = raw_data.xs(ticker, axis=1, level=1) if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
        data_feed = bt.feeds.PandasData(
            dataname=ticker_df,
            fromdate=pd.to_datetime(TEST_START),
            todate=pd.to_datetime(TEST_END)
        )
        cerebro.adddata(data_feed, name=ticker)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02, timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    cerebro.addstrategy(AblationPPOStrategy, model_path=model_path, feature_df=feature_df)
    results = cerebro.run()
    strat = results[0]

    return {
        'final_value': cerebro.broker.getvalue(),
        'total_return': strat.analyzers.returns.get_analysis().get('rtot', 0.0),
        'sharpe': strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0),
        'max_drawdown': strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0),
        'values': strat.history_values,
        'dates': strat.dates
    }

if __name__ == "__main__":
    if not os.path.exists(ABLATION_MODEL_PATH):
        train_ablation_model()
    else:
        print("[Ablation] Found pre-trained ablated model.")

    from backtest import get_backtest_data
    raw_data, close_prices, full_features_df, _ = get_backtest_data()

    # 构建样本外期间纯价格收益率特征（消融版特征）
    ablation_features_df = np.log(close_prices / close_prices.shift(1)).dropna()

    print("\n[Evaluation] Evaluating Full PPO Model (Returns + RSI + MACD)...")
    full_metrics = evaluate_strategy(FULL_MODEL_PATH, full_features_df, raw_data)

    print("[Evaluation] Evaluating Ablated PPO Model (Returns Only)...")
    ablated_metrics = evaluate_strategy(ABLATION_MODEL_PATH, ablation_features_df, raw_data)

    print("\n" + "=" * 70)
    print("                    ABLATION STUDY RESULTS                          ")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Full PPO (Proposed)':<20} | {'Ablated (No Indicators)':<20}")
    print("-" * 70)
    print(f"{'Final Portfolio Value':<25} | ${full_metrics['final_value']:,.2f}          | ${ablated_metrics['final_value']:,.2f}")
    print(f"{'Cumulative Return':<25} | {full_metrics['total_return']*100:.2f}%               | {ablated_metrics['total_return']*100:.2f}%")
    print(f"{'Sharpe Ratio':<25} | {full_metrics['sharpe']:.4f}               | {ablated_metrics['sharpe']:.4f}")
    print(f"{'Max Drawdown':<25} | -{full_metrics['max_drawdown']:.2f}%               | -{ablated_metrics['max_drawdown']:.2f}%")
    print("=" * 70)

    # 导出消融对比图
    plt.figure(figsize=(11, 5.5))
    min_len = min(len(full_metrics['dates']), len(ablated_metrics['dates']))
    plt.plot(full_metrics['dates'][:min_len], full_metrics['values'][:min_len], label="Full Model (Returns + RSI + MACD)", color="#10B981", linewidth=2)
    plt.plot(ablated_metrics['dates'][:min_len], ablated_metrics['values'][:min_len], label="Ablated Model (Returns Only)", color="#F59E0B", linestyle="--", linewidth=1.8)
    plt.title("Ablation Study: Impact of Momentum Indicators on Out-of-Sample Performance", fontsize=13, pad=12)
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Portfolio Value ($)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")

    output_path = os.path.join(BASE_DIR, "ablation_result.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n[Artifact Generated] Ablation trajectory chart saved as: {output_path}")