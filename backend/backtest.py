import os
import json
import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from stable_baselines3 import PPO
from train_ppo import TICKERS, LOOKBACK, calculate_rsi, calculate_macd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "ppo_mag7_agent.zip")
TEST_START = "2025-01-01"
TEST_END = "2026-08-30"

# Fetch and prepare aligned historical price data
def get_backtest_data():
    print(f"[Backtest Engine] Downloading out-of-sample data ({TEST_START} to {TEST_END})...")
    # Fetch extra lookback window to calculate technical indicators cleanly
    raw = yf.download(TICKERS, start="2024-10-01", end=TEST_END, progress=False)
    
    close_prices = raw['Close'] if 'Close' in raw.columns else raw.xs('Close', axis=1, level=0)
    close_prices = close_prices.ffill().bfill()
    
    log_returns = np.log(close_prices / close_prices.shift(1))
    rsi_frames = pd.DataFrame(index=close_prices.index)
    macd_frames = pd.DataFrame(index=close_prices.index)
    
    for t in TICKERS:
        rsi_frames[t] = calculate_rsi(close_prices[t])
        macd_frames[t] = calculate_macd(close_prices[t])
        
    combined_df = pd.concat([log_returns, rsi_frames, macd_frames], axis=1).dropna()
    filtered_prices = close_prices.loc[close_prices.index >= TEST_START]
    return raw, close_prices, combined_df, filtered_prices

# Strategy 1: PPO Dynamic Rebalance Strategy
class PPORebalanceStrategy(bt.Strategy):
    params = (
        ('model_path', MODEL_PATH),
        ('combined_df', None),
        ('lookback', LOOKBACK),
        ('tickers', TICKERS),
    )

    def __init__(self):
        self.model = PPO.load(self.p.model_path)
        self.tickers = self.p.tickers
        self.combined_df = self.p.combined_df
        self.last_weights = np.ones(len(self.tickers), dtype=np.float32) / len(self.tickers)
        self.history_values = []
        self.dates = []

    def next(self):
        dt = self.datas[0].datetime.date(0)
        dt_str = dt.strftime('%Y-%m-%d')
        
        self.history_values.append(self.broker.getvalue())
        self.dates.append(dt)

        if dt_str not in self.combined_df.index.strftime('%Y-%m-%d'):
            return

        idx = self.combined_df.index.get_indexer([pd.Timestamp(dt_str)], method='pad')[0]
        if idx < self.p.lookback:
            return

        window_data = self.combined_df.iloc[idx - self.p.lookback + 1 : idx + 1].values
        market_features = window_data.flatten()
        obs = np.concatenate([market_features, self.last_weights]).astype(np.float32)

        raw_action, _ = self.model.predict(obs, deterministic=True)
        exp_action = np.exp(raw_action - np.max(raw_action))
        target_weights = exp_action / np.sum(exp_action)

        for i, ticker in enumerate(self.tickers):
            data_feed = self.getdatabyname(ticker)
            self.order_target_percent(data_feed, target=float(target_weights[i]))

        self.last_weights = target_weights

# Strategy 2: Markowitz Modern Portfolio Theory (Rolling Maximum Sharpe Ratio)
class MarkowitzStrategy(bt.Strategy):
    params = (
        ('tickers', TICKERS),
        ('rebalance_days', 21),    # Rebalance every ~21 trading days (monthly)
        ('estimation_window', 63)  # Use 3-month trailing window for covariance matrix
    )

    def __init__(self):
        self.tickers = self.p.tickers
        self.history_values = []
        self.dates = []
        self.day_count = 0

    def optimize_weights(self):
        prices = []
        for ticker in self.tickers:
            data_feed = self.getdatabyname(ticker)
            prices.append([data_feed.close[-i] for i in range(self.p.estimation_window, 0, -1)])
        
        returns = np.log(np.array(prices)[:, 1:] / np.array(prices)[:, :-1])
        mean_returns = np.mean(returns, axis=1) * 252
        cov_matrix = np.cov(returns) * 252

        num_assets = len(self.tickers)
        
        def neg_sharpe(weights):
            p_ret = np.sum(mean_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) + 1e-8)
            return -(p_ret - 0.02) / p_vol

        bounds = tuple((0.0, 0.40) for _ in range(num_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        init_guess = np.ones(num_assets) / num_assets

        res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x if res.success else init_guess

    def next(self):
        self.history_values.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.date(0))
        self.day_count += 1

        if len(self.datas[0]) < self.p.estimation_window:
            return

        if self.day_count % self.p.rebalance_days == 0:
            target_weights = self.optimize_weights()
            for i, ticker in enumerate(self.tickers):
                data_feed = self.getdatabyname(ticker)
                self.order_target_percent(data_feed, target=float(target_weights[i]))

# Strategy 3: 1/N Buy-and-Hold Benchmark Strategy
class Benchmark1NStrategy(bt.Strategy):
    params = (
        ('tickers', TICKERS),
    )

    def __init__(self):
        self.tickers = self.p.tickers
        self.allocated = False
        self.history_values = []
        self.dates = []

    def next(self):
        self.history_values.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.date(0))

        if not self.allocated:
            equal_weight = 1.0 / len(self.tickers)
            for ticker in self.tickers:
                data_feed = self.getdatabyname(ticker)
                self.order_target_percent(data_feed, target=equal_weight)
            self.allocated = True

def run_simulation(strategy_class, raw_data, combined_df=None):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    
    # 0.10% Transaction Fee (10 bps) + 0.05% Slippage Model
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

    if strategy_class == PPORebalanceStrategy:
        cerebro.addstrategy(strategy_class, combined_df=combined_df)
    else:
        cerebro.addstrategy(strategy_class)

    results = cerebro.run()
    strat = results[0]
    
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)
    total_return = strat.analyzers.returns.get_analysis().get('rtot', 0.0)
    
    return {
        'final_value': cerebro.broker.getvalue(),
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'values': strat.history_values,
        'dates': strat.dates
    }

def export_backtest_json(ppo_metrics, mpt_metrics, bench_metrics):
    min_len = min(len(ppo_metrics['dates']), len(mpt_metrics['dates']), len(bench_metrics['dates']))
    history_data = []
    
    for i in range(min_len):
        dt_str = ppo_metrics['dates'][i].strftime('%Y-%m-%d')
        history_data.append({
            "date": dt_str,
            "ppo_value": round(float(ppo_metrics['values'][i]), 2),
            "mpt_value": round(float(mpt_metrics['values'][i]), 2),
            "benchmark_value": round(float(bench_metrics['values'][i]), 2)
        })
        
    json_path = os.path.join(BASE_DIR, "backtest_history.json")
    with open(json_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"[JSON Export] Time-series history saved to: {json_path}")

if __name__ == "__main__":
    raw_data, close_prices, combined_df, _ = get_backtest_data()

    print("\n[1/3] Running PPO Rebalance Strategy...")
    ppo_metrics = run_simulation(PPORebalanceStrategy, raw_data, combined_df)

    print("\n[2/3] Running Markowitz (MPT) Strategy...")
    mpt_metrics = run_simulation(MarkowitzStrategy, raw_data)

    print("\n[3/3] Running 1/N Equal-Weight Benchmark Strategy...")
    bench_metrics = run_simulation(Benchmark1NStrategy, raw_data)

    print("\n" + "=" * 80)
    print(f"{'Metric':<22} | {'PPO Strategy':<15} | {'Markowitz (MPT)':<15} | {'1/N Benchmark':<15}")
    print("=" * 80)
    print(f"{'Initial Capital':<22} | $100,000.00     | $100,000.00     | $100,000.00")
    print(f"{'Final Value':<22} | ${ppo_metrics['final_value']:,.2f}   | ${mpt_metrics['final_value']:,.2f}   | ${bench_metrics['final_value']:,.2f}")
    print(f"{'Cumulative Return':<22} | {ppo_metrics['total_return']*100:.2f}%          | {mpt_metrics['total_return']*100:.2f}%          | {bench_metrics['total_return']*100:.2f}%")
    print(f"{'Sharpe Ratio':<22} | {ppo_metrics['sharpe']:.4f}          | {mpt_metrics['sharpe']:.4f}          | {bench_metrics['sharpe']:.4f}")
    print(f"{'Max Drawdown':<22} | -{ppo_metrics['max_drawdown']:.2f}%          | -{mpt_metrics['max_drawdown']:.2f}%          | -{bench_metrics['max_drawdown']:.2f}%")
    print(f"{'Friction Model':<22} | 0.10% + Slippage| 0.10% + Slippage| 0.10% + Slippage")
    print("=" * 80)

    # Export three-way comparison chart
    plt.figure(figsize=(12, 6))
    min_len = min(len(ppo_metrics['dates']), len(mpt_metrics['dates']), len(bench_metrics['dates']))
    plt.plot(ppo_metrics['dates'][:min_len], ppo_metrics['values'][:min_len], label="PPO DRL Strategy", color="#10B981", linewidth=2)
    plt.plot(mpt_metrics['dates'][:min_len], mpt_metrics['values'][:min_len], label="Markowitz MPT (Max Sharpe)", color="#3B82F6", linewidth=1.5)
    plt.plot(bench_metrics['dates'][:min_len], bench_metrics['values'][:min_len], label="1/N Benchmark", color="#94A3B8", linestyle="--", linewidth=1.2)
    plt.title("Out-of-Sample Alpha Trajectory: DRL vs. Markowitz vs. 1/N Benchmark", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Portfolio Value ($)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    
    chart_output = os.path.join(BASE_DIR, "backtest_result.png")
    plt.savefig(chart_output, dpi=300, bbox_inches="tight")
    print(f"\n[Chart Saved] Performance trajectory saved as: {chart_output}")

    # Export multi-strategy time series to JSON
    export_backtest_json(ppo_metrics, mpt_metrics, bench_metrics)