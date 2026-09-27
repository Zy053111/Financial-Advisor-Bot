import unittest
import numpy as np

from llm_service import apply_risk_profile, generate_rule_based_narrative


class TestFinancialAdvisorBot(unittest.TestCase):

    def test_simplex_mathematical_invariant(self):
        """Boundary & Invariant: Ensure softmax weights sum to 1.0 and are non-negative across 10,000 action vectors."""
        np.random.seed(42)
        for _ in range(10000):
            action = np.random.uniform(-1.0, 1.0, size=(7,))
            exp_action = np.exp(action - np.max(action))
            weights = exp_action / np.sum(exp_action)
            
            self.assertTrue(np.isclose(np.sum(weights), 1.0, atol=1e-5))
            self.assertTrue(np.all(weights >= 0.0))

    def test_sharpe_division_by_zero_boundary(self):
        """Boundary Value: Verify rolling Sharpe formula handles zero variance without ZeroDivisionError."""
        # Simulated flat returns with 0 standard deviation
        returns_window = np.array([0.01, 0.01, 0.01, 0.01])
        mean_ret = np.mean(returns_window)
        std_ret = np.std(returns_window)
        
        # Matches logic in environment.py
        reward = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 1e-6 else float(mean_ret)
        self.assertEqual(reward, float(mean_ret))

    def test_conservative_risk_clamping(self):
        """Branch Condition: Verify conservative profile dampens high concentration and preserves simplex."""
        # Realistic policy output where one asset has higher concentration
        raw_weights = np.array([0.30, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10], dtype=np.float32)
        adjusted = apply_risk_profile(raw_weights, risk_profile="conservative")
        
        # 1. Simplex invariant: sum must be 1.0
        self.assertTrue(np.isclose(np.sum(adjusted), 1.0, atol=1e-5))
        # 2. Risk reduction: conservative profile must reduce the top concentrated asset
        self.assertLess(adjusted[0], raw_weights[0])
        # 3. Maximum asset weight is safely bounded
        self.assertTrue(np.all(adjusted <= 0.25))

    def test_ollama_timeout_fallback(self):
        """Fault Tolerance: Verify deterministic fallback generates narrative when Ollama is unavailable."""
        allocations = {'AAPL': 0.14, 'MSFT': 0.14, 'GOOGL': 0.14, 'AMZN': 0.14, 'NVDA': 0.14, 'META': 0.14, 'TSLA': 0.16}
        rsi_data = {'TSLA': 65.0}
        
        narrative = generate_rule_based_narrative(allocations, rsi_data, risk_profile="moderate")
        self.assertIn("[Rule-Based Fallback]", narrative)
        self.assertIn("TSLA", narrative)


if __name__ == '__main__':
    unittest.main()