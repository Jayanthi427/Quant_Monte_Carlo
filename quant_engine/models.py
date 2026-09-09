import numpy as np
from quant_engine.simulation import MonteCarloSimulator

class ModelComparisonEngine:
    """
    Compares candidate stochastic models under rigorous quantitative criteria.
    """
    @staticmethod
    def run_comparison(S0: float, sigma: float, mu: float, r: float, days: int, df: float = 5.0) -> dict:
        sim_t = MonteCarloSimulator(S0=S0, sigma=sigma, mu_P=mu, r=r, df=df, days=days, paths=5000, seed=42)
        p_paths_t = sim_t.simulate_p_measure()
        ret_t = np.diff(np.log(p_paths_t), axis=1).flatten()

        sim_norm = MonteCarloSimulator(S0=S0, sigma=sigma, mu_P=mu, r=r, df=1000.0, days=days, paths=5000, seed=42)
        p_paths_norm = sim_norm.simulate_p_measure()
        ret_norm = np.diff(np.log(p_paths_norm), axis=1).flatten()

        return {
            "Student-t Monte Carlo": {
                "Tail Risk (Excess Kurtosis)": float((ret_t**4).mean() / (ret_t.var()**2) - 3.0),
                "VaR 95%": float(-np.percentile(ret_t, 5)),
                "Expected Shortfall 95%": float(-ret_t[ret_t <= np.percentile(ret_t, 5)].mean()),
                "Fat Tail Capable": "Yes"
            },
            "Standard Geometric Brownian Motion": {
                "Tail Risk (Excess Kurtosis)": float((ret_norm**4).mean() / (ret_norm.var()**2) - 3.0),
                "VaR 95%": float(-np.percentile(ret_norm, 5)),
                "Expected Shortfall 95%": float(-ret_norm[ret_norm <= np.percentile(ret_norm, 5)].mean()),
                "Fat Tail Capable": "No"
            }
        }
