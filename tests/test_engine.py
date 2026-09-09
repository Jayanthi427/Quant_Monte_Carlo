import pytest
import numpy as np
from quant_engine.pricing import black_scholes_call, monte_carlo_option_pricing
from quant_engine.backtesting import kupiec_pof_test, christoffersen_independence_test

def test_black_scholes_known_values():
    # ATM European Call Benchmark
    price = black_scholes_call(S0=100.0, K=100.0, T=1.0, r=0.05, q=0.01, sigma=0.20)
    assert round(price, 4) == 9.8263

def test_monte_carlo_bs_convergence():
    # Monte Carlo price must fall within 95% CI of Black-Scholes benchmark
    res = monte_carlo_option_pricing(
        S0=100.0, K=100.0, T=1.0, r=0.05, q=0.01, sigma=0.20,
        num_paths=100000, seed=42
    )
    assert res['ci_95'][0] <= res['bs_price'] <= res['ci_95'][1]
    assert abs(res['pricing_error']) < 0.05

def test_kupiec_pof_ideal_data():
    # 5% breach rate under 95% VaR should fail to reject H0
    np.random.seed(42)
    returns = np.random.normal(0, 1, 1000)
    var_forecasts = np.full(1000, 1.64485)
    res = kupiec_pof_test(returns, var_forecasts, alpha=0.95)
    assert not res['reject_null_at_5pct']
    assert res['p_value'] > 0.05

def test_kupiec_pof_bad_data():
    # 15% breach rate under 95% VaR must reject H0
    np.random.seed(42)
    returns = np.random.normal(0, 2, 1000)
    var_forecasts = np.full(1000, 1.64485)
    res = kupiec_pof_test(returns, var_forecasts, alpha=0.95)
    assert res['reject_null_at_5pct']
    assert res['p_value'] < 0.05

def test_christoffersen_independence():
    # Independent exceptions should pass independence test
    np.random.seed(42)
    returns = np.random.normal(0, 1, 1000)
    var_forecasts = np.full(1000, 1.64485)
    res = christoffersen_independence_test(returns, var_forecasts)
    assert not res['reject_independence_at_5pct']
