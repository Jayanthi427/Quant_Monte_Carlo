import numpy as np
from scipy import stats

def black_scholes_call(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Computes exact analytical European Call Price under Black-Scholes-Merton model."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S0 - K)
    
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return float(call_price)

def monte_carlo_option_pricing(
    S0: float, K: float, T: float, r: float, q: float, sigma: float, 
    num_paths: int = 100000, use_antithetic: bool = True, seed: int = 42
) -> dict:
    """
    Monte Carlo European Call Pricing under Risk-Neutral Q-Measure.
    Guarantees convergence to Black-Scholes as num_paths -> infinity.
    """
    rng = np.random.default_rng(seed)
    
    if use_antithetic:
        half_paths = num_paths // 2
        z_half = rng.standard_normal(half_paths)
        z = np.concatenate([z_half, -z_half])
    else:
        z = rng.standard_normal(num_paths)
        
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoffs = np.maximum(ST - K, 0.0)
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    mc_price = float(np.mean(discounted_payoffs))
    std_err = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs)))
    ci_95 = (mc_price - 1.95996 * std_err, mc_price + 1.95996 * std_err)
    
    bs_price = black_scholes_call(S0, K, T, r, q, sigma)
    pricing_error = mc_price - bs_price
    
    return {
        'mc_price': mc_price,
        'bs_price': bs_price,
        'std_err': std_err,
        'ci_95': ci_95,
        'pricing_error': pricing_error,
        'relative_error_pct': (pricing_error / bs_price) * 100.0 if bs_price > 0 else 0.0
    }
