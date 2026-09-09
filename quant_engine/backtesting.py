import numpy as np
from scipy import stats

def kupiec_pof_test(returns: np.ndarray, var_forecasts: np.ndarray, alpha: float = 0.95) -> dict:
    """
    Kupiec Proportion of Failures (POF) Likelihood Ratio Backtest.
    H0: Model breach probability equals theoretical target p = 1 - alpha.
    """
    returns = np.asarray(returns, dtype=float)
    var_forecasts = np.asarray(var_forecasts, dtype=float)
    
    p_target = 1.0 - alpha
    losses = -returns
    breaches = losses > var_forecasts
    
    x = int(np.sum(breaches))
    N = int(len(returns))
    
    if N == 0:
        return {'p_value': 1.0, 'breach_rate': 0.0, 'status': 'INSUFFICIENT_DATA'}
        
    pi_hat = x / N
    
    if p_target == 0 or p_target == 1:
        log_L0 = 0.0
    else:
        log_L0 = (N - x) * np.log(1.0 - p_target) + x * np.log(p_target)
        
    if pi_hat == 0 or pi_hat == 1:
        log_L1 = 0.0
    else:
        log_L1 = (N - x) * np.log(1.0 - pi_hat) + x * np.log(pi_hat)
        
    lr_stat = -2.0 * (log_L0 - log_L1)
    lr_stat = max(0.0, lr_stat)
    
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
    
    return {
        'total_observations': N,
        'observed_breaches': x,
        'breach_rate': pi_hat,
        'target_breach_rate': p_target,
        'lr_stat': float(lr_stat),
        'p_value': p_value,
        'reject_null_at_5pct': bool(p_value < 0.05)
    }

def christoffersen_independence_test(returns: np.ndarray, var_forecasts: np.ndarray) -> dict:
    """
    Christoffersen Independence Likelihood Ratio Backtest.
    H0: VaR exceptions are independent over time (no clustering).
    """
    returns = np.asarray(returns, dtype=float)
    var_forecasts = np.asarray(var_forecasts, dtype=float)
    
    losses = -returns
    breaches = (losses > var_forecasts).astype(int)
    
    if len(breaches) < 2:
        return {'p_value': 1.0, 'status': 'INSUFFICIENT_DATA'}
        
    n00 = np.sum((breaches[:-1] == 0) & (breaches[1:] == 0))
    n01 = np.sum((breaches[:-1] == 0) & (breaches[1:] == 1))
    n10 = np.sum((breaches[:-1] == 1) & (breaches[1:] == 0))
    n11 = np.sum((breaches[:-1] == 1) & (breaches[1:] == 1))
    
    pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0
    
    log_L0 = (n00 + n10) * np.log(1.0 - pi + 1e-12) + (n01 + n11) * np.log(pi + 1e-12)
    
    log_L1 = (
        n00 * np.log(1.0 - pi_0 + 1e-12) +
        n01 * np.log(pi_0 + 1e-12) +
        n10 * np.log(1.0 - pi_1 + 1e-12) +
        n11 * np.log(pi_1 + 1e-12)
    )
    
    lr_ind = -2.0 * (log_L0 - log_L1)
    lr_ind = max(0.0, lr_ind)
    
    p_value = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    
    return {
        'n00': int(n00), 'n01': int(n01),
        'n10': int(n10), 'n11': int(n11),
        'pi_0': float(pi_0),
        'pi_1': float(pi_1),
        'lr_ind': float(lr_ind),
        'p_value': p_value,
        'reject_independence_at_5pct': bool(p_value < 0.05)
    }
