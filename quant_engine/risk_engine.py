"""
QUANTITATIVE RISK ENGINE & MONTE CARLO SIMULATION MODULE
========================================================
Implements real-world (P-measure) forecasting, risk-neutral (Q-measure) option pricing,
stochastic process simulations, variance reduction, and statistical validation tests.
"""

import json
import time
import numpy as np
import pandas as pd
from scipy.stats import norm, t, chi2, skew, kurtosis, jarque_bera

class RiskAnalyticsEngine:

    @staticmethod
    def black_scholes_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculates closed-form Black-Scholes-Merton European call option price."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    @staticmethod
    def calculate_var_cvar(final_prices: np.ndarray, spot: float, cl: float = 0.95):
        """
        Calculates Value at Risk (VaR) and Expected Shortfall (CVaR).
        Explicitly defined as Positive Dollar Loss relative to initial Spot S_0.
        """
        alpha = 1.0 - cl
        p_alpha = np.percentile(final_prices, alpha * 100.0)
        var_loss = float(spot - p_alpha)
        
        tail_prices = final_prices[final_prices <= p_alpha]
        if len(tail_prices) > 0:
            cvar_loss = float(spot - np.mean(tail_prices))
        else:
            cvar_loss = var_loss
            
        return var_loss, cvar_loss, float(p_alpha), float(spot - cvar_loss)

    @staticmethod
    def bootstrap_var_se(final_prices: np.ndarray, spot: float, cl: float = 0.95, n_bootstraps: int = 500) -> float:
        """Computes Standard Error of VaR estimate via empirical bootstrapping."""
        rng = np.random.default_rng(42)
        n = len(final_prices)
        var_estimates = np.empty(n_bootstraps)
        alpha_pct = (1.0 - cl) * 100.0
        
        for i in range(n_bootstraps):
            sample = rng.choice(final_prices, size=n, replace=True)
            var_estimates[i] = spot - np.percentile(sample, alpha_pct)
            
        return float(np.std(var_estimates, ddof=1))

    @staticmethod
    def backtest_advanced_var(returns: pd.Series, window: int = 252, cl: float = 0.95):
        """
        Executes rolling-window historical VaR backtesting:
          - Kupiec Unconditional Coverage (LR_uc)
          - Christoffersen Independence (LR_ind)
          - Conditional Coverage (LR_cc)
        """
        alpha = 1.0 - cl
        clean_ret = returns.dropna()
        n_total = len(clean_ret)
        
        if n_total <= 50:
            return {
                "status": "INSUFFICIENT_DATA", "n_obs": 0, "expected_breaches": 0,
                "actual_breaches": 0, "breach_ratio": 0.0, "kupiec_p_value": 0.0,
                "independence_p_value": 0.0, "conditional_coverage_p_value": 0.0
            }

        actual_window = min(window, n_total // 2)
        eval_length = n_total - actual_window
        
        breaches = np.zeros(eval_length, dtype=int)
        for i in range(eval_length):
            hist_window = clean_ret.iloc[i : i + actual_window]
            var_threshold = np.percentile(hist_window, alpha * 100.0)
            if clean_ret.iloc[i + actual_window] < var_threshold:
                breaches[i] = 1

        x = int(np.sum(breaches))
        n = len(breaches)
        
        if n == 0:
            return {
                "status": "NO_OBSERVATIONS", "n_obs": 0, "expected_breaches": 0,
                "actual_breaches": 0, "breach_ratio": 0.0, "kupiec_p_value": 0.0,
                "independence_p_value": 0.0, "conditional_coverage_p_value": 0.0
            }

        p_hat = x / n
        p_hat_clamped = np.clip(p_hat, 1e-6, 1.0 - 1e-6)

        # Kupiec LR_uc
        lr_uc = -2.0 * ((n - x) * np.log(1.0 - alpha) + x * np.log(alpha) - 
                        ((n - x) * np.log(1.0 - p_hat_clamped) + x * np.log(p_hat_clamped)))
        p_val_uc = float(1.0 - chi2.cdf(max(0.0, lr_uc), df=1))

        # Christoffersen LR_ind
        n00 = n01 = n10 = n11 = 0
        for i in range(n - 1):
            b1, b2 = breaches[i], breaches[i+1]
            if b1 == 0 and b2 == 0: n00 += 1
            elif b1 == 0 and b2 == 1: n01 += 1
            elif b1 == 1 and b2 == 0: n10 += 1
            elif b1 == 1 and b2 == 1: n11 += 1

        pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / n if n > 0 else 0.0

        num = ((1.0 - pi)**(n00 + n10)) * (pi**(n01 + n11))
        den = ((1.0 - pi_0)**n00) * (pi_0**n01) * ((1.0 - pi_1)**n10) * (pi_1**n11) + 1e-12
        lr_ind = -2.0 * np.log(max(1e-12, num / den))
        p_val_ind = float(1.0 - chi2.cdf(max(0.0, lr_ind), df=1))

        # Conditional Coverage LR_cc
        lr_cc = lr_uc + lr_ind
        p_val_cc = float(1.0 - chi2.cdf(max(0.0, lr_cc), df=2))

        return {
            "n_obs": n,
            "expected_breaches": int(n * alpha),
            "actual_breaches": x,
            "breach_ratio": float(p_hat / alpha) if alpha > 0 else 0.0,
            "kupiec_p_value": p_val_uc,
            "independence_p_value": p_val_ind,
            "conditional_coverage_p_value": p_val_cc,
            "status": "PASSED" if p_val_cc > 0.05 else "REJECTED"
        }

    @staticmethod
    def evaluate_convergence(spot: float, mu_q: float, sigma: float, T: float, r: float, K: float, path_counts=[1000, 2500, 5000, 10000, 25000]):
        """Runs convergence diagnostics across increasing sample path counts."""
        results = []
        bs_price = RiskAnalyticsEngine.black_scholes_call(spot, K, T, r, 0.0, sigma)
        dt = T / 30.0
        
        for N in path_counts:
            t0 = time.time()
            rng = np.random.default_rng(42)
            half_n = N // 2
            Z = rng.standard_normal((half_n, 30))
            Z = np.vstack([Z, -Z])
            
            ST = spot * np.exp(np.sum((mu_q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))
            payoffs = np.exp(-r * T) * np.maximum(ST - K, 0.0)
            mc_price = float(np.mean(payoffs))
            se = float(np.std(payoffs, ddof=1) / np.sqrt(N))
            elapsed = (time.time() - t0) * 1000.0
            
            abs_err = abs(mc_price - bs_price)
            rel_err = (abs_err / bs_price * 100.0) if bs_price > 1e-6 else 0.0

            results.append({
                "paths": N,
                "mc_price": mc_price,
                "se": se,
                "ci_lower": mc_price - 1.96 * se,
                "ci_upper": mc_price + 1.96 * se,
                "abs_error": abs_err,
                "rel_error_%": rel_err,
                "runtime_ms": elapsed
            })
        return pd.DataFrame(results)

    @staticmethod
    def benchmark_variance_reduction(spot: float, mu_q: float, sigma: float, T: float, r: float, K: float, n_paths: int = 10000):
        """Compares estimation variance between Plain MC, Antithetic Variates, and Control Variates."""
        dt = T / 30.0
        steps = 30
        df_discount = np.exp(-r * T)

        # Plain Monte Carlo
        rng1 = np.random.default_rng(42)
        Z_plain = rng1.standard_normal((n_paths, steps))
        ST_plain = spot * np.exp(np.sum((mu_q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_plain, axis=1))
        payoffs_plain = df_discount * np.maximum(ST_plain - K, 0.0)
        var_plain = float(np.var(payoffs_plain, ddof=1))
        se_plain = float(np.std(payoffs_plain, ddof=1) / np.sqrt(n_paths))
        price_plain = float(np.mean(payoffs_plain))

        # Antithetic Variates
        rng2 = np.random.default_rng(42)
        half_n = n_paths // 2
        Z_half = rng2.standard_normal((half_n, steps))
        Z_anti = np.vstack([Z_half, -Z_half])
        ST_anti = spot * np.exp(np.sum((mu_q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_anti, axis=1))
        payoffs_anti = df_discount * np.maximum(ST_anti - K, 0.0)
        var_anti = float(np.var(payoffs_anti, ddof=1))
        se_anti = float(np.std(payoffs_anti, ddof=1) / np.sqrt(n_paths))
        price_anti = float(np.mean(payoffs_anti))

        # Control Variates (Asset Price as Control)
        exp_ST = spot * np.exp(mu_q * T)
        cov_matrix = np.cov(payoffs_plain, ST_plain)
        c_star = - cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0.0
        payoffs_cv = payoffs_plain + c_star * (ST_plain - exp_ST)
        var_cv = float(np.var(payoffs_cv, ddof=1))
        se_cv = float(np.std(payoffs_cv, ddof=1) / np.sqrt(n_paths))
        price_cv = float(np.mean(payoffs_cv))

        vr_anti = float((1.0 - (var_anti / var_plain)) * 100.0) if var_plain > 0 else 0.0
        vr_cv = float((1.0 - (var_cv / var_plain)) * 100.0) if var_plain > 0 else 0.0

        return pd.DataFrame([
            {"Method": "Plain Monte Carlo", "Price": price_plain, "Std Error": se_plain, "Sample Variance": var_plain, "Var Reduction %": 0.0},
            {"Method": "Antithetic Variates", "Price": price_anti, "Std Error": se_anti, "Sample Variance": var_anti, "Var Reduction %": vr_anti},
            {"Method": "Control Variates (ST)", "Price": price_cv, "Std Error": se_cv, "Sample Variance": var_cv, "Var Reduction %": vr_cv}
        ])

    @staticmethod
    def compute_deep_diagnostics(returns: pd.Series):
        """Computes statistical moments and autocorrelation structures on empirical return series."""
        clean_ret = returns.dropna()
        if len(clean_ret) < 4:
            return {"skewness": 0.0, "kurtosis": 0.0, "jarque_bera_stat": 0.0, "jarque_bera_p_value": 0.0, "return_acf_lag1": 0.0, "vol_clustering_acf_lag1": 0.0}

        jb_stat, jb_pval = jarque_bera(clean_ret)
        acf_1 = float(clean_ret.autocorr(lag=1))
        sq_acf_1 = float((clean_ret**2).autocorr(lag=1))

        return {
            "skewness": float(skew(clean_ret)),
            "kurtosis": float(kurtosis(clean_ret)),
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_p_value": float(jb_pval),
            "return_acf_lag1": 0.0 if np.isnan(acf_1) else acf_1,
            "vol_clustering_acf_lag1": 0.0 if np.isnan(sq_acf_1) else sq_acf_1
        }

    @staticmethod
    def export_reproducible_config(filepath: str, config_dict: dict):
        """Dumps simulation run metadata for full state reproducibility."""
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)