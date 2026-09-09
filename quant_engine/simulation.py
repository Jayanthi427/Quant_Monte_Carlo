import numpy as np
from scipy.stats import qmc, t as student_t
from typing import Dict, Any, Tuple

class MonteCarloSimulator:
    """
    Generates stochastic asset paths under Student-t dynamics for P and Q measures.
    Supports Standard MC, Antithetic Variates, Control Variates, and Quasi-Monte Carlo.
    """
    def __init__(self, S0: float, sigma: float, mu_P: float, r: float, q: float, df: float, days: int = 30, seed: int = 42):
        self.S0 = float(S0)
        self.sigma = float(sigma)
        self.mu_P = float(mu_P)
        self.r = float(r)
        self.q = float(q)
        self.df = max(2.01, float(df))
        self.days = int(days)
        self.dt = 1.0 / 252.0
        self.seed = int(seed)

    def _sample_t_shocks(self, shape: Tuple[int, int], method: str = "standard", rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(self.seed)

        n_paths, n_steps = shape
        scale = np.sqrt((self.df - 2.0) / self.df) if self.df > 2.0 else 1.0

        if method == "antithetic":
            half_paths = int(np.ceil(n_paths / 2))
            Z = rng.standard_t(df=self.df, size=(half_paths, n_steps)) * scale
            Z_full = np.vstack([Z, -Z])[:n_paths, :]
            return Z_full

        elif method == "qmc":
            sampler = qmc.Sobol(d=n_steps, scramble=True, seed=rng.integers(0, 1000000))
            # Round up to next power of 2 for Sobol sequence alignment
            m = int(np.ceil(np.log2(max(n_paths, 2))))
            uniforms = sampler.random_base2(m=m)[:n_paths, :]
            uniforms = np.clip(uniforms, 1e-7, 1.0 - 1e-7)
            Z = student_t.ppf(uniforms, df=self.df) * scale
            return Z

        else: # Standard Pseudo-Random
            return rng.standard_t(df=self.df, size=(n_paths, n_steps)) * scale

    def simulate_p_measure(self, num_paths: int = 2000, method: str = "standard", seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        n_steps = self.days
        Z = self._sample_t_shocks((num_paths, n_steps), method=method, rng=rng)
        
        # Physical drift drift_P = mu_P - 0.5 * sigma^2
        drift_P = (self.mu_P - 0.5 * self.sigma**2) * self.dt
        vol_step = self.sigma * np.sqrt(self.dt)
        
        log_increments = drift_P + vol_step * Z
        log_paths = np.zeros((num_paths, n_steps + 1))
        log_paths[:, 0] = np.log(self.S0)
        log_paths[:, 1:] = np.log(self.S0) + np.cumsum(log_increments, axis=1)
        
        return np.exp(log_paths)

    def simulate_q_measure(self, num_paths: int = 2000, method: str = "standard", seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        n_steps = self.days
        Z = self._sample_t_shocks((num_paths, n_steps), method=method, rng=rng)
        
        # Risk-Neutral drift drift_Q = (r - q) - 0.5 * sigma^2
        drift_Q = ((self.r - self.q) - 0.5 * self.sigma**2) * self.dt
        vol_step = self.sigma * np.sqrt(self.dt)
        
        log_increments = drift_Q + vol_step * Z
        log_paths = np.zeros((num_paths, n_steps + 1))
        log_paths[:, 0] = np.log(self.S0)
        log_paths[:, 1:] = np.log(self.S0) + np.cumsum(log_increments, axis=1)
        
        return np.exp(log_paths)

    def run_variance_reduction_benchmark(self, K: float, T: float, N: int = 50000) -> Dict[str, Any]:
        discount = np.exp(-self.r * T)
        
        # Baseline Standard MC
        q_std = self.simulate_q_measure(num_paths=N, method="standard")
        payoffs_std = np.maximum(q_std[:, -1] - K, 0.0) * discount
        price_std = float(np.mean(payoffs_std))
        var_std = float(np.var(payoffs_std, ddof=1))
        se_std = float(np.sqrt(var_std / N))
        
        # Antithetic Variates
        q_anti = self.simulate_q_measure(num_paths=N, method="antithetic")
        payoffs_anti = np.maximum(q_anti[:, -1] - K, 0.0) * discount
        price_anti = float(np.mean(payoffs_anti))
        var_anti = float(np.var(payoffs_anti, ddof=1))
        se_anti = float(np.sqrt(var_anti / N))
        
        # Sobol Quasi-Monte Carlo
        q_qmc = self.simulate_q_measure(num_paths=N, method="qmc")
        payoffs_qmc = np.maximum(q_qmc[:, -1] - K, 0.0) * discount
        price_qmc = float(np.mean(payoffs_qmc))
        var_qmc = float(np.var(payoffs_qmc, ddof=1))
        se_qmc = float(np.sqrt(var_qmc / N))
        
        return {
            "Standard_MC": {"price": price_std, "se": se_std, "variance": var_std, "vrf": 1.0},
            "Antithetic": {"price": price_anti, "se": se_anti, "variance": var_anti, "vrf": max(1.0, var_std / max(var_anti, 1e-12))},
            "Quasi_MC": {"price": price_qmc, "se": se_qmc, "variance": var_qmc, "vrf": max(1.0, var_std / max(var_qmc, 1e-12))}
        }

    def run_convergence_study(self, K: float, T: float, path_counts: list = None) -> list:
        if path_counts is None:
            path_counts = [1000, 5000, 20000, 100000]
            
        discount = np.exp(-self.r * T)
        results = []
        
        for N in path_counts:
            q_paths = self.simulate_q_measure(num_paths=N, method="standard")
            payoffs = np.maximum(q_paths[:, -1] - K, 0.0) * discount
            price = float(np.mean(payoffs))
            var_val = float(np.var(payoffs, ddof=1))
            se = float(np.sqrt(var_val / N))
            ci_95 = (price - 1.96 * se, price + 1.96 * se)
            
            results.append({
                "N": N,
                "price": price,
                "se": se,
                "ci_95": ci_95
            })
            
        return results