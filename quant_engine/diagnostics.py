import numpy as np
from scipy.stats import skew, kurtosis, jarque_bera

class DiagnosticEngine:
    @staticmethod
    def analyze_returns(returns: np.ndarray, theoretical_df: float = 5.0) -> dict:
        """
        Computes statistical diagnostics including empirical skewness, kurtosis, Jarque-Bera, and ACF.
        """
        n = len(returns)
        if n < 4:
            return {}

        mean_ret = float(np.mean(returns))
        vol_ret = float(np.std(returns, ddof=1))
        sk = float(skew(returns))
        kurt_emp = float(kurtosis(returns, fisher=True)) # Excess kurtosis
        
        theo_kurt = 6.0 / (theoretical_df - 4.0) if theoretical_df > 4.0 else np.inf

        jb_stat, jb_p = jarque_bera(returns)

        # Autocorrelation lag 1
        r_centered = returns - mean_ret
        acf1 = float(np.corrcoef(r_centered[:-1], r_centered[1:])[0, 1]) if n > 1 else 0.0
        
        sq_r = r_centered**2
        sq_acf1 = float(np.corrcoef(sq_r[:-1], sq_r[1:])[0, 1]) if n > 1 else 0.0

        return {
            "mean": mean_ret,
            "volatility": vol_ret,
            "skewness": sk,
            "empirical_kurtosis": kurt_emp,
            "theoretical_kurtosis": theo_kurt,
            "jb_statistic": float(jb_stat),
            "jb_p_value": float(jb_p),
            "acf1": acf1,
            "squared_acf1": sq_acf1,
            "sample_size": n
        }
