import numpy as np
from scipy.stats import chi2

class RiskEngine:
    @staticmethod
    def calculate_var_and_es(returns: np.ndarray, alpha: float = 0.95) -> tuple:
        """
        Calculates Value-at-Risk (VaR) and Expected Shortfall (ES/CVaR) on returns.
        """
        sorted_returns = np.sort(returns)
        index = int((1.0 - alpha) * len(sorted_returns))
        var = -sorted_returns[index]
        tail_losses = sorted_returns[:index]
        es = -np.mean(tail_losses) if len(tail_losses) > 0 else var
        return float(var), float(es)

    @staticmethod
    def kupiec_pof_test(failures: int, N: int, p_expected: float = 0.05, alpha: float = 0.05) -> dict:
        """
        Kupiec Proportion of Failures (POF) Likelihood Ratio Test.
        """
        if N <= 0 or failures < 0 or failures > N:
            return {"statistic": 0.0, "p_value": 1.0, "decision": "INCONCLUSIVE"}
            
        p_hat = failures / N
        if p_hat == 0:
            lr = -2 * N * np.log(1 - p_expected)
        elif p_hat == 1:
            lr = -2 * N * np.log(p_expected)
        else:
            term1 = (N - failures) * np.log((1 - p_expected) / (1 - p_hat))
            term2 = failures * np.log(p_expected / p_hat)
            lr = -2 * (term1 + term2)

        lr = max(0.0, float(lr))
        p_val = float(1.0 - chi2.cdf(lr, df=1))
        decision = "Fail to reject H0" if p_val >= alpha else "Reject H0"

        return {
            "statistic": lr,
            "p_value": p_val,
            "decision": decision,
            "expected_failures": N * p_expected,
            "observed_failures": failures,
            "observed_rate": p_hat
        }

    @staticmethod
    def christoffersen_test(breaches: np.ndarray, p_expected: float = 0.05, alpha: float = 0.05) -> dict:
        """
        Christoffersen Independence Test for VaR breach clustering.
        """
        if len(breaches) < 2:
            return {"statistic": 0.0, "p_value": 1.0, "decision": "INCONCLUSIVE"}

        n00 = n01 = n10 = n11 = 0
        for i in range(len(breaches) - 1):
            b1, b2 = breaches[i], breaches[i+1]
            if b1 == 0 and b2 == 0: n00 += 1
            elif b1 == 0 and b2 == 1: n01 += 1
            elif b1 == 1 and b2 == 0: n10 += 1
            elif b1 == 1 and b2 == 1: n11 += 1

        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

        num = ((1 - pi)**(n00 + n10)) * (pi**(n01 + n11))
        den = ((1 - pi0)**n00) * (pi0**n01) * ((1 - pi1)**n10) * (pi1**n11)

        lr_ind = -2 * np.log(num / den) if den > 0 and num > 0 else 0.0
        lr_ind = max(0.0, float(lr_ind))
        p_val = float(1.0 - chi2.cdf(lr_ind, df=1))
        decision = "Fail to reject H0" if p_val >= alpha else "Reject H0"

        return {"statistic": lr_ind, "p_value": p_val, "decision": decision}
