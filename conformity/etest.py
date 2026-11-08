# Vendored from jaebooker/martingale-swarm (pre-existing work), src/martingale_audit/etest.py, unmodified below this header.
"""Testing by betting.

Null hypothesis for a sequence x_1, x_2, ... in [-1, 1]:

    H0:  E[x_t | everything before t] <= 0   for every t.

Start with wealth 1 and at each step stake a fraction lam_t in [0, max_bet] on
x_t being positive, where lam_t depends only on the past:

    W_t = W_{t-1} * (1 + lam_t * x_t).

Under H0, W is a non-negative supermartingale, so by Ville's inequality

    P( W ever reaches 1/alpha ) <= alpha.

That bound holds at every step simultaneously. You can look after each message,
stop whenever you like, and the error rate is still alpha. No fixed sample size
and no correction for peeking.

The staking rule is aGRAPA (Waudby-Smith & Ramdas, "Estimating means of bounded
random variables by betting", 2023), truncated to one-sided bets. The rule only
affects power. Validity requires the conditional null above as well as a
predictable, bounded stake. This module does not establish that null for
extracted transcript scores or noisy measurements of an underlying belief.
"""
import math
import sys
from typing import Dict, Iterable, List


class EProcess:
    def __init__(self, max_bet: float = 0.5, prior_var: float = 0.25):
        if not 0.0 < max_bet < 1.0:
            raise ValueError("max_bet must be in (0, 1) so wealth stays positive")
        if not math.isfinite(prior_var) or prior_var <= 0:
            raise ValueError("prior_var must be positive and finite")
        self.max_bet = max_bet
        self.prior_var = prior_var
        self.n = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self.log_wealth = 0.0
        self.max_log_wealth = 0.0
        self.path: List[float] = []

    def next_bet(self) -> float:
        """Stake for the next observation. Uses past observations only."""
        if self.n == 0:
            return 0.0
        mean = self._sum / (self.n + 1)                       # shrunk toward 0
        var = (self.prior_var + self._sumsq) / (self.n + 1) - mean * mean
        if mean <= 0.0:
            return 0.0
        return min(self.max_bet, mean / (max(var, 1e-12) + mean * mean))

    def update(self, x: float) -> float:
        if not -1.0 <= x <= 1.0:
            raise ValueError("x must lie in [-1, 1]")
        lam = self.next_bet()
        self.log_wealth += math.log1p(lam * x)
        self.max_log_wealth = max(self.max_log_wealth, self.log_wealth)
        self.n += 1
        self._sum += x
        self._sumsq += x * x
        self.path.append(self.wealth)
        return self.wealth

    def update_many(self, xs: Iterable[float]) -> "EProcess":
        for x in xs:
            self.update(x)
        return self

    @property
    def wealth(self) -> float:
        """Conditional-null e-value; log wealth remains usable after overflow."""
        return _exp_wealth(self.log_wealth)

    @property
    def max_wealth(self) -> float:
        """Running maximum for the p-value; this maximum is NOT an e-value."""
        return _exp_wealth(self.max_log_wealth)

    @property
    def p_value(self) -> float:
        """Anytime-valid under the conditional null: inverse running maximum."""
        return min(1.0, math.exp(-self.max_log_wealth))

    def rejected(self, alpha: float = 0.05) -> bool:
        _check_alpha(alpha)
        return self.max_log_wealth >= math.log(1.0 / alpha)


def _exp_wealth(log_wealth: float) -> float:
    return math.exp(log_wealth) if log_wealth <= math.log(sys.float_info.max) else math.inf


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _check_e_values(values: Iterable[float]) -> List[float]:
    vals = list(values)
    if any(math.isnan(v) or v < 0 for v in vals):
        raise ValueError("e-values must be nonnegative and not NaN")
    return vals


def e_bh(e_values: Dict[str, float], alpha: float = 0.05) -> List[str]:
    """e-BH (Wang & Ramdas 2022). Controls the false discovery rate at alpha
    for one application to valid e-values from a prespecified family, under
    arbitrary cross-hypothesis dependence. Repeatedly unioning discoveries is
    not covered. Running maximum wealth must not be supplied as an e-value."""
    _check_alpha(alpha)
    _check_e_values(e_values.values())
    items = sorted(e_values.items(), key=lambda kv: -kv[1])
    n = len(items)
    k_star = 0
    for k, (_, e) in enumerate(items, start=1):
        if e >= n / (alpha * k):
            k_star = k
    return [name for name, _ in items[:k_star]]


def merge(e_values: Iterable[float]) -> float:
    """Mean of valid terminal e-values for a prespecified family.

    Valid under arbitrary dependence. This cannot repair invalid component
    e-values, adaptive membership, or the use of running maxima as inputs.
    Empty input returns the neutral value 1.
    """
    vals = _check_e_values(e_values)
    return sum(v / len(vals) for v in vals) if vals else 1.0
