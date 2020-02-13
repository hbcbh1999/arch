from typing import Optional, Union

import numpy as np
from numpy import arange, asarray, ceil, log, pi, power
from numpy.linalg import lstsq
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import add_trend

from arch.typing import ArrayLike1D, ArrayLike2D
from arch.unitroot.critical_values.engle_granger import EngleGrangerCV
from arch.utility.array import ensure1d, ensure2d


def _sse(y, x):
    x, y = asarray(x), asarray(y)
    b = lstsq(x, y, None)[0]
    return ((y - x.dot(b)) ** 2).sum()


def _ic(sse, k, nobs, ic):
    llf = -nobs / 2 * (log(2 * pi) + log(sse / nobs) + 1)
    if ic == "aic":
        penalty = 2
    elif ic == "hqic":
        penalty = 2 * log(log(nobs))
    else:  # bic
        penalty = log(nobs)
    return -llf + k * penalty


class FMOLS(object):
    pass


class CCR(object):
    """canonical cointegration regression estimator"""

    pass


class DynamicOLS(object):
    """dynamic OLS"""

    def __init__(
        self,
        y,
        x,
        lags=None,
        leads=None,
        common=True,
        max_lag=None,
        max_lead=None,
        ic="aic",
    ):
        self._y = y
        self._x = x
        self._lags = lags
        self._leads = leads
        self._common = common
        self._max_lag = max_lag
        self._max_lead = max_lead
        self._ic = ic
        self._res = None
        self._compute()

    def _compute(self):
        y, x = self._y, self._x
        k = x.shape[1]
        nobs = y.shape[0]
        delta_lead_lags = x.diff()
        max_lag = int(ceil(12.0 * power(nobs / 100.0, 1 / 4.0)))
        lag_len = max_lag if self._lags is None else self._lags
        lead_len = max_lag if self._leads is None else self._leads

        lags = pd.concat([delta_lead_lags.shift(i) for i in range(1, lag_len + 1)], 1)
        lags.columns = [
            "D.{col}.LAG.{lag}".format(col=col, lag=i)
            for i in range(1, lag_len + 1)
            for col in x
        ]
        contemp = delta_lead_lags
        contemp.columns = ["D.{col}.LAG.0".format(col=col) for col in x]
        leads = pd.concat(
            [delta_lead_lags.shift(-i) for i in range(1, lead_len + 1)], 1
        )
        leads.columns = [
            "D.{col}.LEAD.{lead}".format(col=col, lead=i)
            for i in range(1, lead_len + 1)
            for col in x
        ]
        full = pd.concat([y, x, lags.iloc[:, ::-1], contemp, leads], 1).dropna()
        lhs = full.iloc[:, [0]]
        rhs = add_constant(full.iloc[:, 1:])
        base_iloc = arange(k + 1).tolist()
        sses = {}

        if self._leads is None:
            q_range = range(max_lag)
        else:
            q_range = range(self._leads, self._leads + 1)
        if self._lags is None:
            p_range = range(max_lag)
        else:
            p_range = range(self._lags, self._lags + 1)

        for p in p_range:
            for q in q_range:
                lead_lag_iloc = arange(
                    1 + k * (1 + lag_len - p), 1 + k * (1 + lag_len + 1 + q)
                ).tolist()
                _rhs = rhs.iloc[:, base_iloc + lead_lag_iloc]
                sses[(p, q, _rhs.shape[1])] = _sse(lhs, _rhs)
        sses = pd.Series(sses)
        param_counts = sses.index.get_level_values(2)
        ics = {
            idx: _ic(sses[idx], k, nobs, self._ic)
            for k, idx in zip(param_counts, sses.index)
        }
        ics = pd.Series(ics)
        sel_idx = ics.idxmin()
        p, q = sel_idx[:2]
        lead_lag_iloc = arange(
            1 + k * (1 + lag_len - p), 1 + k * (1 + lag_len + 1 + q)
        ).tolist()
        _rhs = rhs.iloc[:, base_iloc + lead_lag_iloc]
        mod = OLS(lhs, _rhs)
        res = mod.fit()
        self._res = res
        print(res.summary())

    @property
    def result(self):
        return self._res


def _cross_section(y, x, trend):
    if trend not in ("n", "c", "ct", "t"):
        raise ValueError('trend must be one of "n", "c", "ct" or "t"')
    x = add_trend(x, trend)
    res = OLS(y, x).fit()
    return res.resid


def engle_granger(
    y: ArrayLike1D,
    x: ArrayLike2D,
    trend: str = "c",
    *,
    lags: Optional[int] = None,
    max_lags: Optional[int] = None,
    method: str = "aic",
    df_adjust: Union[bool, int] = True,
):

    y = np.asarray(ensure1d(y, "x", False))
    x = np.asarray(ensure2d(x, "x"))
    resid = _cross_section(y, x, trend)
    from arch.unitroot.unitroot import ADF

    adf = ADF(resid, lags, trend="n", max_lags=max_lags, method=method)
    # TODO: pvalue, crit val method need to be better
    eg_cv = EngleGrangerCV()
    cv = pd.Series({p: eg_cv[trend, p, x.shape[1] + 1] for p in (10, 5, 1)})
    return CointegrationTestResult(adf.stat, adf.pvalue, cv, "Engle-Granger Test")


def phillips_ouliaris(y, x, trend="c", lags=None, df_adjust=True):
    _cross_section(y, x, trend)
    pass


class CointegrationTestResult(object):
    def __init__(
        self, stat: float, pvalue: float, crit_vals: pd.Series, name: str
    ) -> None:
        self._stat = stat
        self._pvalue = pvalue
        self._crit_vals = crit_vals
        self._name = name
        self._null = "No Cointegration"
        self._alternative = "Cointegration"

    @property
    def stat(self) -> float:
        return self._stat

    @property
    def pvalue(self) -> float:
        return self._pvalue

    @property
    def crit_vals(self) -> pd.Series:
        return self._crit_vals

    @property
    def null(self) -> str:
        return self._null

    @property
    def alternative(self) -> str:
        return self._alternative

    def __str__(self) -> str:
        out = f"{self._name}\nStatistic:{self._stat}\nP-value:{self.pvalue}"
        out += f"\nNull: {self._null}, Alternative: {self._alternative}"
        cv_str = ", ".join([f"{k}%: {v}" for k, v in self.crit_vals.items()])
        out += f"\nCrit. Vals: {cv_str}"
        return out

    def __repr__(self):
        return self.__str__() + f"\nID: {hex(id(self))}"


g = np.random.default_rng(0)
y = g.standard_normal((500, 1))
y = np.cumsum(y, 0)
x = y + g.standard_normal((500, 1))
print(engle_granger(y, x))
