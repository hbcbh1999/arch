from typing import Optional, Sequence

import numpy as np
from numpy import column_stack, ones, zeros
from numpy.linalg import lstsq
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat

from arch.covariance.kernel import CovarianceEstimator
from arch.typing import ArrayLike, NDArray


class PreWhitenRecoloredCovariance(CovarianceEstimator):
    def __init__(
        self,
        x: ArrayLike,
        kernel: str = "bartlett",
        lags: Optional[int] = None,
        diagonal_lags: Optional[int] = None,
        method: str = "aic",
        max_lag: Optional[int] = None,
        diagonal: bool = True,
        bandwidth: Optional[float] = None,
        df_adjust: int = 0,
        center: bool = True,
        weights: Optional[ArrayLike] = None,
    ):
        super().__init__(
            x, bandwidth=bandwidth, df_adjust=df_adjust, center=center, weights=weights
        )
        self._kernel = kernel
        self._lags = lags
        self._diagonal_lags = diagonal_lags
        self._method = method
        self._diagonal = diagonal
        self._max_lag = max_lag

    def _ic_single(self, idx, resid, regressors, lags, lag, nparam):
        add_lags = lags[:, lag:]
        params = np.linalg.lstsq(regressors, add_lags, rcond=None)[0]
        add_lags_resid = add_lags - regressors @ params
        curr_resid = resid[:, [idx]].copy()
        nobs = resid.shape[0]
        ic = np.full(add_lags.shape[1] + 1, np.inf)
        best_resids = resid
        for i in range(add_lags.shape[1] + 1):
            if i > 0:
                params = np.linalg.lstsq(add_lags_resid[:, :i], curr_resid, rcond=None)
                new_resid = curr_resid - add_lags_resid[:, :i] @ params[0]
                resid[:, [idx]] = new_resid
            sigma = resid.T @ resid / nobs
            _, ld = np.linalg.slogdet(sigma)
            if self._method == "aic":
                ic[i] = ld + 2 * (nparam + i) / nobs
            elif self._method == "hqc":
                ic[i] = ld + np.log(np.log(nobs)) * (nparam + i) / nobs
            else:  # bic
                ic[i] = ld + np.log(nobs) * (nparam + i) / nobs
            if ic[i] == ic.min():
                best_resids = resid.copy()
        return np.argmin(ic), best_resids

    def _select_lags(self):
        nobs, nvar = self._x.shape
        max_lag = int(nobs ** (1 / 3))
        # Ensure at least nvar obs left over
        max_lag = min(max_lag, (nobs - nvar) // nvar)
        if max_lag == 0:
            import warnings

            warnings.warn(
                "The maximum number of lags is 0 since the number of time series "
                f"observations {nobs} is small relative to the number of time "
                f"series {nvar}.",
                RuntimeWarning,
            )

        lhs_data = []
        rhs_data = [[] for _ in range(max_lag)]
        indiv_lags = []
        for i in range(nvar):
            r, l = lagmat(self._x[:, i], max_lag, trim="both", original="sep")
            lhs_data.append(l)
            indiv_lags.append(r)
            for k in range(max_lag):
                rhs_data[k].append(r[:, [k]])

        lhs = column_stack(lhs_data)
        rhs = column_stack([column_stack(r) for r in rhs_data])

        if self._center:
            rhs = add_constant(rhs, True)
        c = int(self._center)
        lhs_obs = lhs.shape[0]
        ic = zeros(max_lag + 1)
        ics = {}
        for i in range(max_lag + 1):
            indiv_lag_len = []
            x = rhs[:, : (i * nvar) + c]
            nparam = 0
            resid = lhs
            if i > 0 or self._center:
                params = lstsq(x, lhs, rcond=None)[0]
                resid = lhs - x @ params
                nparam = params.size
            for idx in range(nvar):
                lag_len, resid = self._ic_single(
                    idx, resid, x, indiv_lags[idx], i, nparam
                )
                indiv_lag_len.append(lag_len + i)

            sigma = resid.T @ resid / lhs_obs
            _, ld = np.linalg.slogdet(sigma)
            if self._method == "aic":
                ic = ld + 2 * nparam / lhs_obs
            elif self._method == "hqc":
                ic = ld + np.log(np.log(lhs_obs)) * nparam / lhs_obs
            else:  # bic
                ic = ld + np.log(lhs_obs) * nparam / lhs_obs
            ics[(i, tuple(indiv_lag_len))] = ic
        ic = np.array([crit for crit in ics.values()])
        models = [key for key in ics.keys()]
        return models[ic.argmin()]

    def _estimate_var(self, common: int, individual: Sequence[int]):
        max_lag = max(common, max(individual))

    def cov(self):
        pass

    def bandwidth_scale(self) -> float:
        return 1.0

    def kernel_const(self) -> float:
        return 1.0

    def _weights(self, bw) -> NDArray:
        return ones(0)

    def rate(self) -> float:
        return 2 / 9
