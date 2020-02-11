from typing import Optional

from numpy import column_stack, zeros
from numpy.linalg import lstsq

from arch.typing import ArrayLike

from .covariance import CovarianceEstimator


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
        bandwith: Optional[float] = None,
        df_adjust: int = 0,
        center: bool = True,
        weights: Optional[ArrayLike] = None,
    ):
        super().__init__(
            x, bandwith=bandwith, df_adjust=df_adjust, center=center, weights=weights
        )
        self._kernel = kernel
        self._lags = lags
        self._diagonal_lags = diagonal_lags
        self._method = method
        self._diagonal = diagonal
        self._max_lag = max_lag

    def _select_lags(self):
        nobs, nvar = self._x.shape
        max_lag = int(nobs ** 1 / 3)
        max_lag = min(max_lag, nobs // nvar)
        if max_lag == 0:
            import warnings

            warnings.warn(
                "The maximum number of lags is 0 since the number of time series "
                f"observations {nobs} is small relative to the number of time "
                f"series {nvar}.",
                RuntimeWarning,
            )
        from statsmodels.tsa.tsatools import lagmat

        lhs_data = []
        rhs_data = [[]] * max_lag
        for i in range(nvar):
            l, r = lagmat(self._x[:, i], max_lag, trim="both", original="sep")
            lhs_data.append(l)
            for k in range(max_lag):
                rhs_data[k].append(r[:, [k]])
        lhs = column_stack(lhs_data)
        rhs = column_stack([column_stack(r) for r in rhs_data])
        from statsmodels.tools import add_constant

        if self._center:
            rhs = add_constant(rhs, True)
        c = int(self._center)
        sigma = zeros((max_lag + 1, nvar, nvar))
        lhs_obs = lhs.shape[0]

        if self._center:
            resid = lhs - lhs.mean(0)
        sigma[0] = resid.T @ resid / lhs_obs
        for i in range(max_lag + 1):
            x = rhs[:, : (i * nvar) + c]
            resid = lhs
            if i > 0 or self._center:
                params = lstsq(x, lhs, rcond=None)[0]
                resid = lhs - x @ params
            sigma[i] = resid.T @ resid / lhs_obs
        # TODO: Write up diagonal extensions

    def cov(self):
        pass
