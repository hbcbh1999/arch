from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pandas import DataFrame

from arch.typing import ArrayLike, NDArray, NDArrayOrFrame
from arch.utility.array import AbstractDocStringInheritor, ensure2d
from arch.vendor import cached_property

__all__ = [
    "Bartlett",
    "Parzen",
    "ParzenCauchy",
    "ParzenGeometric",
    "ParzenRiesz",
    "TukeyHamming",
    "TukeyHanning",
    "TukeyParzen",
    "CovarianceEstimate",
]


class CovarianceEstimate(object):
    r"""
    Covariance estimate using a long-run covariance estimator

    Parameters
    ----------
    short_run : ndarray
        The short-run covariance estimate.
    one_sided_strict : ndarray
        THe one-sided strict covariance estimate.
    columns : {None, list[str]}
        Column labels to use if covariance estiamtes are returned as
        DataFrames.

    Notes
    -----
    If :math:`\Gamma_0` is the short-run covariance and :math:`\Lambda_1` is
    the one-sided strict covariance, then the long-run covariance is defined

    .. math::

        \Gamma_0 + \Lambda_1 + \Lambda_1^\prime

    and the one-sided covariance is

    .. math::

        \Lambda_0 = \Gamma_0 + \Lambda_1.
    """

    def __init__(self, short_run: NDArray, one_sided_strict: NDArray, columns=None):
        self._sr = short_run
        self._oss = one_sided_strict
        self._columns = columns

    def _wrap(self, value):
        if self._columns is not None:
            return DataFrame(value, columns=self._columns, index=self._columns)
        return value

    @cached_property
    def long_run(self) -> NDArrayOrFrame:
        """
        The long-run covariance estimate.
        """
        return self._wrap(self._sr + self._oss + self._oss.T)

    @cached_property
    def short_run(self) -> NDArrayOrFrame:
        """
        The short-run covariance estimate.
        """
        return self._wrap(self._sr)

    @cached_property
    def one_sided(self) -> NDArrayOrFrame:
        """
        The one-sided covariance estimate.
        """
        return self._wrap(self._sr + self._oss)

    @cached_property
    def one_sided_strict(self) -> NDArrayOrFrame:
        """
        The one-sided strict covariance estimate.
        """
        return self._wrap(self._oss)


class CovarianceEstimator(ABC):
    _name = ""

    def __init__(
        self,
        x: ArrayLike,
        bandwith: Optional[float] = None,
        df_adjust: int = 0,
        center: bool = True,
        weights: Optional[ArrayLike] = None,
    ):

        self._x = ensure2d(x, "x")
        if bandwith is not None and (not np.isscalar(bandwith) or bandwith < 0):
            raise ValueError("bandwith must be a non-negative scalar.")
        self._bandwidth = bandwith
        if df_adjust < 0 or not np.isscalar(df_adjust):
            raise ValueError("df_adjust must be a non-negative integer.")
        self._df_adjust = df_adjust
        self._df = self._x.shape[0] - self._df_adjust
        if self._df <= 0:
            raise ValueError(
                "Degrees of freedom is <= 0 after adjusting the sample "
                "size of x using df_adjust. df_adjust must be less than"
                f" {self._x.shape[0]}"
            )
        self._center = center
        if weights is None:
            xw = self._x_weights = np.ones((self._x.shape[1], 1))
        else:
            xw = self._x_weights = ensure2d(np.asarray(weights), "weights").T
        if (
            xw.shape[0] != self._x.shape[1]
            or xw.shape[1] != 1
            or np.any(xw < 0)
            or np.all(xw == 0)
        ):
            raise ValueError(
                f"weights must be a 1 by {self._x.shape[1]} (x.shape[1]) "
                f"array with non-negative values where at least one value is "
                "strictly greater than 0."
            )

    def __str__(self):
        bandwidth = "auto" if self._bandwidth is None else str(self._bandwidth)
        df_adjust = "auto" if self._df_adjust is None else str(self._df_adjust)
        out = (
            f"Kernel: {self._name}",
            f"Bandwidth: {bandwidth}",
            f"Degree of Freedom Adjustment: {df_adjust}",
            f"Centered: {self.centered}",
        )
        return "\n".join(out)

    @property
    def name(self) -> str:
        """The covarianc estimator's name."""
        return self._name

    @property
    def centered(self) -> bool:
        """Flag indicating whether the data are centered (demeaned)."""
        return self._center


    @property
    @abstractmethod
    def kernel_const(self) -> float:
        """
        The constant used in optimal bandwidth calculation.
        """
        return 1.0

    @property
    @abstractmethod
    def bandwidth_scale(self) -> float:
        """
        The power used in optimal bandwidth calculation.
        """
        return 1.0

    @property
    @abstractmethod
    def rate(self) -> float:
        """
        The optimal rate used in bandwidth selection.

        Controls the number of lags used in the variance estimate that
        determines the estimate of the optimal bandwidth.
        """
        return 2 / 9

    def _alpha_q(self) -> float:
        q = self.bandwidth_scale
        v = self._x @ self._x_weights
        nobs = v.shape[0]
        n = int(4 * ((nobs / 100) ** self.rate))
        f_0s = 0.0
        f_qs = 0.0
        for j in range(n + 1):
            sig_j = np.squeeze(v[j:].T @ v[: (nobs - j)]) / nobs
            scale = 1 + j != 0
            f_0s += scale * sig_j
            f_qs += (scale ** q) * sig_j
        return (f_qs / f_0s) ** 2

    @cached_property
    def opt_bandwidth(self) -> float:
        r"""
        Estimate optimal bandwidth.

        Returns
        -------
        float
            The estimated optimal bandwidth.

        Notes
        -----
        Computed as

        .. math::

           \hat{b}_T = c_k \left[\hat{\alpha}\left(q\right) T \right]^{\frac{1}{2q+1}}

        where :math:`c_k` is a kernel-dependent constant, T is the sample size,
        q determines the optimal bandwidth rate for the kernel.
        """
        c = self.kernel_const
        q = self.bandwidth_scale
        nobs = self._x.shape[0]
        alpha_q = self._alpha_q()
        return c * (alpha_q * nobs) ** (1 / (2 * q + 1))

    @abstractmethod
    def _weights(self, bw) -> NDArray:
        return np.ones(0)

    @cached_property
    def kernel_weights(self) -> NDArray:
        """
        Weights used in covariance calculation.

        Returns
        -------
        ndarray
            The weight vector including 1 in position 0.
        """
        if self._bandwidth is not None:
            bw = self._bandwidth
        else:
            bw = self.opt_bandwidth
        return self._weights(bw)

    @cached_property
    def cov(self) -> CovarianceEstimate:
        """
        The estimated covariances.

        Returns
        -------
        CovarianceEstimate
            Covariance estiamte instance containing 4 estimates:

            * long_run
            * short_run
            * one_sided
            * one_sided_strict

        See Also
        --------
        CovarianceEstimate
        """
        x = np.asarray(self._x)
        if self._center:
            x -= x.mean(0)
        k = x.shape[1]
        df = self._df
        sr = x.T @ x / df
        w = self.kernel_weights
        num_weights = w.shape[0]
        oss = np.zeros((k, k))
        for i in range(num_weights):
            oss += w[i] * x[i + 1 :].T @ x[: -(i + 1)]

        labels = self._x.columns if isinstance(self._x, DataFrame) else None
        return CovarianceEstimate(sr, oss, labels)


class Bartlett(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.1447

    @property
    def bandwidth_scale(self) -> float:
        return 1.0

    @property
    def rate(self) -> float:
        return 2 / 9

    def _weights(self, bw: float) -> NDArray:
        int_bw = int(bw)
        return (int_bw - np.arange(float(int_bw))) / (int_bw + 1)


class Parzen(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 2.6614

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self, bw: float) -> NDArray:
        int_bw = int(bw)
        x = np.arange(float(int_bw)) / (float(int_bw) + 1)
        w = np.empty_like(x)
        loc = x <= 0.5
        w[loc] = 1 - 6 * x[loc] ** 2 * (1 - x[loc])
        w[~loc] = 2 * (1 - x[~loc]) ** 3
        return w


class ParzenRiesz(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.1340

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self, bw: float) -> NDArray:
        int_bw = int(bw)
        x = np.arange(float(int_bw)) / (float(int_bw) + 1)
        return 1 - x ** 2


class ParzenGeometric(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.0000

    @property
    def bandwidth_scale(self) -> float:
        return 1

    @property
    def rate(self) -> float:
        return 2 / 9

    def _weights(self, bw: float) -> NDArray:
        int_bw = int(bw)
        x = np.arange(float(int_bw)) / (float(int_bw) + 1)
        return 1 / (1 + x)


class ParzenCauchy(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.0924

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self, bw: float) -> NDArray:
        int_bw = int(bw)
        x = np.arange(float(int_bw)) / (float(int_bw) + 1)
        return 1 / (1 + x ** 2)


class TukeyHamming(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.6694

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self, bw: float) -> NDArray:
        int_bw = int(bw)
        x = np.arange(float(int_bw)) / (float(int_bw) + 1)
        return 0.54 + 0.46 * np.cos(np.pi * x)


class TukeyHanning(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.7462

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self, bw: float) -> NDArray:
        int_bw = int(bw)
        x = np.arange(float(int_bw)) / (float(int_bw) + 1)
        return 0.5 + 0.5 * np.cos(np.pi * x)


class TukeyParzen(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def kernel_const(self) -> float:
        return 1.8576

    @property
    def bandwidth_scale(self) -> float:
        return 2

    @property
    def rate(self) -> float:
        return 4 / 25

    def _weights(self, bw: float) -> NDArray:
        int_bw = int(bw)
        x = np.arange(float(int_bw)) / (float(int_bw) + 1)
        return 0.436 + 0.564 * np.cos(np.pi * x)
