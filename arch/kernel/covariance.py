from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import numpy as np
from pandas import DataFrame

from arch.typing import NDArray, NDArrayOrFrame
from arch.utility.array import AbstractDocStringInheritor, DocStringInheritor, ensure2d
from arch.vendor import cached_property


class CovarianceEstimate(object):
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
        return self._wrap(self._sr + self._oss + self._oss.T)

    @cached_property
    def short_run(self) -> NDArrayOrFrame:
        return self._wrap(self._sr)

    @cached_property
    def one_sided(self) -> NDArrayOrFrame:
        return self._wrap(self._sr + self._oss)

    @cached_property
    def one_sided_strict(self) -> NDArrayOrFrame:
        return self._wrap(self._oss)


class CovarianceEstimator(ABC):
    _name = ""

    def __init__(
        self,
        x,
        bandwith: Optional[float] = None,
        df_adjust: int = 0,
        center: bool = True,
    ):

        self._x = ensure2d(x, "x")
        if bandwith is not None and bandwith < 0:
            raise ValueError("bandwith must be a non-negative integer.")
        self._bandwidth = bandwith
        if df_adjust < 0:
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
        return self._name

    @property
    def centered(self) -> bool:
        return self._center

    @abstractmethod
    @property
    def bandwidth_const(self) -> float:
        return 1.0

    @abstractmethod
    @property
    def bandwidth_scale(self) -> float:
        return 1.0

    @abstractmethod
    @property
    def rate(self) -> float:
        return 0.0

    def opt_bandiwdth(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def _weights(self, bw) -> NDArray:
        return np.ones(1)

    @abstractmethod
    @property
    def weights(self) -> NDArray:
        if self._bandwidth is not None:
            bw = self._bandwidth
        else:
            bw = self.opt_bandiwdth()
        return self._weights(bw)

    def cov(self) -> CovarianceEstimate:
        x = np.asarray(self._x)
        if self._center:
            x -= x.mean(0)
        k = x.shape[1]
        df = self._df
        sr = x.T @ x / df
        w = self.weights
        num_weights = w.shape[0]
        oss = np.zeros((k, k))
        for i in range(num_weights):
            oss += w[i] * x[i + 1 :].T @ x[: -(i + 1)]

        labels = self._x.columns if isinstance(self._x, DataFrame) else None
        return CovarianceEstimate(sr, oss, labels)


class Bartlett(CovarianceEstimator, metaclass=AbstractDocStringInheritor):
    @property
    def bandwidth_const(self) -> float:
        return 1.1447

    @property
    def bandwidth_scale(self) -> float:
        return 1.0

    @property
    def rate(self) -> float:
        return 2 / 9
