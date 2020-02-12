import numpy as np
import pytest

from arch.covariance.kernel import (
    Bartlett,
    CovarianceEstimate,
    Parzen,
    ParzenCauchy,
    ParzenGeometric,
    ParzenRiesz,
    TukeyHamming,
    TukeyHanning,
    TukeyParzen,
)


@pytest.fixture(scope="module", params=[1, 2])
def data(request):
    ndim = request.param
    if ndim == 1:
        return np.random.standard_normal((200))
    return np.random.standard_normal((200, ndim))


@pytest.mark.parametrize(
    "est",
    [
        ParzenRiesz,
        ParzenGeometric,
        ParzenCauchy,
        Parzen,
        Bartlett,
        TukeyParzen,
        TukeyHanning,
        TukeyHamming,
    ],
)
def test_covariance_smoke(data, est):
    cov = est(data).cov
    ndim = data.ndim
    assert isinstance(cov, CovarianceEstimate)
    assert cov.long_run.shape == (ndim, ndim)
    assert isinstance(str(est), str)
    assert isinstance(repr(est), str)
