from arch.compat.python import range

from collections import defaultdict
import glob

import numpy as np
import scipy.stats as stats
from statsmodels.regression.linear_model import OLS, WLS

from yapf.yapflib.yapf_api import FormatCode

PERCENTILES = list(np.arange(0.1, 1.0, 0.1)) + list(np.arange(1.0, 100.0, 0.5))
PERCENTILES = PERCENTILES[::-1]

files = glob.glob("engle-granger/*.npz")
data = defaultdict(list)
for f in files:
    contents = np.load(f)
    temp = contents["quantiles"]
    temp.shape = temp.shape + (1,)
    data[contents["trend"][0]].append(temp)
    sample_sizes = contents["sample_sizes"]

final = {}
for key in data:
    final[key] = np.concatenate(data[key], -1)  # type: np.ndarray

trends = ("nc", "c", "ct", "ctt")
critical_values = (1, 5, 10)
cv_approx = {}
final_cv = {}
for trend in trends:
    print(trend)
    results = final[trend]

    # For percentiles 1, 5 and 10, regress on a constant, and powers of 1/T
    out = defaultdict(list)
    for cv in critical_values:
        num_ex = results.shape[-1]
        loc = np.argmin(np.abs(np.array(PERCENTILES) - cv))
        all_lhs = np.squeeze(results[loc])
        tau = np.ones((num_ex, 1)).dot(sample_sizes[None, :])
        tau = tau.T
        tau = tau.ravel()
        tau = tau[:, None]
        n = all_lhs.shape[0]
        rhs = (1.0 / tau) ** np.arange(4)

        for i in range(all_lhs.shape[1]):
            lhs = all_lhs[:, i, :].ravel()
            res = OLS(lhs, rhs).fit()
            params = res.params.copy()
            if res.pvalues[-1] > 0.05:
                params[-1] = 0.00
            out[cv].append(params)
        values = np.array(out[cv]).tolist()
        out[cv] = [[round(val, 5) for val in row] for row in values]

    final_cv[trend] = dict(out)

formatted_str = (
    str(final_cv)
    .replace(" ", "")
    .replace("],", "],\n")
    .replace(":", ":\n")
    .replace("},", "},\n")
)

header = """
import numpy as np

eg_num_variables = np.arange(1, 13)
"""

footer = """
class EngleGrangerCV(object):
    def __getitem__(self, item):
        # item ['nc',10,3]
        if (len(item) != 3 or
                item[0] not in eg_cv.keys() or
                item[1] not in (10,5,1) or
                item[2] not in eg_num_variables):
            raise KeyError('Item not found: {0}'.format(item))
        return np.array(eg_cv[item[0]][item[1]][item[2] - 1])


engle_granger_cv = EngleGrangerCV()
"""

formatted_code = FormatCode(header + "eg_cv = " + formatted_str + footer)[0]

with open("../engle_granger.py", "w") as eg:
    eg.write(formatted_code)

# %%
large_p = {}
small_p = {}
tau_max = {}
tau_star = {}
tau_min = {}
for trend in trends:
    data = final[trend].mean(3)
    data_std = final[trend].std(3)
    percentiles = np.array(PERCENTILES)
    lhs = stats.norm().ppf(percentiles / 100.0)
    lhs_large = lhs
    for i in range(1, data.shape[2]):
        avg_test_stats = data[:, -1, i]
        avg_test_std = data_std[:, -1, i]
        avg_test_stats = avg_test_stats[:, None]
        m = lhs.shape[0]
        rhs = avg_test_stats ** np.arange(4)
        rhs_large = rhs
        res_large = WLS(lhs_large, rhs, weights=1.0 / avg_test_std).fit()
        large_p[(trend, i)] = res_large.params.tolist()

        # Compute tau_max, by finding the func maximum
        p = res_large.params
        poly_roots = np.roots(np.array([3, 2, 1.0]) * p[:0:-1])
        tau_max[(trend, i)] = float(np.squeeze(np.real(np.max(poly_roots))))

        # Small p regression using only p<=15%
        cutoff = np.where(percentiles <= 15.0)[0]
        avg_test_stats = data[cutoff][:, -1, i]
        avg_test_std = data_std[cutoff][:, -1, i]
        avg_test_stats = avg_test_stats[:, None]
        lhs_small = lhs[cutoff]
        m = lhs_small.shape[0]
        rhs = avg_test_stats ** np.arange(3)
        res_small = WLS(lhs_small, rhs, weights=1.0 / avg_test_std).fit()
        small_p[(trend, i)] = res_small.params.tolist()

        # Compute tau star
        err_large = lhs_large - rhs_large.dot(res_large.params)
        # Missing 1 parameter here, replace with 0
        params = np.append(res_small.params, 0.0)
        err_small = lhs_large - rhs_large.dot(params)
        # Find the location that minimizes the total absolute error
        m = lhs_large.shape[0]
        abs_err = np.zeros((m, 1))
        for j in range(m):
            abs_err[j] = np.abs(err_large[:j]).sum() + np.abs(err_small[j:]).sum()
        loc = np.argmin(abs_err)
        tau_star[(trend, i)] = rhs_large[loc, 1]
        # Compute tau min
        tau_min[(trend, i)] = -params[1] / (2 * params[2])
