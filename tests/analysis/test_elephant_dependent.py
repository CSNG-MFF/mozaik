from mozaik.analysis.elephant_dependent import *
import numpy as np
import pytest
from scipy import stats
import pylab


class TestCriticalityAnalysis:
    def gen_powerlaw_distr(self, ca, a, b):
        x = np.linspace(1, 1000, 1000)
        f = ca.powerlaw(x, a, b)
        n_samples = int(f.sum())
        f /= f.sum()
        distr = stats.rv_discrete(name="powerlaw", values=(x, f), seed=10)
        samples = distr.rvs(size=n_samples)
        return samples

    @pytest.mark.parametrize(
        "exponent", [-0.2, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 0.2]
    )
    def test_fit_powerlaw(self, exponent):
        ca = CriticalityAnalysis(None, {"num_bins": 100})
        data = self.gen_powerlaw_distr(ca, 1000, exponent)
        distr, bins = ca.create_hist(data, ca.parameters["num_bins"])
        _, exponent_fit, _, _ = ca.fit_powerlaw_distribution(bins, distr)

        assert np.isclose(exponent, exponent_fit, rtol=0.1)
