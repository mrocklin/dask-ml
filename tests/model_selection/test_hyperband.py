import pytest
import numpy as np
import scipy.stats as stats
import math

from distributed import Client
from distributed.utils_test import loop, cluster

from dask_ml.datasets import make_classification
from dask_ml.model_selection import Hyperband


class ConstantFunction:
    def _fn(self):
        return self.value

    def get_params(self, deep=None, **kwargs):
        return {k: getattr(self, k) for k, v in kwargs.items()}

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def partial_fit(self, *args, **kwargs):
        pass

    def score(self, *args, **kwargs):
        return self._fn()

    def fit(self, *args):
        pass


@pytest.mark.parametrize("max_iter", [81])
def test_info(max_iter, loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            X, y = make_classification(n_samples=20, n_features=20, chunks=20)

            model = ConstantFunction()

            params = {'value': stats.uniform(0, 1)}
            alg = Hyperband(model, params, max_iter=max_iter)
            info = alg.info()
            paper_alg_info = _hyperband_paper_alg(max_iter)

            assert set(paper_alg_info.keys()) == set([b['bracket']
                                                      for b in info['brackets']])
            assert info['total_partial_fit_calls'] == sum(paper_alg_info.values())
            for bracket in info['brackets']:
                k = bracket['bracket']
                assert bracket['partial_fit_calls'] == paper_alg_info[k]
            assert info['total_partial_fit_calls'] == sum(b['partial_fit_calls']
                                                          for b in info['brackets'])
            assert info['total_models'] == sum(b['num_models']
                                               for b in info['brackets'])


def _hyperband_paper_alg(R, eta=3):
    """
    Algorithm 1 from the Hyperband paper. Only a slight modification is made,
    the ``if to_keep <= 1``: if 1 model is left there's no sense in training
    any further.

    References
    ----------
    1. "Hyperband: A novel bandit-based approach to hyperparameter optimization",
       2016 by L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar.
       https://arxiv.org/abs/1603.06560
    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R
    brackets = reversed(range(s_max + 1))
    hists = {}
    for s in brackets:
        n = int(math.ceil(B / R * eta**s / (s + 1)))
        r = int(R * eta**-s)

        T = {n for n in range(n)}
        hist = {n: 0 for n in range(n)}
        for i in range(s + 1):
            n_i = math.floor(n * eta**-i)
            r_i = r * eta**i
            L = {model: r_i for model in T}
            hist.update(L)
            to_keep = math.floor(n_i / eta)
            T = {model for i, model in enumerate(T)
                 if i < to_keep}
            if to_keep <= 1:
                break

        hists['bracket={s}'.format(s=s)] = hist

    num_partial_fit_calls = {k: sum(v.values()) for k, v in hists.items()}
    return num_partial_fit_calls
