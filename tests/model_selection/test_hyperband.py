import pytest
import numpy as np
import scipy.stats as stats
import math

from sklearn.linear_model import SGDClassifier

from distributed import Client
from distributed.utils_test import loop, cluster
import dask.array as da

from dask_ml.datasets import make_classification
from dask_ml.model_selection import HyperbandCV
from dask_ml.wrappers import Incremental


class ConstantFunction(dict):
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


@pytest.mark.parametrize("array_type", ["numpy", "dask.array"])
@pytest.mark.parametrize("library", ["dask-ml", "sklearn"])
def test_sklearn(array_type, library, loop, max_iter=9):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            chunk_size = 20
            X, y = make_classification(n_samples=100, n_features=20,
                                       random_state=42, chunks=chunk_size)
            if array_type == "numpy":
                X = X.compute()
                y = y.compute()
                chunk_size = X.shape[0]

            kwargs = dict(tol=1e-3, penalty='elasticnet', random_state=42)
            if library == "sklearn":
                model = SGDClassifier(**kwargs)
            elif library == "dask-ml":
                model = Incremental(SGDClassifier(), **kwargs)

            params = {'alpha': np.logspace(-2, 1, num=1000),
                      'l1_ratio': np.linspace(0, 1, num=1000),
                      'average': [True, False]}
            search = HyperbandCV(model, params, max_iter=max_iter,
                                 random_state=42)
            search.fit(X, y, classes=da.unique(y))

            models = [v[0] for v in search._models_and_meta.values()]
            assert all(hasattr(model, "t_") for model in models)

            # Test fraction of 0.15 is hardcoded into _hyperband
            def _iters(model):
                t_ = (model.estimator.t_ if hasattr(model, 'estimator')
                      else model.t_)
                return (t_ - 1) / (chunk_size * (1 - 0.15))
            iters = {_iters(model) for model in models}
            assert len(iters) > 1
            assert 1 <= min(iters) < max(iters) <= max_iter


@pytest.mark.parametrize("max_iter", [81])
def test_info(max_iter, loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            model = ConstantFunction()
            params = {'value': stats.uniform(0, 1)}
            alg = HyperbandCV(model, params, max_iter=max_iter, random_state=0)
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

            X, y = make_classification(n_samples=10, n_features=4, chunks=10)
            alg.fit(X, y)
            info_after_fit = alg.info(history=alg.history_)
            assert info_after_fit == info


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
    brackets = reversed(range(int(s_max + 1)))
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
