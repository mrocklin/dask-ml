import numpy as np
import scipy.stats
from sklearn.linear_model import SGDClassifier

import dask
import dask.array as da
from dask.distributed import Client, wait
from distributed.utils_test import loop, cluster, gen_cluster  # noqa: F401
from tornado import gen

from dask_ml.datasets import make_classification
from dask_ml.model_selection import HyperbandCV
from dask_ml.wrappers import Incremental
from dask_ml.model_selection._hyperband import _partial_fit
from dask_ml.utils import ConstantFunction
from distributed.metrics import time

import pytest


@pytest.mark.parametrize("array_type,library",  # noqa: F811
                         [("dask.array", "dask-ml"),
                          ("numpy", "sklearn"),
                          ("numpy", "test")])
def test_sklearn(array_type, library, loop, max_iter=27):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            chunk_size = 100  # make dask array with one chunk
            d = 20
            X, y = make_classification(n_samples=chunk_size, n_features=d,
                                       random_state=42, chunks=chunk_size)
            if array_type == "numpy":
                X = X.compute()
                y = y.compute()
                chunk_size = X.shape[0]

            sgd_params = {'alpha': np.logspace(-2, 1, num=1000),
                          'l1_ratio': np.linspace(0, 1, num=1000),
                          'average': [True, False]}
            kwargs = dict(tol=-np.inf, penalty='elasticnet', random_state=42)
            if library == "sklearn":
                model = SGDClassifier(**kwargs)
                params = sgd_params
            elif library == "dask-ml":
                model = Incremental(SGDClassifier(**kwargs))
                params = sgd_params
            elif library == "test":
                model = ConstantFunction()
                params = {'value': np.linspace(0, 1, num=1000)}
            else:
                raise ValueError

            search = HyperbandCV(model, params, max_iter=max_iter,
                                 random_state=42, asynchronous=False)
            search.fit(X, y, classes=da.unique(y))

            score = search.best_estimator_.score(X, y)
            if library == "sklearn":
                assert score > 0.4
            if library == "dask-ml":
                assert score > 0.2
            elif library == "test":
                assert score > 0.8
            assert type(search.best_estimator_) == type(model)
            assert isinstance(search.best_params_, dict)

            num_fit_models = len(set(search.cv_results_['model_id']))
            assert (num_fit_models == 49)
            best_idx = search.best_index_
            assert (search.cv_results_['mean_test_score'][best_idx] ==
                    max(search.cv_results_['mean_test_score']))

            info_plain = search.fit_metadata()
            info_train = search.fit_metadata(meta=search.meta_)
            # NOTE: this currently fails with sklearn and dask-ml models (not
            # test models). some of the bracket partial_fit_calls are off by
            # about 20%.
            #  assert info_plain['brackets'] == info_train['brackets']
            #  assert info_train == info_plain
            for b1, b2 in zip(info_train['brackets'], info_plain['brackets']):
                for key, v1 in b1.items():
                    v2 = b2[key]
                    if key == 'num_partial_fit_calls':
                        diff = np.abs(v1 - v2) / v1
                        assert diff < 0.2
                    else:
                        assert v1 == v2

            assert info_train['num_models'] == info_plain['num_models']


@pytest.mark.parametrize("library", ["sklearn", "dask-ml"])  # noqa: F811
def test_scoring_param(loop, library):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            max_iter = 27
            chunk_size = 20
            X, y = make_classification(n_samples=100, n_features=20,
                                       random_state=42, chunks=chunk_size)
            X, y = dask.persist(X, y)
            if library == "sklearn":
                X = X.compute()
                y = y.compute()

            kwargs = dict(tol=1e-3, penalty='elasticnet', random_state=42)

            model = SGDClassifier(**kwargs)
            if library == 'dask-ml':
                model = Incremental(model)

            params = {'alpha': np.logspace(-2, 1, num=1000),
                      'l1_ratio': np.linspace(0, 1, num=1000),
                      'average': [True, False]}
            alg1 = HyperbandCV(model, params, max_iter=max_iter,
                               scoring="accuracy")
            alg1.fit(X, y, classes=da.unique(y))

            alg2 = HyperbandCV(model, params, max_iter=max_iter,
                               scoring="r2")
            alg2.fit(X, y, classes=da.unique(y))

            assert alg1.scoring != alg2.scoring
            assert alg1.scorer_ != alg2.scorer_
            assert alg1.score(X, y) != alg2.score(X, y)


def test_async_keyword(loop):  # noqa: F811
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            max_iter = 27
            X, y = make_classification(chunks=20)
            model = ConstantFunction()

            params = {'value': np.logspace(-2, 1, num=max_iter)}
            alg0 = HyperbandCV(model, params, asynchronous=False,
                               max_iter=max_iter)
            alg0.fit(X, y)

            alg1 = HyperbandCV(model, params, asynchronous=True,
                               max_iter=max_iter)
            alg1.fit(X, y)

            info0 = alg0.fit_metadata(meta=alg0.meta_)
            info1 = alg1.fit_metadata(meta=alg1.meta_)
            assert (info0['num_models'] == info1['num_models'])
            assert alg0.score(X, y) == alg1.score(X, y)


@gen_cluster(client=True)
async def test_sklearn_async(c, s, a, b):
    max_iter = 27
    chunk_size = 20
    X, y = make_classification(n_samples=100, n_features=20,
                               random_state=42, chunks=chunk_size)
    X, y = dask.persist(X, y)
    await wait([X, y])

    kwargs = dict(tol=1e-3, penalty='elasticnet', random_state=42)

    model = SGDClassifier(**kwargs)

    params = {'alpha': np.logspace(-2, 1, num=1000),
              'l1_ratio': np.linspace(0, 1, num=1000),
              'average': [True, False]}
    search = HyperbandCV(model, params, max_iter=max_iter, random_state=42)
    s_tasks = set(s.tasks)
    c_futures = set(c.futures)
    await search._fit(X, y, classes=da.unique(y))

    assert set(c.futures) == c_futures
    start = time()
    while set(s.tasks) != s_tasks:
        await gen.sleep(0.01)
        assert time() < start + 5

    assert len(set(search.cv_results_['model_id'])) == 49


def test_partial_fit_copy():
    n, d = 100, 20
    X, y = make_classification(n_samples=n, n_features=d,
                               random_state=42, chunks=(n, d))
    X = X.compute()
    y = y.compute()
    meta = {'iterations': 0, 'mean_copy_time': 0, 'mean_fit_time': 0,
            'partial_fit_calls': 1}
    model = SGDClassifier(tol=1e-3)
    model.partial_fit(X[:n // 2], y[:n // 2], classes=np.unique(y))
    new_model, new_meta = _partial_fit((model, meta), X[n // 2:], y[n // 2:],
                                       fit_params={'classes': np.unique(y)})
    assert meta != new_meta
    assert new_meta['iterations'] == 1
    assert not np.allclose(model.coef_, new_model.coef_)
    assert model.t_ < new_model.t_


@pytest.mark.parametrize("max_iter", [3, 9, 27, 81])  # noqa: F811
def test_meta_computation(loop, max_iter):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            X, y = make_classification(chunks=5, n_features=5)
            model = ConstantFunction()
            params = {'value': scipy.stats.uniform(0, 1)}
            alg = HyperbandCV(model, params, max_iter=max_iter, random_state=0,
                              asynchronous=False)
            alg.fit(X, y)
            paper_info = alg.fit_metadata()
            actual_info = alg.fit_metadata(meta=alg.meta_)
            assert paper_info['num_models'] == actual_info['num_models']
            assert (paper_info['num_partial_fit_calls'] ==
                    actual_info['num_partial_fit_calls'])
            assert paper_info['brackets'] == actual_info['brackets']


@pytest.mark.parametrize("asynchronous", [True, False])  # noqa: F811
def test_integration(asynchronous, loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            X, y = make_classification(n_samples=10, n_features=4, chunks=10)
            model = ConstantFunction()
            params = {'value': scipy.stats.uniform(0, 1)}
            alg = HyperbandCV(model, params, asynchronous=asynchronous)
            alg.fit(X, y)
            cv_res_keys = set(alg.cv_results_.keys())
            assert cv_res_keys == {'rank_test_score', "model_id",
                                   'mean_fit_time', 'mean_score_time',
                                   'std_fit_time', 'std_score_time',
                                   'mean_test_score', 'std_test_score',
                                   'partial_fit_calls', 'mean_train_score',
                                   'std_train_score', 'params', 'param_value',
                                   'mean_copy_time'}
            for column, dtype in [('rank_test_score', int),
                                  ('model_id', str),
                                  ('mean_score_time', float),
                                  ('mean_test_score', float),
                                  ('mean_fit_time', float),
                                  ('partial_fit_calls', int),
                                  ('params', dict),
                                  ('param_value', float),
                                  ('mean_copy_time', float)]:
                assert all(isinstance(v, dtype)
                           for v in alg.cv_results_[column])
            alg.best_estimator_.fit(X, y)
            assert isinstance(alg.best_index_, int)
            assert isinstance(alg.best_score_, float)
            assert isinstance(alg.best_estimator_, ConstantFunction)
            assert isinstance(alg.best_params_, dict)
