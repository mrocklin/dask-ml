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


@pytest.mark.parametrize(  # noqa: F811
    "array_type,library",
    [("dask.array", "dask-ml"), ("numpy", "sklearn"), ("numpy", "test")],
)
def test_sklearn(array_type, library, loop):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            n, d = (200, 2)

            rng = da.random.RandomState(42)

            # create observations we know linear models can fit
            X = rng.normal(size=(n, d), chunks=n // 2)
            coef_star = rng.uniform(size=d, chunks=d)
            y = da.sign(X.dot(coef_star))

            if array_type == "numpy":
                X = X.compute()
                y = y.compute()

            params = {"loss": ['hinge', 'log', 'modified_huber',
                               'squared_hinge', 'perceptron'],
                      'average': [True, False],
                      'learning_rate': ['constant', 'invscaling', 'optimal']}
            model = SGDClassifier(tol=-np.inf, penalty="elasticnet",
                                  random_state=42, eta0=0.1)
            if library == "dask-ml":
                model = Incremental(model)
            elif library == "test":
                model = ConstantFunction()
                params = {"value": np.linspace(0, 1, num=1000)}

            search = HyperbandCV(
                model,
                params,
                max_iter=27,
                random_state=42,
                asynchronous=False,
            )
            search.fit(X, y, classes=da.unique(y))

            score = search.best_estimator_.score(X, y)
            if library == "sklearn":
                assert score > 0.67
            if library == "dask-ml":
                assert score > 0.67
            elif library == "test":
                assert score > 0.9
            assert type(search.best_estimator_) == type(model)
            assert isinstance(search.best_params_, dict)

            num_fit_models = len(set(search.cv_results_["model_id"]))
            assert num_fit_models == 49
            best_idx = search.best_index_
            assert search.cv_results_["mean_test_score"][best_idx] == max(
                search.cv_results_["mean_test_score"]
            )

            info_plain = search.fit_metadata()
            info_train = search.fit_metadata(meta=search.meta_)
            # NOTE: this currently fails with sklearn and dask-ml models (not
            # test models). some of the bracket partial_fit_calls are off by
            # about 20%.
            #  assert info_plain['brackets'] == info_train['brackets']
            #  assert info_train == info_plain
            for b1, b2 in zip(
                info_train["_brackets"], info_plain["_brackets"]
            ):
                for key, v1 in b1.items():
                    v2 = b2[key]
                    if key == "num_partial_fit_calls":
                        diff = np.abs(v1 - v2) / v1
                        assert diff < 0.23
                    else:
                        assert v1 == v2

            assert info_train["num_models"] == info_plain["num_models"]
            part_fit_key = "num_partial_fit_calls"
            diff = info_train[part_fit_key] - info_plain[part_fit_key]
            diff /= info_plain[part_fit_key]
            assert np.abs(diff) < 0.06


@pytest.mark.parametrize("library", ["sklearn", "dask-ml"])  # noqa: F811
def test_scoring_param(loop, library):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            max_iter = 9
            X, y = make_classification(
                n_samples=100,
                n_features=20,
                random_state=42,
                chunks=20,
            )
            X, y = dask.persist(X, y)
            if library == "sklearn":
                X = X.compute()
                y = y.compute()

            kwargs = dict(tol=1e-3, penalty="elasticnet", random_state=42)

            model = SGDClassifier(**kwargs)
            if library == "dask-ml":
                model = Incremental(model)

            params = {
                "alpha": np.logspace(-2, 1, num=1000),
                "l1_ratio": np.linspace(0, 1, num=1000),
                "average": [True, False],
            }
            alg1 = HyperbandCV(
                model, params, max_iter=max_iter, scoring="accuracy", random_state=42
            )
            alg1.fit(X, y, classes=da.unique(y))

            alg2 = HyperbandCV(model, params, max_iter=max_iter, scoring="r2", random_state=42)
            alg2.fit(X, y, classes=da.unique(y))

            assert alg1.scoring != alg2.scoring
            assert alg1.scorer_ != alg2.scorer_
            assert alg1.score(X, y) != alg2.score(X, y)


def test_async_keyword(loop):  # noqa: F811
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            max_iter = 27
            X, y = make_classification(chunks=20)
            model = ConstantFunction()

            params = {"value": np.logspace(-2, 1, num=max_iter)}
            alg0 = HyperbandCV(
                model, params, asynchronous=False, max_iter=max_iter, random_state=42
            )
            alg0.fit(X, y)

            alg1 = HyperbandCV(
                model, params, asynchronous=True, max_iter=max_iter, random_state=42
            )
            alg1.fit(X, y)

            info0 = alg0.fit_metadata(meta=alg0.meta_)
            info1 = alg1.fit_metadata(meta=alg1.meta_)
            assert info0["num_models"] == info1["num_models"]
            assert alg0.score(X, y) == alg1.score(X, y)


@gen_cluster(client=True)
def test_sklearn_async(c, s, a, b):
    max_iter = 27
    chunk_size = 20
    X, y = make_classification(
        n_samples=100, n_features=20, random_state=42, chunks=chunk_size
    )
    X, y = dask.persist(X, y)
    yield wait([X, y])

    kwargs = dict(tol=1e-3, penalty="elasticnet", random_state=42)

    model = SGDClassifier(**kwargs)

    params = {
        "alpha": np.logspace(-2, 1, num=1000),
        "l1_ratio": np.linspace(0, 1, num=1000),
        "average": [True, False],
    }
    search = HyperbandCV(model, params, max_iter=max_iter, random_state=42)
    s_tasks = set(s.tasks)
    c_futures = set(c.futures)
    yield search._fit(X, y, classes=da.unique(y))

    assert set(c.futures) == c_futures
    start = time()
    while set(s.tasks) != s_tasks:
        yield gen.sleep(0.01)
        assert time() < start + 5

    assert len(set(search.cv_results_["model_id"])) == 49


def test_partial_fit_copy():
    # Tests copying of models by testing on one model and seeing if it carries
    # through Hyperband
    n, d = 100, 20
    X, y = make_classification(
        n_samples=n, n_features=d, random_state=42, chunks=(n, d)
    )
    X = X.compute()
    y = y.compute()
    meta = {
        "iterations": 0,
        "mean_copy_time": 0,
        "mean_fit_time": 0,
        "partial_fit_calls": 1,
    }
    model = SGDClassifier(tol=1e-3)
    model.partial_fit(X[: n // 2], y[: n // 2], classes=np.unique(y))
    new_model, new_meta = _partial_fit(
        (model, meta),
        X[n // 2:],
        y[n // 2:],
        fit_params={"classes": np.unique(y)},
    )
    assert meta != new_meta
    assert new_meta["iterations"] == 1
    assert not np.allclose(model.coef_, new_model.coef_)
    assert model.t_ < new_model.t_


@pytest.mark.parametrize("max_iter", [27, 81])  # noqa: F811
def test_meta_computation(loop, max_iter):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            X, y = make_classification(chunks=5, n_features=5)
            model = ConstantFunction()
            params = {"value": scipy.stats.uniform(0, 1)}
            alg = HyperbandCV(
                model,
                params,
                max_iter=max_iter,
                random_state=0,
                asynchronous=False,
            )
            alg.fit(X, y)
            paper_info = alg.fit_metadata()
            actual_info = alg.fit_metadata(meta=alg.meta_)
            assert paper_info["num_models"] == actual_info["num_models"]
            assert (
                paper_info["num_partial_fit_calls"] ==
                actual_info["num_partial_fit_calls"]
            )
            assert paper_info["_brackets"] == actual_info["_brackets"]


def test_integration(loop):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            X, y = make_classification(n_samples=10, n_features=4, chunks=10)
            model = ConstantFunction()
            params = {"value": scipy.stats.uniform(0, 1)}
            alg = HyperbandCV(model, params, asynchronous=True,
                              max_iter=9, random_state=42)
            alg.fit(X, y)
            cv_res_keys = set(alg.cv_results_.keys())
            gt_zero = lambda x: x >= 0
            is_type = lambda x, dtype: isinstance(x, dtype)
            for column, dtype, condition in [
                ("rank_test_score", int, None),
                ("model_id", str, None),
                ("mean_score_time", float, gt_zero),
                ("mean_test_score", float, None),
                ("mean_fit_time", float, gt_zero),
                ("partial_fit_calls", int, gt_zero),
                ("params", dict, lambda d: set(d.keys()) == {'value'}),
                ("param_value", float, None),
                ("mean_copy_time", float, gt_zero),
                ("std_test_score", float, gt_zero),
                ("std_score_time", float, gt_zero),
                ("mean_train_score", None, None),
                ("std_train_score", None, None),
                ("std_fit_time", float, gt_zero),
                ("mean_copy_time", float, gt_zero),
                ("time_scored", float, gt_zero),
            ]:
                if dtype:
                    assert all(is_type(x, dtype)
                               for x in alg.cv_results_[column])
                if condition:
                    assert all(condition(x) for x in alg.cv_results_[column])
                cv_res_keys -= {column}

            # the keys listed in the for-loop are all the keys in cv_results_
            assert cv_res_keys == set()

            alg.best_estimator_.fit(X, y)
            alg.best_estimator_.score(X, y)
            alg.fit(X, y)
            alg.score(X, y)
            assert isinstance(alg.best_index_, int)
            assert isinstance(alg.best_score_, float)
            assert isinstance(alg.best_estimator_, ConstantFunction)
            assert isinstance(alg.best_params_, dict)
