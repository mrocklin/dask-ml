import math
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.utils import check_random_state

from tornado import gen
from toolz import groupby
from dask.distributed import as_completed, default_client, futures_of
from dask_searchcv.model_selection import DaskBaseSearchCV
import dask.array as da
from ..datasets import make_classification
from ..wrappers import Incremental
from ._split import train_test_split


def _get_nr(R, eta=3):
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R

    brackets = list(reversed(range(int(s_max + 1))))
    N = [math.ceil(B / R * eta**s / (s + 1)) for s in brackets]
    R = [int(R * eta**-s) for s in brackets]
    return list(map(int, N)), list(map(int, R)), brackets


def _partial_fit(model_meta_future, X, y, meta=None, fit_params={},
                 dry_run=False, **kwargs):
    assert isinstance(X, np.ndarray)
    assert y is None or isinstance(y, np.ndarray)
    model, m = model_meta_future
    if meta is None:
        meta = m
    while meta['iterations'] < meta['partial_fit_calls']:
        if not dry_run:
            model.partial_fit(X, y, **fit_params)
        meta["iterations"] += 1
    return model, meta


def _score(model_and_meta, x, y, dry_run=False):
    model, meta = model_and_meta
    #  seed = hash(((k, v) for k, v in meta.items())) % 2**32
    #  rng = np.random.RandomState(seed)
    if dry_run:
        #  score = rng.rand()
        score = 0
    else:
        # TODO: change this to dask_ml.get_scorer
        #  score = model.score(x, y)
        score = 0
    meta.update(score=score)
    assert meta['iterations'] > 0
    return meta


def _top_k(results, k=1, sort="score"):
    res = sorted(results, key=lambda x: x[sort])[-k:]
    assert len(res) == k
    return res


def _to_promote(result, completed_jobs, eta=None):
    bracket_models = [r for r in completed_jobs
                      if r['bracket_iter'] == result['bracket_iter'] and
                      r['bracket'] == result['bracket']]

    bracket_completed = len(bracket_models) == result['num_models']
    if bracket_completed:
        to_keep = len(bracket_models) // eta
        if to_keep <= 1:
            return []
        top_results = _top_k(bracket_models, k=to_keep, sort="score")
        for job in top_results:
            job["num_models"] = to_keep
            job['partial_fit_calls'] *= eta
            job['bracket_iter'] += 1
        return top_results
    return []


def _create_model(model, params, random_state=42):
    model = clone(model).set_params(**params)
    if 'random_state' in model.get_params():
        model.set_params(random_state=random_state)
    #  meta = {}  # right now no meta information
    return model, None


def _bracket_name(s):
    return "bracket={s}".format(s=s)


def _model_id(s, n_i):
    return "bracket={s}-{n_i}".format(s=s, n_i=n_i)


async def _hyperband(model, params, X, y, max_iter=None, eta=None,
                     dry_run=False, fit_params={}, random_state=42):
    client = default_client()
    rng = check_random_state(random_state)
    N, R, brackets = _get_nr(max_iter, eta=eta)
    params = iter(ParameterSampler(params, n_iter=sum(N),
                                   random_state=random_state))
    if isinstance(model, Incremental):  # this should probably be handled elsewhere
        model = model.estimator
    model_futures = {_model_id(s, n_i):
                     client.submit(_create_model, model, next(params),
                                   random_state=random_state)
                     for s, n, r in zip(brackets, N, R) for n_i in range(n)}

    # lets assume everything in fit_params is small and make it concrete
    fit_params = await client.compute(fit_params)

    r = train_test_split(X, y, test_size=0.15, random_state=random_state)
    X_train, X_test, y_train, y_test = r
    if isinstance(X, da.Array):
        X_train = futures_of(X_train.persist())
        # X_test = futures_of(X_test.persist())  # if we should operate in chunks
        X_test = client.compute(X_test)  # if we should pass a single chunk
    else:
        X_train, X_test = await client.scatter([X_train, X_test])
    if isinstance(y, da.Array):
        y_train = futures_of(y_train.persist())
        # y_test = futures_of(y_test.persist())  # if we should operate in chunks
        y_test = client.compute(y_test)  # if we should pass a single chunk
    else:
        y_train, y_test = await client.scatter([y_train, y_test])

    info = {s: [{"partial_fit_calls": r, "bracket": _bracket_name(s),
                 "num_models": n, 'bracket_iter': 0,
                 "model_id": _model_id(s, n_i),
                 "iterations": 0}
                for n_i in range(n)]
            for s, n, r in zip(brackets, N, R)}

    idx = {(i, j): rng.choice(len(X_train))
           for i, metas in enumerate(info.values())
           for j, meta in enumerate(metas)}

    model_meta_futures = {meta["model_id"]:
                          client.submit(_partial_fit,
                                        model_futures[meta["model_id"]],
                                        X_train[idx[(i, j)]],
                                        y_train[idx[(i, j)]],
                                        meta=meta, id=meta['model_id'],
                                        dry_run=dry_run,
                                        fit_params=fit_params, pure=False)
                          for i, bracket_metas in enumerate(info.values())
                          for j, meta in enumerate(bracket_metas)}

    assert set(model_meta_futures.keys()) == set(model_futures.keys())

    score_futures = [client.submit(_score, model_meta_future, X_test, y_test,
                                   dry_run=dry_run, pure=False)
                     for _id, model_meta_future in model_meta_futures.items()]

    completed_jobs = {}
    seq = as_completed(score_futures)
    async for future in seq:
        result = await future
        assert result['iterations'] > 0

        completed_jobs[result['model_id']] = result
        jobs = _to_promote(result, completed_jobs.values(), eta=eta)
        for job in jobs:
            assert job['iterations'] > 0
            i = rng.choice(len(X_train))

            # This block prevents communication of the model
            model_meta_future = model_meta_futures[job["model_id"]]
            model_meta_future = client.submit(_partial_fit, model_meta_future,
                                              X_train[i], y_train[i],
                                              meta=job, dry_run=dry_run,
                                              id=job['model_id'],
                                              fit_params=fit_params)
            model_meta_futures[job["model_id"]] = model_meta_future

            score_future = client.submit(_score, model_meta_future, X_test,
                                         y_test, dry_run=dry_run)
            seq.add(score_future)

    assert completed_jobs.keys() == model_meta_futures.keys()
    return list(completed_jobs.values()), model_meta_futures


class HyperbandCV(DaskBaseSearchCV):
    def __init__(self, model, params, max_iter=81, eta=3, random_state=42,
                 scoring=None, iid=True, cv=2, cache_cv=False, **kwargs):
        self.model = model
        self.params = params
        self.max_iter = max_iter
        self.eta = eta
        self.random_state = random_state
        self.scoring = scoring
        if not iid:
            raise ValueError('Please specify iid=True. Hyperband assumes that '
                             'each observation is independent and '
                             'identitically distributed because it trains on '
                             'each block of X and y')
        if cv != 2:
            raise ValueError('Please specify cv=2. Future work is to '
                             'allow arbitrary cross-validation (and pull '
                             'request welcome!).')
        if cache_cv:
            raise ValueError('Please specify cv_cache=False. We do not '
                             'support caching intermediate results yet, '
                             'but this is on the roadmap (and pull requests '
                             'welcome!).')

        super(HyperbandCV, self).__init__(model, iid=iid, cv=cv,
                                          cache_cv=cache_cv, **kwargs)

    def fit(self, X, y, **fit_params):
        return default_client().sync(self._fit, X, y, **fit_params)

    @gen.coroutine
    def _fit(self, X, y, **fit_params):
        if isinstance(X, np.ndarray):
            X = da.from_array(X, chunks=X.shape)
        if isinstance(y, np.ndarray):
            y = da.from_array(y, chunks=y.shape)
        r = yield _hyperband(self.model, self.params, X, y,
                             max_iter=self.max_iter,
                             fit_params=fit_params, eta=self.eta,
                             random_state=self.random_state)
        history, model_futures = r
        self.history_ = history
        self._model_futures = model_futures

        # TODO: set best index, best model, etc
        return self

    @property
    def _models_and_meta(self):
        return default_client().gather(self._model_futures)

    def info(self, history=None):
        res = default_client().sync(self._info, history=history)
        assert isinstance(res, dict)
        return res

    @gen.coroutine
    def _info(self, history=None):
        if history is None:
            X, y = make_classification(n_samples=10, n_features=4, chunks=10,
                                       random_state=self.random_state)
            r = yield _hyperband(self.model, self.params, X, y,
                                 max_iter=self.max_iter,
                                 dry_run=True, eta=self.eta,
                                 random_state=self.random_state)
            history, _ = r

        brackets = groupby("bracket", history)
        keys = sorted(brackets.keys())
        values = [brackets[k] for k in keys]
        bracket_info = [{'num_models': max(r['num_models'] for r in v),
                         'partial_fit_calls': sum(r['partial_fit_calls']
                                                  for r in v),
                         'bracket': k}
                        for k, v in zip(keys, values)]
        info = {'total_partial_fit_calls': sum(r['partial_fit_calls']
                                               for r in history),
                'total_models': len(set([r['model_id'] for r in history]))}
        res = {'brackets': bracket_info}
        res.update(**info)
        assert isinstance(res, dict)
        return res
