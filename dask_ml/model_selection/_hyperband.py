import math
import numpy as np
import toolz
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
from distributed.metrics import time
from distributed import Future
from copy import deepcopy


def _get_nr(R, eta=3):
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R

    brackets = list(reversed(range(int(s_max + 1))))
    N = [math.ceil(B / R * eta**s / (s + 1)) for s in brackets]
    R = [int(R * eta**-s) for s in brackets]
    return list(map(int, N)), list(map(int, R)), brackets


def _partial_fit(model_and_meta, X, y, meta=None, fit_params={},
                 dry_run=False, **kwargs):
    start = time()
    assert isinstance(X, np.ndarray)
    assert y is None or isinstance(y, np.ndarray)
    model = deepcopy(model_and_meta[0])
    if meta is None:
        meta = deepcopy(model_and_meta[1])
    else:
        meta = deepcopy(meta)
    meta['mean_copy_time'] += time() - start
    while meta['iterations'] < meta['partial_fit_calls']:
        if not dry_run:
            model.partial_fit(X, y, **fit_params)
        meta["iterations"] += 1
    meta['mean_fit_time'] += time() - start
    return model, meta


def _score(model_and_meta, x, y, dry_run=False):
    start = time()
    model = deepcopy(model_and_meta[0])
    meta = deepcopy(model_and_meta[1])
    meta['mean_copy_time'] += time() - start
    if dry_run:
        score = 0
        #  BUG: uncommenting these lines gives a bug in test_info for R=27, 81
        #  seed = hash(((k, v) for k, v in meta.items())) % 2**32
        #  rng = np.random.RandomState(seed)
        #  score = rng.rand()
    else:
        # TODO: change this to dask_ml.get_scorer
        score = model.score(x, y)
    meta.update(score=score)
    assert meta['iterations'] > 0
    meta['mean_score_time'] += time() - start
    meta['mean_test_score'] = score
    return meta


def _top_k(results, k=1, sort="score"):
    res = toolz.topk(k, results, key=sort)
    return list(res)


def _to_promote(result, completed_jobs, eta=None, asynchronous=None):
    bracket_models = [r for r in completed_jobs
                      if r['bracket_iter'] == result['bracket_iter'] and
                      r['bracket'] == result['bracket']]

    if not asynchronous:
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
    raise ValueError


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
                     dry_run=False, fit_params={}, random_state=42,
                     asynchronous=None):
    client = default_client()
    rng = check_random_state(random_state)
    N, R, brackets = _get_nr(max_iter, eta=eta)
    params = iter(ParameterSampler(params, n_iter=sum(N),
                                   random_state=random_state))
    if isinstance(model, Incremental):  # this should probably be handled elsewhere
        model = model.estimator
    params = {_model_id(s, n_i): next(params)
              for s, n, r in zip(brackets, N, R) for n_i in range(n)}
    model_futures = {_model_id(s, n_i):
                     client.submit(_create_model, model,
                                   params[_model_id(s, n_i)],
                                   random_state=random_state)
                     for s, n, r in zip(brackets, N, R) for n_i in range(n)}

    # lets assume everything in fit_params is small and make it concrete
    fit_params = await client.compute(fit_params)

    r = train_test_split(X, y, test_size=0.15, random_state=random_state)
    X_train, X_test, y_train, y_test = r
    if isinstance(X, da.Array):
        X_train = futures_of(X_train.persist())
        X_test = client.compute(X_test)
    else:
        X_train, X_test = await client.scatter([X_train, X_test])
    if isinstance(y, da.Array):
        y_train = futures_of(y_train.persist())
        y_test = client.compute(y_test)
    else:
        y_train, y_test = await client.scatter([y_train, y_test])

    info = {s: [{"partial_fit_calls": r, "bracket": _bracket_name(s),
                 "num_models": n, 'bracket_iter': 0,
                 "model_id": _model_id(s, n_i),
                 'mean_fit_time': 0, 'mean_score_time': 0,
                 'std_fit_time': 0, 'std_score_time': 0,
                 'mean_test_score': 0, 'std_test_score': 0,
                 'mean_train_score': None, 'std_train_score': None,
                 "iterations": 0, 'mean_copy_time': 0}
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
    all_models = {k: [v]
                        for k, v in deepcopy(model_meta_futures).items()}
    history = []
    async for future in seq:
        result = await future
        history += [result]

        completed_jobs[result['model_id']] = result
        jobs = _to_promote(result, completed_jobs.values(), eta=eta,
                           asynchronous=asynchronous)
        for job in jobs:
            assert job['iterations'] > 0
            i = rng.choice(len(X_train))

            # This block prevents communication of the model
            model_id = job['model_id']
            model_meta_future = model_meta_futures[model_id]
            model_meta_future = client.submit(_partial_fit, model_meta_future,
                                              X_train[i], y_train[i],
                                              meta=job, dry_run=dry_run,
                                              id=model_id,
                                              fit_params=fit_params)
            all_models[model_id] += [model_meta_future]
            model_meta_futures[model_id] = model_meta_future

            score_future = client.submit(_score, model_meta_future, X_test,
                                         y_test, dry_run=dry_run)
            seq.add(score_future)

    assert completed_jobs.keys() == model_meta_futures.keys()
    return (params, model_meta_futures, history, all_models,
            list(completed_jobs.values()))


class HyperbandCV(DaskBaseSearchCV):
    def __init__(self, model, params, max_iter=81, eta=3, asynchronous=None,
                 random_state=42, scoring=None):
        self.model = model
        self.params = params
        self.max_iter = max_iter
        self.eta = eta
        self.random_state = random_state
        self.scoring = scoring
        if asynchronous is None:
            asynchronous = False
        self.asynchronous = asynchronous

        super(HyperbandCV, self).__init__(model)

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
                             asynchronous=self.asynchronous,
                             fit_params=fit_params, eta=self.eta,
                             random_state=self.random_state)
        params, model_meta_futures, history, all_models, jobs = r

        self.jobs_ = jobs
        self.history_ = history
        self._model_meta_futures = model_meta_futures
        self.__all_models = all_models

        cv_res, best_idx = _get_cv_results(jobs, params)
        best_id = cv_res['model_id'][best_idx]

        best_model_and_meta = yield model_meta_futures[best_id]
        self.best_estimator_ = best_model_and_meta[0]

        self.cv_results_ = cv_res
        self.best_index_ = best_idx
        self.n_splits_ = 1  # TODO: increase this! It's hard-coded right now
        self.multimetric_ = False

        return self

    @property
    def _models_and_meta(self):
        return default_client().gather(self._model_meta_futures)

    @property
    def _all_models(self):
        models_and_meta = default_client().gather(self.__all_models)
        models = {k: [vi[0] for vi in v] for k, v in models_and_meta.items()}
        return models

    def info(self, jobs=None):
        res = default_client().sync(self._info, jobs=jobs)
        assert isinstance(res, dict)
        return res

    @gen.coroutine
    def _info(self, jobs=None):
        if jobs is None:
            X, y = make_classification(n_samples=10, n_features=4, chunks=10,
                                       random_state=self.random_state)
            r = yield _hyperband(self.model, self.params, X, y,
                                 max_iter=self.max_iter,
                                 asynchronous=self.asynchronous,
                                 dry_run=True, eta=self.eta,
                                 random_state=self.random_state)
            _, _, _, _, jobs = r

        brackets = groupby("bracket", jobs)
        keys = sorted(brackets.keys())
        values = [brackets[k] for k in keys]
        bracket_info = [{'num_models': max(r['num_models'] for r in v),
                         'partial_fit_calls': sum(r['partial_fit_calls']
                                                  for r in v),
                         'bracket': k}
                        for k, v in zip(keys, values)]
        info = {'total_partial_fit_calls': sum(r['partial_fit_calls']
                                               for r in jobs),
                'total_models': len(set([r['model_id']
                                         for r in jobs]))}
        res = {'brackets': bracket_info}
        res.update(**info)
        assert isinstance(res, dict)
        return res


def _get_cv_results(history, params):
    scores = [h['score'] for h in history]
    best_idx = int(np.argmax(scores))
    keys = set(toolz.merge(history).keys())
    for unused in ['bracket', 'iterations', 'num_models', 'bracket_iter',
                   'score']:
        keys.discard(unused)
    cv_results = {k: [h[k] for h in history] for k in keys}

    params = [params[model_id] for model_id in cv_results['model_id']]
    cv_results['params'] = params
    params = {'param_' + k: [param[k] for param in params]
              for k in params[0].keys()}
    ranks = np.argsort(scores)[::-1]
    cv_results['rank_test_score'] = ranks.tolist()
    cv_results.update(params)
    return cv_results, best_idx
