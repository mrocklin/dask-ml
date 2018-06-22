import math
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler

from toolz import groupby
from distributed import as_completed, default_client
from dask_searchcv.model_selection import DaskBaseSearchCV


def _get_nr(R, eta=3):
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R

    brackets = list(reversed(range(int(s_max + 1))))
    N = [math.ceil(B / R * eta**s / (s + 1)) for s in brackets]
    R = [int(R * eta**-s) for s in brackets]
    return list(map(int, N)), list(map(int, R)), brackets


def _partial_fit(model_and_meta, x, y, meta=None, fit_params={},
                 dry_run=False):
    model, m = model_and_meta
    if meta is None:
        meta = m
    while meta['iterations'] < meta['partial_fit_calls']:
        if not dry_run:
            model.partial_fit(x, y, **fit_params)
        meta["iterations"] += 1
    return model, meta


def _score(model_and_meta, x, y, seed=42, dry_run=False):
    seed = seed % 2**32 - 1
    model, meta = model_and_meta
    rng = np.random.RandomState(seed)
    if dry_run:
        score = rng.rand()
    else:
        # TODO: change this to dask_ml.get_scorer
        #  score = model.score(x, y)
        score = rng.rand()
    meta.update(score=score)
    return meta


def _top_k(results, k=1, sort="score"):
    scores = [r[sort] for r in results]
    idx = np.argsort(scores)
    out = [results[i] for i in idx[-k:]]
    return out


def _to_promote(result, completed_jobs, async_=None, eta=None):
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


def _create_model(model, params):
    model = clone(model).set_params(**params)
    return model, {'iterations': 0}


def _bracket_name(s):
    return "bracket={s}".format(s=s)


def _model_id(s, n_i):
    return "bracket={s}-{n_i}".format(s=s, n_i=n_i)


def _hyperband(client, model, params, X, y, max_iter=None, eta=None,
               async_=None, dry_run=False, fit_params={}):
    N, R, brackets = _get_nr(max_iter, eta=eta)
    params = iter(ParameterSampler(params, n_iter=sum(N)))
    model_futures = {_model_id(s, n_i): client.submit(_create_model, model,
                                                      next(params))
                     for s, n, r in zip(brackets, N, R) for n_i in range(n)}

    info = {s: [{"partial_fit_calls": r, "bracket": _bracket_name(s),
                 "num_models": n, 'bracket_iter': 0,
                 "model_id": _model_id(s, n_i),
                 "iterations": 0}
                for n_i in range(n)]
            for s, n, r in zip(brackets, N, R)}

    kwargs = {'dry_run': dry_run, 'fit_params': fit_params}
    model_meta_futures = {meta["model_id"]:
                          client.submit(_partial_fit,
                                        model_futures[meta["model_id"]], X, y,
                                        meta=meta, **kwargs)
                          for _, r_metas in info.items()
                          for meta in r_metas}

    score_futures = [client.submit(_score, model_meta_future, X, y,
                                   seed=hash(_id), dry_run=dry_run)
                     for _id, model_meta_future in model_meta_futures.items()]

    completed_jobs = {}
    seq = as_completed(score_futures)
    for future in seq:
        result = future.result()

        # TODO if we want to map partial_fit/score to blocks
        #  i = get_xy_block_id(result)
        #  print(result)

        completed_jobs[result['model_id']] = result
        jobs = _to_promote(result, completed_jobs.values(), eta=eta)
        for job in jobs:
            # This block prevents communication of the model
            model_future = model_futures[job["model_id"]]
            model_future = client.submit(_partial_fit, model_future, X, y,
                                         meta=job, dry_run=dry_run,
                                         fit_params=fit_params)

            score_future = client.submit(_score, model_future, X, y,
                                         seed=hash(job['model_id']),
                                         dry_run=dry_run)
            model_futures[job["model_id"]] = model_future
            seq.add(score_future)

    return list(completed_jobs.values())


class HyperbandCV(DaskBaseSearchCV):
    def __init__(self, model, params, max_iter=81, client=None, async_=False):
        self.model = model
        self.params = params
        self.client = default_client()
        self.async_ = async_
        self.max_iter = max_iter

    def fit(self, X, y, **fit_params):
        history = _hyperband(self.client, self.model, self.params, X, y,
                             async_=self.async_, max_iter=self.max_iter,
                             fit_params=fit_params)
        self.history = history

        # TODO: set best index, best model, etc
        return self

    def info(self):
        x = y = None
        history = _hyperband(self.client, self.model, self.params, x, y,
                             async_=self.async_, R=self.max_iter, dry_run=True)
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
        return res
