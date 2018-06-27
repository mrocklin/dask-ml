from copy import deepcopy
import math

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.utils import check_random_state
from sklearn.metrics.scorer import check_scoring
from tornado import gen
import toolz

import dask.array as da
from dask.distributed import as_completed, default_client, futures_of
from distributed.metrics import time
from dask_searchcv.model_selection import DaskBaseSearchCV

from ..wrappers import Incremental
from ._split import train_test_split
from .._utils import copy_learned_attributes


def _get_hyperband_params(R, eta=3):
    """
    Arguments
    ---------
    R : int
        The maximum number of iterations desired.
    Returns
    -------
    N : list
        The number of models for each bracket
    R : list
        The number of iterations for each bracket
    brackets : list
        The bracket identifier.
    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R

    brackets = list(reversed(range(int(s_max + 1))))
    N = [math.ceil(B / R * eta**s / (s + 1)) for s in brackets]
    R = [int(R * eta**-s) for s in brackets]
    return list(map(int, N)), list(map(int, R)), brackets


def _partial_fit(model_and_meta, X, y, meta=None, fit_params={}):
    """
    Call partial_fit on a classifiers with X and y

    Arguments
    ---------
    model_and_meta : tuple, (model: any, meta: dict)
        model needs to support partial_fit. meta is assumed to have keys
        ``mean_copy_time, iterations, partial_fit_calls, mean_fit_time``.
        partial_fit will be called on the model until
        ``meta['iterations'] >= meta['partial_fit_calls']``
    X, y : np.ndarray, np.ndarray
        Training data
    meta : dict
        If present, replace ``model_and_meta[1]`` with this object
    fit_params : dict
        Keyword args to pass to partial_fit

    Returns
    -------
    model : any
        The model that has been fit.
    meta : dict
        A new dictionary with updated information.

    This function does not modify any item in place.

    """
    start = time()
    model = deepcopy(model_and_meta[0])
    if meta is None:
        meta = deepcopy(model_and_meta[1])
    else:
        meta = deepcopy(meta)
    meta['mean_copy_time'] += time() - start
    while meta['iterations'] < meta['partial_fit_calls']:
        # TODO: handle this unwrapping elsewhere
        if isinstance(model, Incremental):
            model.estimator.partial_fit(X, y, **fit_params)
        else:
            model.partial_fit(X, y, **fit_params)
        meta["iterations"] += 1
    if isinstance(model, Incremental):
        copy_start = time()
        copy_learned_attributes(model.estimator, model)
        meta['mean_copy_time'] += time() - copy_start
    meta['mean_fit_time'] += time() - start
    return model, meta


def _score(model_and_meta, x, y, scorer=None):
    model = model_and_meta[0]
    score = scorer(model, x, y)

    start = time()
    meta = deepcopy(model_and_meta[1])
    meta['mean_copy_time'] += time() - start
    meta.update(score=score)
    assert meta['iterations'] > 0
    meta['mean_score_time'] += time() - start
    meta['mean_test_score'] = score
    return meta


def _to_promote(result, completed_jobs, eta=None):
    bracket_models = [r for r in completed_jobs
                      if r['bracket_iter'] == result['bracket_iter'] and
                      r['bracket'] == result['bracket']]

    bracket_completed = len(bracket_models) == result['num_models']
    if bracket_completed:
        to_keep = len(bracket_models) // eta
        top = toolz.topk(to_keep, bracket_models, key='score')
        if to_keep <= 1:
            return []
        for job in top:
            job["num_models"] = to_keep
            job['partial_fit_calls'] *= eta
            job['bracket_iter'] += 1
        return top
    return []


def _create_model(model, params, random_state=42):
    model = clone(model).set_params(**params)
    if 'random_state' in model.get_params():
        model.set_params(random_state=random_state)
    return model, None  # right now no meta information


def _model_id(s, n_i):
    return "bracket={s}-{n_i}".format(s=s, n_i=n_i)


async def _hyperband(model, params, X, y, max_iter=None, eta=None,
                     fit_params={}, random_state=42, test_size=None,
                     scorer=None):
    client = default_client()
    rng = check_random_state(random_state)
    N, R, brackets = _get_hyperband_params(max_iter, eta=eta)
    #  params = iter(ParameterSampler(params, n_iter=sum(N), random_state=rng))
    params = [ParameterSampler(params, 1, random_state=rng.randint(sum(N)))
              for _ in range(sum(N))]
    params = iter([list(p)[0] for p in params])

    params = {_model_id(s, n_i): next(params)
              for s, n, r in zip(brackets, N, R) for n_i in range(n)}
    model_futures = {_model_id(s, n_i):
                     client.submit(_create_model, model,
                                   params[_model_id(s, n_i)],
                                   random_state=rng.randint(100 * sum(N)))
                     for s, n, r in zip(brackets, N, R) for n_i in range(n)}

    # lets assume everything in fit_params is small and make it concrete
    fit_params = await client.compute(fit_params)

    # TODO: pass in 0.15 as a keyword argument
    r = train_test_split(X, y, test_size=test_size, random_state=rng)
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

    info = {s: [{"partial_fit_calls": r, "bracket": "bracket={}".format(s),
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
                                        meta=meta, fit_params=fit_params)
                          for i, bracket_metas in enumerate(info.values())
                          for j, meta in enumerate(bracket_metas)}

    assert set(model_meta_futures.keys()) == set(model_futures.keys())

    score_futures = [client.submit(_score, model_meta_future, X_test, y_test,
                                   scorer=scorer)
                     for _id, model_meta_future in model_meta_futures.items()]

    completed_jobs = {}
    seq = as_completed(score_futures)
    history = []
    async for future in seq:
        result = await future
        history += [result]

        completed_jobs[result['model_id']] = result
        promoted = _to_promote(result, completed_jobs.values(), eta=eta)
        for job in promoted:
            i = rng.choice(len(X_train))

            # This block prevents communication of the model
            model_id = job['model_id']
            model_meta_future = model_meta_futures[model_id]
            model_meta_future = client.submit(_partial_fit, model_meta_future,
                                              X_train[i], y_train[i],
                                              meta=job, fit_params=fit_params)
            model_meta_futures[model_id] = model_meta_future

            score_future = client.submit(_score, model_meta_future, X_test,
                                         y_test, scorer=scorer)
            seq.add(score_future)

    assert completed_jobs.keys() == model_meta_futures.keys()
    return (params, model_meta_futures, history, list(completed_jobs.values()))


class HyperbandCV(DaskBaseSearchCV):
    """Find the best parameters for a particular model with cross-validation

    This algorithm is state-of-the-art and only requires computational budget
    as input. It will find close to the best possible parameters with the given
    computational budget.

    Parameters
    ----------
    model : object
        An object that has support for ``partial_fit``, ``get_params``,
        ``set_params`` and ``score``. This can be an instance of scikit-learn's
        BaseEstimator
    params : dict
        The various parameters to search over.
    max_iter : int, default=81
        The maximum number of partial_fit calls to any one model.
    eta : int, default=3
        How aggressive to be in model tuning. It is not recommended to change
        this value, and if changed we recommend ``eta=2`` or ``eta=4``. Higher
        values implies high confidence in model selection.
    random_state : int or np.random.RandomState
        A random state for this class. Setting this helps enforce determinism.
    scoring : str or callable
        The scoring method by which to score different classifiers.

    Examples
    --------
    >>> import numpy as np
    >>> from dask_ml.model_selection import HyperbandCV
    >>> from dask_ml.datasets import make_classification
    >>> from sklearn.linear_model import SGDClassifier
    >>>
    >>> X, y = make_classification(chunks=20)
    >>> est = SGDClassifier(tol=1e-3)
    >>> params = {'alpha': np.logspace(-4, 0, num=1000),
    >>>           'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
    >>>           'average': [True, False]}
    >>>
    >>> search = HyperbandCV(est, params)
    >>> search.fit(X, y, classes=np.unique(y))
    >>> search.best_params_
    ... {'loss': 'log', 'average': False, 'alpha': 0.0080502}

    Attributes
    ----------
    cv_results_ : dict of lists
        Information about the cross validation scores for each model.
        All lists are ordered the same, and this value can be imported into
        a pandas DataFrame. This dict has keys of

            * ``rank_test_score``
            * ``model_id``
            * ``mean_fit_time``
            * ``mean_score_time``
            * ``std_fit_time``
            * ``std_score_time``
            * ``mean_test_score``
            * ``std_test_score``
            * ``partial_fit_calls``
            * ``mean_train_score``
            * ``std_train_score``
            * ``params``
            * ``param_value``
            * ``mean_copy_time``

    meta_ : dict
        Information about every model that was trained. Can be used as input to
        ``fit_metadata``.
    history_ : list of dicts
        Information about every model after it is scored. Most models will be
        in here more than once because poor performing models are "killed" off
        early.
    best_params_ : dict
        The params that produced the best performing model
    best_estimator_ : any
        The best performing model
    best_index_ :
    n_splits_ : int
        The number of cross-validation splits.
    multimetric_ :
        Whether or whether this model selection algorithm is multimetric.
    test_size : float, default 0.15
        The fraction of test set that should be used for testing.

    Notes
    -----
    Hyperband is state of the art via an adaptive scheme: that only spends time
    on high-performing modelsi, because our goal is to find the highest
    performing model. This means that it stops training models that perform
    poorly.

    References
    ----------
    1. "Hyperband: A novel bandit-based approach to hyperparameter
       optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh,
       and A. Talwalkar.  https://arxiv.org/abs/1603.06560

    """
    def __init__(self, model, params, max_iter=81, eta=3, random_state=42,
                 scoring=None, test_size=0.15):
        self.model = model
        self.params = params
        self.max_iter = max_iter
        self.eta = eta
        self.random_state = random_state
        self.scoring = scoring
        self.test_size = test_size

        super(HyperbandCV, self).__init__(model, scoring=scoring)

    def fit(self, X, y, **fit_params):
        """Find the best parameters for a particular model

        Arguments
        ---------
        X, y : two dask.arrays or np.ndarrays
            Input to ``fit``. X.ndim == 2 and y.ndim == 1. If dask arrays are
            passed, each ``partial_fit`` call will be over one chunk of the
            array.
        fit_params : dict
            Arguments to pass to ``model.partial_fit``
        """
        return default_client().sync(self._fit, X, y, **fit_params)

    @gen.coroutine
    def _fit(self, X, y, **fit_params):
        if isinstance(X, np.ndarray):
            X = da.from_array(X, chunks=X.shape)
        if isinstance(y, np.ndarray):
            y = da.from_array(y, chunks=y.shape)

        # We always want a concrete scorer, so return_dask_score=False
        # We want this because we're always scoring NumPy arrays
        self.scorer_ = check_scoring(self.model, scoring=self.scoring)
        r = yield _hyperband(self.model, self.params, X, y,
                             max_iter=self.max_iter,
                             fit_params=fit_params, eta=self.eta,
                             random_state=self.random_state,
                             test_size=self.test_size,
                             scorer=self.scorer_)
        params, model_meta_futures, history, meta = r

        self.meta_ = meta
        self.history_ = history

        cv_res, best_idx = _get_cv_results(meta, params)
        best_id = cv_res['model_id'][best_idx]

        best_model_and_meta = yield model_meta_futures[best_id]
        self.best_estimator_ = best_model_and_meta[0]

        self.cv_results_ = cv_res
        self.best_index_ = best_idx
        self.n_splits_ = 1  # TODO: increase this! It's hard-coded right now
        self.multimetric_ = False

        return self

    def fit_metadata(self, meta=None):
        """Get information about how much computation is required for ``fit``.

        Arguments
        ---------
        meta : dict, optional.
            Information about the computation that occured, available through
            ``HyperbandCV.meta_``.

        Returns
        -------
        info : dict
            Information about the computation performed by ``fit``. Has
            information about the total number of ``partial_fit`` calls
            and the total number of models created, available under the keys
            ``total_partial_fit_calls`` and ``total_models``.

            There is also an additional key ``brackets`` that details the same
            information for each bracket, and the number of times
            ``partial_fit`` is called on the models in this bracket through
            the key ``iters``.
        """
        if meta is None:
            bracket_info = _hyperband_paper_alg(self.max_iter, eta=self.eta)
            total_models = sum(b['num_models'] for b in bracket_info)
        else:
            brackets = toolz.groupby("bracket", meta)
            bracket_info = [{'num_models': max(vi['num_models'] for vi in v),
                             'partial_fit_calls': sum(vi['partial_fit_calls']
                                                      for vi in v),
                             'bracket': k,
                             'iters': {vi['partial_fit_calls'] for vi in v}}
                            for k, v in brackets.items()]
            total_models = len(set([r['model_id'] for r in meta]))
        for bracket in bracket_info:
            bracket['iters'] = sorted(list(bracket['iters']))
        total_partial_fit = sum(b['partial_fit_calls'] for b in bracket_info)
        bracket_info = sorted(bracket_info, key=lambda x: x['bracket'])

        info = {'total_partial_fit_calls': total_partial_fit,
                'total_models': total_models,
                'brackets': bracket_info}
        return info


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


def _hyperband_paper_alg(R, eta=3):
    """
    Algorithm 1 from the Hyperband paper. Only a slight modification is made,
    the ``if to_keep <= 1``: if 1 model is left there's no sense in training
    any further.

    References
    ----------
    1. "Hyperband: A novel bandit-based approach to hyperparameter
       optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh,
       and A. Talwalkar.  https://arxiv.org/abs/1603.06560
    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R
    brackets = reversed(range(int(s_max + 1)))
    hists = {}
    for s in brackets:
        n = int(math.ceil(B / R * eta**s / (s + 1)))
        r = int(R * eta**-s)

        T = set(range(n))
        hist = {'num_models': n, 'models': {n: 0 for n in range(n)},
                'iters': []}
        for i in range(s + 1):
            n_i = math.floor(n * eta**-i)
            r_i = r * eta**i
            L = {model: r_i for model in T}
            hist['models'].update(L)
            hist['iters'] += [r_i]
            to_keep = math.floor(n_i / eta)
            T = {model for i, model in enumerate(T)
                 if i < to_keep}
            if to_keep <= 1:
                break

        hists['bracket={s}'.format(s=s)] = hist

    info = [{'bracket': k, 'num_models': hist['num_models'],
             'partial_fit_calls': sum(hist['models'].values()),
             'iters': set(hist['iters'])}
            for k, hist in hists.items()]
    return info
