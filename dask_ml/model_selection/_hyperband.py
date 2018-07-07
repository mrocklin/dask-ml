from __future__ import division

from collections import defaultdict
from copy import deepcopy
import functools
import logging
import math

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.utils import check_random_state
from sklearn.metrics.scorer import check_scoring
from tornado import gen
import toolz

import dask
import dask.array as da
from dask.distributed import as_completed, default_client, futures_of, Future
from distributed.utils import log_errors
from distributed.metrics import time

from ._split import train_test_split
from ._search import DaskBaseSearchCV


logger = logging.getLogger(__name__)


def _partial_fit(model_and_meta, X, y, fit_params):
    """
    Call partial_fit on a classifiers with training data X and y

    Arguments
    ---------
    model_and_meta : Tuple[Estimator, dict]
    X, y : np.ndarray, np.ndarray
        Training data
    fit_params : dict
        Extra keyword arguments to pass to partial_fit

    Returns
    -------
    model : Estimator
        The model that has been fit.
    meta : dict
        A new dictionary with updated information.
    """
    with log_errors(pdb=True):
        model, meta = model_and_meta

        model = deepcopy(model)
        model.partial_fit(X, y, **(fit_params or {}))

        meta = dict(meta)
        meta['time_step'] += 1

        return model, meta


def _score(model_and_meta, X, y, scorer):
    model, meta = model_and_meta
    score = scorer(model, X, y)

    meta = dict(meta)
    meta.update(score=score)
    return meta


def _create_model(model, ident, **params):
    """ Create a model by cloning and then setting params """
    with log_errors(pdb=True):
        model = clone(model).set_params(**params)
        return model, {'ident': ident, 'params': params, 'time_step': -1}


def inverse(start, batch):
    """ Decrease target number of models inversely with time """
    return int(start / (1 + batch))


@gen.coroutine
def _hyperband(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    start=1000,
    test_size=None,
    fit_params=None,
    random_state=None,
    scorer=None,
    target=inverse,
):
    original_model = model
    fit_params = fit_params or {}
    client = default_client()
    rng = check_random_state(random_state)
    param_iterator = iter(ParameterSampler(params, start, random_state=rng))
    target = functools.partial(target, start)

    info = {}
    models = {}
    scores = {}

    for ident in range(start):
        params = next(param_iterator)
        model = client.submit(_create_model, original_model, ident,
                              random_state=rng.randint(2**31), **params)
        info[ident] = {'params': params, 'param_index': ident}
        models[ident] = model

    # assume everything in fit_params is small and make it concrete
    fit_params = yield client.compute(fit_params)

    # Convert testing data into a single element on the cluster
    # This assumes that it fits into memory on a single worker
    if isinstance(X_test, da.Array):
        X_test = client.compute(X_test)
    else:
        y_test = yield client.scatter(y_test)
    if isinstance(y_test, da.Array):
        y_test = client.compute(y_test)
    else:
        y_test = yield client.scatter(y_test)

    # Convert to batches of delayed objects of numpy arrays
    X_train = X_train.to_delayed()
    if hasattr(X_train, 'squeeze'):
        X_train = X_train.squeeze()
    y_train = y_train.to_delayed()
    if hasattr(y_train, 'squeeze'):
        y_train = y_train.squeeze()
    X_train, y_train = dask.optimize(X_train.tolist(), y_train.tolist())

    # Create order by which we process batches
    # TODO: create a non-repetitive random and uniform ordering
    order = list(range(len(X_train)))
    rng.shuffle(order)
    seen = {}
    tokens = {}

    def get_futures(time_step):
        j = order[time_step % len(order)]

        if time_step < len(order) and j not in seen:  # new future, need to tell scheduler about it
            X_future = client.compute(X_train[j])
            y_future = client.compute(y_train[j])
            seen[j] = (X_future.key, y_future.key)

            # This is a hack to keep the futures in the scheduler but not in memory
            X_token = client.submit(len, X_future)
            y_token = client.submit(len, y_future)
            tokens[time_step] = (X_token, y_token)

            return X_future, y_future

        else:
            x_key, y_key = seen[j]
            return Future(x_key), Future(y_key)

    # Submit initial partial_fit and score computations on first batch of data
    X_future, y_future = get_futures(0)
    for ident, model in models.items():
        model = client.submit(_partial_fit, model, X_future, y_future, fit_params)
        score = client.submit(_score, model, X_test, y_test, scorer)
        models[ident] = model
        scores[ident] = score

    done = defaultdict(set)
    seq = as_completed(scores.values(), with_results=True)
    current_time_step = 0
    next_time_step = current_time_step + 1
    optimistic = set()  # set of fits that we might or might not want to keep
    history = []

    # async for future, result in seq:
    while not seq.is_empty():
        future, meta = yield seq.__anext__()
        if future.cancelled():
            continue
        time_step = meta['time_step']
        ident = meta['ident']

        done[time_step].add(ident)
        info[ident].update(meta)
        history.append(meta)

        # Evolve the model by a few time steps, then call score on the last one
        model = models[ident]
        for i in range(time_step, next_time_step):
            X_future, y_future = get_futures(i + 1)
            model = client.submit(_partial_fit, model, X_future, y_future,
                                  fit_params, priority=-i + meta['score'])
        score = client.submit(_score, model, X_test, y_test, scorer,
                              priority=-time_step + meta['score'])
        models[ident] = model
        scores[ident] = score
        optimistic.add(ident)  # we're not yet sure that we want to do this

        # We've now finished a full set of models
        # It's time to select the ones that get to survive and remove the rest
        if time_step == current_time_step and len(done[time_step]) >= len(models):

            # Step forward in time until we'll want to contract models again
            current_time_step = next_time_step
            next_time_step = current_time_step + 1
            while target(current_time_step) == target(next_time_step):
                next_time_step += 1

            # Select the best models by score
            good = set(toolz.topk(target(current_time_step), models, key=lambda i: info[i]['score']))
            bad = set(models) - good

            # Delete the futures of the other models.  This cancels optimistically submitted tasks
            for ident in bad:
                del models[ident]
                del scores[ident]

            # Add back into the as_completed iterator
            for ident in optimistic & good:
                seq.add(scores[ident])
            optimistic.clear()

            assert len(models) == target(current_time_step)

            if len(good) == 1:  # found the best one?  Break.
                break

    [best] = good
    model, meta = yield models[best]
    raise gen.Return((info[best], model, history))


class HyperbandCV(DaskBaseSearchCV):
    """Find the best parameters for a particular model with cross-validation

    This algorithm is state-of-the-art and only requires computational budget
    as input. It does not require a trade-off between "evaluate many
    parameters" and "train for a long time" like RandomizedSearchCV. Hyperband
    will find close to the best possible parameters with the given
    computational budget [1]_.*

    :sup:`* This will happen with high probability, and "close" means "within
    a log factor of the lower bound"`

    Parameters
    ----------
    model : object
        An object that has support for ``partial_fit``, ``get_params``,
        ``set_params`` and ``score``. This can be an instance of scikit-learn's
        BaseEstimator
    params : dict
        The various parameters to search over.
    max_iter : int, default=81
        The maximum number of partial_fit calls to any one model. This should
        be the number of ``partial_fit`` calls required for the model to
        converge.
    random_state : int or np.random.RandomState
        A random state for this class. Setting this helps enforce determinism.
    scoring : str or callable
        The scoring method by which to score different classifiers.
    test_size : float
        Hyperband uses one test set for all example, and this controls the
        size of that test set. It should be a floating point value between 0
        and 1 to represent the number of examples to put into the test set.

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
    {'loss': 'log', 'average': False, 'alpha': 0.0080502}

    Attributes
    ----------
    cv_results_ : dict of lists
        Information about the cross validation scores for each model.
        All lists are ordered the same, and this value can be imported into
        a pandas DataFrame. This dict has keys of

        * ``rank_test_score``
        * ``model_id``
        * ``mean_test_score``
        * ``std_test_score``
        * ``partial_fit_calls``
        * ``mean_train_score``
        * ``std_train_score``
        * ``params``
        * ``param_value``

    meta_ : dict
        Information about every model that was trained. Can be used as input to
        :func:`~dask_ml.model_selection.HyperbandCV.fit_metadata`.
    history_ : list of dicts
        Information about every model after it is scored. Most models will be
        in here more than once because poor performing models are "killed" off
        early.
    best_params_ : dict
        The params that produced the best performing model
    best_estimator_ : any
        The best performing model
    best_index_ : int
        The index of the best performing classifier to be used in
        ``cv_results_``.
    n_splits_ : int
        The number of cross-validation splits.
    best_score_ : float
        The best validation score on the test set.
    best_params_ : dict
        The params that are given to the model that achieves ``best_score_``.

    Notes
    -----
    Hyperband is state of the art via an adaptive scheme. Hyperband
    only spends time on high-performing models, because our goal is to find
    the highest performing model. This means that it stops training models
    that perform poorly.

    There are some limitations to the `current` implementation of Hyperband:

    1. The testing dataset must fit in the memory of a single worker
    2. HyperbandCV does not implement cross validation

    References
    ----------
    .. [1] "Hyperband: A novel bandit-based approach to hyperparameter
           optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A.
           Rostamizadeh, and A. Talwalkar.  https://arxiv.org/abs/1603.06560
    .. [2] "Massively Parallel Hyperparameter Tuning", 2018 by L. Li, K.
            Jamieson, A. Rostamizadeh, K. Gonina, M. Hardt, B. Recht, A.
            Talwalkar.  https://openreview.net/forum?id=S1Y7OOlRZ

    """

    def __init__(
        self,
        model,
        params,
        start=1000,
        random_state=None,
        scoring=None,
        test_size=0.15,
    ):
        self.model = model
        self.params = params
        self.start = start
        self.test_size = test_size
        self.random_state = random_state

        self.best_score = None
        self.best_params = None

        super(HyperbandCV, self).__init__(model, scoring=scoring)

    def fit(self, X, y, **fit_params):
        """Find the best parameters for a particular model

        Parameters
        ----------
        X, y : array-like
        **fit_params
            Additional partial fit keyword arguments for the estimator.
        """
        return default_client().sync(self._fit, X, y, **fit_params)

    @gen.coroutine
    def _fit(self, X, y, X_test=None, y_test=None, **fit_params):
        # We always want a concrete scorer, so return_dask_score=False
        # We want this because we're always scoring NumPy arrays
        self.scorer_ = check_scoring(self.model, scoring=self.scoring)
        self.best_score = -np.inf
        info, model, history = yield _hyperband(
            self.model,
            self.params,
            X,
            y,
            X_test=X_test,
            y_test=y_test,
            start=self.start,
            fit_params=fit_params,
            random_state=self.random_state,
            test_size=self.test_size,
            scorer=self.scorer_,
        )

        self.best_index_ = info['ident']
        self.best_estimator = model

        self.n_splits_ = 1  # TODO: increase this! It's hard-coded right now
        self.multimetric_ = False

        raise gen.Return(self)


def _get_cv_results(history, params):
    scores = [h["score"] for h in history]
    best_idx = int(np.argmax(scores))
    keys = set(toolz.merge(history).keys())
    for unused in [
        "bracket",
        "iterations",
        "num_models",
        "bracket_iter",
        "score",
    ]:
        keys.discard(unused)
    cv_results = {k: [h[k] for h in history] for k in keys}

    params = [params[model_id] for model_id in cv_results["model_id"]]
    cv_results["params"] = params
    params = {
        "param_" + k: [param[k] for param in params] for k in params[0].keys()
    }
    ranks = np.argsort(scores)[::-1]
    cv_results["rank_test_score"] = ranks.tolist()
    cv_results.update(params)
    return cv_results, best_idx
