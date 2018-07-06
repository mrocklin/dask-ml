from __future__ import division

from collections import defaultdict
from copy import deepcopy
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
from dask.distributed import as_completed, default_client, futures_of
from distributed.utils import log_errors
from distributed.metrics import time

from ._split import train_test_split
from ._search import DaskBaseSearchCV


logger = logging.getLogger(__name__)


def _partial_fit(model_and_meta, X, y, fit_params):
    """
    Call partial_fit on a classifiers with X and y

    Arguments
    ---------
    model_and_meta : tuple, (model: any, meta: dict)
        model needs to support partial_fit. meta is assumed to have keys
        ``iterations, partial_fit_calls``.
        partial_fit will be called on the model until
        ``meta['iterations'] >= meta['partial_fit_calls']``
    X, y : np.ndarray, np.ndarray
        Training data
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
    with log_errors(pdb=True):
        model, meta = model_and_meta

        model = deepcopy(model)
        try:
            model.partial_fit(X, y, **(fit_params or {}))
        except Exception:
            import pdb; pdb.set_trace()

        meta = dict(meta)
        meta['epoch'] += 1

        return model, meta


def _score(model_and_meta, X, y, scorer):
    model, meta = model_and_meta
    score = scorer(model, X, y)

    meta = deepcopy(meta)
    meta.update(score=score)
    meta["mean_test_score"] = score
    return meta


def _create_model(model, params, ident, random_state=42):
    with log_errors(pdb=True):
        model = clone(model).set_params(**params)
        if "random_state" in model.get_params():
            model.set_params(random_state=random_state)
        return model, {'ident': ident, 'params': params, 'epoch': -1}


@gen.coroutine
def _hyperband(
    original_model,
    params,
    X,
    y,
    X_test,
    y_test,
    start=1000,
    eta=1.5,
    test_size=None,
    fit_params=None,
    random_state=None,
    scorer=None,
):
    fit_params = fit_params or {}
    client = default_client()
    rng = check_random_state(random_state)
    param_iterator = iter(ParameterSampler(params, 1000000, random_state=rng))

    info = {}
    models = {}
    scores = {}

    for ident in range(start):
        params = next(param_iterator)
        model = client.submit(_create_model, original_model, params, ident,
                              random_state=rng.randint(2**31))
        info[ident] = {'params': params, 'param_index': ident}
        models[ident] = model

    # lets assume everything in fit_params is small and make it concrete
    fit_params = yield client.compute(fit_params)

    # convert testing data into a single element on the cluster
    if isinstance(X_test, da.Array):
        X_test = client.compute(X_test)
    else:
        y_test = yield client.scatter(y_test)
    if isinstance(y_test, da.Array):
        y_test = client.compute(y_test)
    else:
        y_test = yield client.scatter(y_test)

    X = X.to_delayed()
    if hasattr(X, 'squeeze'):
        X = X.squeeze()
    y = y.to_delayed()
    if hasattr(y, 'squeeze'):
        y = y.squeeze()

    X, y = dask.optimize(X.tolist(), y.tolist())

    # create order by which we process batches
    order = list(range(len(X)))
    rng.shuffle(order)

    X_futures = {}
    y_futures = {}

    j = order[0]
    X_future = X_futures[order[0]] = client.compute(X[order[0]])
    y_future = y_futures[order[0]] = client.compute(y[order[0]])

    for ident, model in models.items():
        model = client.submit(_partial_fit, model, X_future, y_future, fit_params)
        score = client.submit(_score, model, X_test, y_test, scorer)
        models[ident] = model
        scores[ident] = score

    done = defaultdict(set)
    seq = as_completed(scores.values(), with_results=True)
    current_epoch = 0
    target = start / (1 + current_epoch)
    optimistic = set()

    while not seq.is_empty():  # async for future, result in seq:
        future, meta = yield seq.__anext__()
        if future.cancelled():
            continue
        epoch = meta['epoch']
        ident = meta['ident']
        done[epoch].add(ident)
        info[ident].update(meta)

        # submit next epoch
        try:
            model = models[ident]
        except KeyError:
            import pdb; pdb.set_trace()
        j = order[(epoch + 1) % len(order)]
        if j not in X_futures:
            X_futures[j] = client.compute(X[j])
            y_futures[j] = client.compute(y[j])
        X_future = X_futures[j]
        y_future = y_futures[j]

        model = client.submit(_partial_fit, model, X_future, y_future,
                              fit_params, priority=-epoch + meta['score'])
        score = client.submit(_score, model, X_test, y_test, scorer,
                priority=-epoch + meta['score'])
        models[ident] = model
        scores[ident] = score
        optimistic.add(ident)

        # finished entire bracket of models
        if epoch == current_epoch and len(done[epoch]) >= len(models):
            del X_futures[order[epoch % len(order)]]
            del y_futures[order[epoch % len(order)]]
            current_epoch += 1
            target = max(1, int(start / (1 + current_epoch)))
            print(current_epoch, target)

            good = set(toolz.topk(target, models, key=lambda i: info[i]['score']))
            bad = set(models) - good
            if len(set(models) - bad) != target:
                import pdb; pdb.set_trace()
            for ident in bad:
                del models[ident]
                del scores[ident]

            for ident in optimistic & good:
                seq.add(scores[ident])

            # if current_epoch % len(order) == 0:
            #     rng.shuffle(order)

            optimistic.clear()

            assert len(models) == target

            if len(good) == 1:
                break

    [ident] = good
    model, meta = yield models[ident]
    raise gen.Return((info[ident], model))


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
    batch_size : float, default=0.2
        The fraction of the dataset that one partial_fit call will see. At
        most, ``batch_size * max_iter`` samples will be seen by
        ``partial_fit``.
    eta : int, default=3
        How aggressive to be in model tuning. It is not recommended to change
        this value, and if changed we recommend ``eta=4``.
        The theory behind Hyperband suggests ``eta=np.e``. Higher
        values imply higher confidence in model selection.
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

    At least one model sees ``max_iter * batch_size`` samples, which
    should be large enough for convergence (or close to it). Often
    there are hard constraints on ``max_iter * batch_size``
    (e.g., time or deadline constraints).

    ``max_iter`` is (almost) proportional to the number of parameters
    evaluated, as well as being the maximum number of times
    ``partial_fit`` is called for any model.
    ``batch_size`` is the fraction of the dataset that each ``partial_fit``
    call sees.

    ``max_iter`` should be set to be reasonable given the problem and
    parameter search space, but ideally
    large enough so that early-stopping is beneficial. Higher values will
    evaluate more parameters. We recommend setting ``max_iter * batch_size``,
    then using :func:`~dask_ml.model_selection.HyperbandCV.fit_metadata`
    alongside natural constraints (time or deadline constraints) to determine
    ``max_iter`` and ``batch_size``.

    The authors of Hyperband use ``max_iter=300`` and ``batch_size=0.25``
    to tune deep learning models in [1]_. They tune 6 stochastic gradient
    descent parameters alongside two problem formulation hyperparameters.
    For :func:`~sklearn.linear_model.SGDClassifier`, examples of "stochastic
    gradient descent parameters" are ``learning_rate`` and ``eta0``.
    Examples of "problem formulation hyperparameters" are ``alpha`` and
    ``l1_ratio`` when ``penalty='elasticnet'``.

    There are some limitations to the `current` implementation of Hyperband:

    1. The full dataset is requested to be in distributed memory
    2. The testing dataset must fit in the memory of a single worker
    3. HyperbandCV does not implement cross validation

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
        eta=2,
        random_state=None,
        scoring=None,
        test_size=0.15,
    ):
        self.model = model
        self.params = params
        self.start = start
        self.eta = eta
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
        info, model = yield _hyperband(
            self.model,
            self.params,
            X,
            y,
            X_test=X_test,
            y_test=y_test,
            start=self.start,
            eta=self.eta,
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
        n = int(math.ceil(B / R * eta ** s / (s + 1)))
        r = int(R * eta ** -s)

        T = set(range(n))
        hist = {
            "num_models": n,
            "models": {n: 0 for n in range(n)},
            "iters": [],
        }
        for i in range(s + 1):
            n_i = math.floor(n * eta ** -i)
            r_i = r * eta ** i
            L = {model: r_i for model in T}
            hist["models"].update(L)
            hist["iters"] += [r_i]
            to_keep = math.floor(n_i / eta)
            T = {model for i, model in enumerate(T) if i < to_keep}
            if to_keep <= 1:
                break

        hists["bracket={s}".format(s=s)] = hist

    info = [
        {
            "bracket": k,
            "num_models": hist["num_models"],
            "num_partial_fit_calls": sum(hist["models"].values()),
            "iters": set(hist["iters"]),
        }
        for k, hist in hists.items()
    ]
    return info
