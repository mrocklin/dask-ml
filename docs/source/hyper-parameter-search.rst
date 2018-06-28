Hyper Parameter Search
======================

.. autosummary::
   sklearn.pipeline.make_pipeline
   sklearn.model_selection.GridSearchCV
   dask_ml.model_selection.GridSearchCV
   sklearn.model_selection.RandomizedSearchCV
   dask_ml.model_selection.RandomizedSearchCV
   dask_ml.model_selection.HyperbandCV

Most estimators have a set of *hyper-parameters*.
These are parameters that are not learned during training but instead must be
set ahead of time. Traditionally we use Scikit-Learn tools like
:class:`sklearn.model_selection.GridSearchCV` and
:class:`sklearn.model_selection.RandomizedSearchCV` to tune our
hyper-parameters by searching over the space of hyper-parameters to find the
combination that gives the best performance on a cross-validation set.

Pipelines
---------

This search for hyper-parameters can become significantly more expensive when
we have not a single estimator, but many estimators arranged into a pipeline.
A :class:`sklearn.pipeline.Pipeline` makes it possible to define the entire modeling
process, from raw data to fit estimator, in a single python object. You can
create a pipeline with :func:`sklearn.pipeline.make_pipeline`.

.. ipython:: python

   from sklearn.pipeline import make_pipeline
   from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
   from sklearn.linear_model import SGDClassifier

   pipeline = make_pipeline(CountVectorizer(),
                            TfidfTransformer(),
                            SGDClassifier())
   pipeline

Pipelines work by calling the usual ``fit`` and ``transform`` methods in succession.
The result of the prior ``transform`` is passed into the next ``fit`` step.
We'll see an example in the next section.

Efficient Search
----------------

Computation
^^^^^^^^^^^

We implement a state-of-the-art algorithm to choose hyperparameters in
:class:`dask_ml.model_selection.HyperbandCV` [1]_. The goal of hyperparameter
selection is to find the best or highest-scoring set of hyperparameters for a
particular model. If we want to achieve that goal with as little computation as
possible, it makes sense to spend time on high-performing models and not waste
computation on low performing models.

Hyperband only requires `one` input, some computational budget. Notably, it
does not require a tradeoff between "train many parameters for a short time" or
"train few parameters for a long time" like
:class:`dask_ml.model_selection.RandomizedSearchCV`.
With this input, Hyperband has guarantees on finding close to the best set of
parameters possible given this computational input.*

:class:`dask_ml.model_selection.HyperbandCV` also implements the asynchronous
variant of Hyperband [2]_, which is well suited for the very parallel
architectures Dask enables. The goal of this variant is to find the best set of
parameters in the shortest time as possible, not as little computation as
possible. It does this by not waiting for `every` model to finish before
deciding to perform more computation on particular models.

.. [1] "Hyperband: A novel bandit-based approach to hyperparameter
       optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A.
       Rostamizadeh, and A. Talwalkar.  https://arxiv.org/abs/1603.06560
.. [2] "Massively Parallel Hyperparameter Tuning", 2018 by L. Li, K.
        Jamieson, A. Rostamizadeh, K. Gonina, M. Hardt, B. Recht, A.
        Talwalkar.  https://openreview.net/forum?id=S1Y7OOlRZ

:sup:`* This will happen with high probability, and "close" means "within a log factor of the lower bound"`

Caching
^^^^^^^

However now each of our estimators in our pipeline have hyper-parameters,
both expanding the space over which we want to search as well as adding
hierarchy to the search process.  For every parameter we try in the first stage
in the pipeline we want to try several in the second, and several more in the
third, and so on.

The common combination of pipelines and hyper-parameter search provide an
opportunity for dask to speed up model training not just by simple parallelism,
but also by searching the space in a more structured way.

If you use the drop-in replacements
:class:`dask_ml.model_selection.GridSearchCV` and
:class:`dask_ml.model_selection.RandomizedSearchCV` to fit a ``Pipeline``, you can improve
the training time since Dask will cache and reuse the intermediate steps.

.. ipython:: python

   # from sklearn.model_selection import GridSearchCV  # replace import
   from dask_ml.model_selection import GridSearchCV
   param_grid = {
       'tfidftransformer__norm': ['l1', 'l2', None],
       'sgdclassifier__loss': ['hing', 'log'],
       'sgdclassifier__alpha': [1e-5, 1e-3, 1e-1],
   }

   clf = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)

With the regular scikit-learn version, each stage of the pipeline must be fit
for each of the combinations of the parameters, even if that step isn't being
searched over. For example, the ``CountVectorizer`` must be fit 3 * 2 * 2 = 12
times, even though it's identical each time.

See :ref:`examples/hyperparameter-search.ipynb` for an example.

.. _dask-searchcv: http://dask-searchcv.readthedocs.io/en/latest/
