Hyper Parameter Search
======================

*Parallel tools for hyperparameter optimization of Scikit-Learn API-compatible
models.*

For parallel hyperparameter optimization you have a few options:

.. autosummary::
   sklearn.model_selection.GridSearchCV
   sklearn.model_selection.RandomizedSearchCV

.. autosummary::
   dask_ml.model_selection.GridSearchCV
   dask_ml.model_selection.RandomizedSearchCV
   dask_ml.model_selection.HyperbandCV

The scikit-learn implementations can already run parallel on a single machine
using the ``njobs=`` parameter.  Additionally you can scale them using the
:doc:`Dask-Joblib connection <joblib>`

The Dask-ML variants implement many (but not all) of the same parameters,
and should be a drop-in replacement for the subset that they do implement, but
are more efficient in a few ways:.

- :ref:`Flexible Backends <flexible-backends>`: Hyperparameter
  optimization can be done in parallel using threads, processes, or distributed
  across a cluster.  Though this is also the same with the Scikit-Learn
  versions with the Dask-joblib connection.

- :ref:`Work well with Dask collections <works-with-dask-collections>`. Dask
  arrays, dataframes, and delayed can be passed to the ``fit`` method.

- :ref:`Adaptive algorithms <hyperband>` like Hyperband, that learn during the
  computation and spend more time training models that seem to perform better
  over time.

- :ref:`Avoid repeated work <avoid-repeated-work>`. Candidate estimators with
  identical parameters and inputs will only be fit once. For
  composite-estimators such as ``Pipeline`` this can be significantly more
  efficient as it can avoid expensive repeated computations.

Both scikit-learn's and Dask-ML's model selection meta-estimators can be used
with Dask's :ref:`joblib backend <joblib>`.

.. _flexible-backends:

Flexible Backends
^^^^^^^^^^^^^^^^^

Dask-searchcv can use any of the dask schedulers. By default the threaded
scheduler is used, but this can easily be swapped out for the multiprocessing
or distributed scheduler:

.. code-block:: python

    # Distribute grid-search across a cluster
    from dask.distributed import Client
    scheduler_address = '127.0.0.1:8786'
    client = Client(scheduler_address)

    search.fit(digits.data, digits.target)


.. _works-with-dask-collections:

Works Well With Dask Collections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dask collections such as ``dask.array``, ``dask.dataframe`` and
``dask.delayed`` can be passed to ``fit``. This means you can use dask to do
your data loading and preprocessing as well, allowing for a clean workflow.
This also allows you to work with remote data on a cluster without ever having
to pull it locally to your computer:

.. code-block:: python

    import dask.dataframe as dd

    # Load data from s3
    df = dd.read_csv('s3://bucket-name/my-data-*.csv')

    # Do some preprocessing steps
    df['x2'] = df.x - df.x.mean()
    # ...

    # Pass to fit without ever leaving the cluster
    search.fit(df[['x', 'x2']], df['y'])

.. _hyperband:

Adaptive algorithms
^^^^^^^^^^^^^^^^^^^

Adaptive algorithms learn during the hyperparameter selection process and focus
computation where it matters most.

Dask-ML implements one adaptive algorithm, Hyperband [1]_ [2]_, which is useful
in the incremental learning case where we train models over batches of data.
This requires that your models support the ``.partial_fit(...)`` method.
It works by trying many possible parameters on the first few chunks of data,
and then only continues training for those parameters that perform well.

.. autosummary:: dask_ml.model_selection.HyperbandCV

.. [1] "Hyperband: A novel bandit-based approach to hyperparameter
       optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A.
       Rostamizadeh, and A. Talwalkar.  https://arxiv.org/abs/1603.06560
.. [2] "Massively Parallel Hyperparameter Tuning", 2018 by L. Li, K.
        Jamieson, A. Rostamizadeh, K. Gonina, M. Hardt, B. Recht, A.
        Talwalkar.  https://openreview.net/forum?id=S1Y7OOlRZ

.. _avoid-repeated-work:

Avoid Repeated Work
^^^^^^^^^^^^^^^^^^^

However now each of our estimators in our pipeline have hyper-parameters,
both expanding the space over which we want to search as well as adding
hierarchy to the search process.  For every parameter we try in the first stage
in the pipeline we want to try several in the second, and several more in the
third, and so on.

When searching over composite estimators like ``sklearn.pipeline.Pipeline`` or
``sklearn.pipeline.FeatureUnion``, Dask-ML will avoid fitting the same
estimator + parameter + data combination more than once. For pipelines with
expensive early steps this can be faster, as repeated work is avoided.

For example, given the following 3-stage pipeline and grid (modified from `this
scikit-learn example
<http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html>`__).

.. code-block:: python

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier())])

    grid = {'vect__ngram_range': [(1, 1)],
            'tfidf__norm': ['l1', 'l2'],
            'clf__alpha': [1e-3, 1e-4, 1e-5]}

the Scikit-Learn grid-search implementation looks something like (simplified):

.. code-block:: python

	scores = []
	for ngram_range in parameters['vect__ngram_range']:
		for norm in parameters['tfidf__norm']:
			for alpha in parameters['clf__alpha']:
				vect = CountVectorizer(ngram_range=ngram_range)
				X2 = vect.fit_transform(X, y)
				tfidf = TfidfTransformer(norm=norm)
				X3 = tfidf.fit_transform(X2, y)
				clf = SGDClassifier(alpha=alpha)
				clf.fit(X3, y)
				scores.append(clf.score(X3, y))
	best = choose_best_parameters(scores, parameters)


As a directed acyclic graph, this might look like:

.. figure:: images/unmerged_grid_search_graph.svg
   :alt: "scikit-learn grid-search directed acyclic graph"
   :align: center


In contrast, the dask version looks more like:

.. code-block:: python

	scores = []
	for ngram_range in parameters['vect__ngram_range']:
		vect = CountVectorizer(ngram_range=ngram_range)
		X2 = vect.fit_transform(X, y)
		for norm in parameters['tfidf__norm']:
			tfidf = TfidfTransformer(norm=norm)
			X3 = tfidf.fit_transform(X2, y)
			for alpha in parameters['clf__alpha']:
				clf = SGDClassifier(alpha=alpha)
				clf.fit(X3, y)
				scores.append(clf.score(X3, y))
	best = choose_best_parameters(scores, parameters)


With a corresponding directed acyclic graph:

.. figure:: images/merged_grid_search_graph.svg
   :alt: "Dask-ML grid-search directed acyclic graph"
   :align: center


Looking closely, you can see that the Scikit-Learn version ends up fitting
earlier steps in the pipeline multiple times with the same parameters and data.
Due to the increased flexibility of Dask over Joblib, we're able to merge these
tasks in the graph and only perform the fit step once for any
parameter/data/estimator combination. For pipelines that have relatively
expensive early steps, this can be a big win when performing a grid search.

Pipelines
---------

Dask-ML uses scikit-learn's :class:`sklearn.pipeline.Pipeline` to express
pipelines of estimators that are chained together. If the individual
estimators work well with Dask's collections, the pipeline will as well.
