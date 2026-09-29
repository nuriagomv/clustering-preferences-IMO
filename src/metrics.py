"""Scores for recovered preference vectors, and post-processing of clusters.

``analyze_instance`` compares one true weight vector with one estimate.
``regroup`` canonicalizes a solution before those scores are computed:
clusters are ordered, empty clusters are zeroed, and identical weight
vectors are merged.
"""

import math

import numpy as np
from scipy.stats import kendalltau, spearmanr, wasserstein_distance
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import cosine_similarity


def analyze_instance(w_true, w_estimated):
    """Compare a true preference vector with its estimate.

    Parameters
    ----------
    w_true, w_estimated : array-like, shape (K,)
        Weight vectors on the simplex (they are not re-normalized here).

    Returns
    -------
    equal_maxs : bool
        True when the estimated vector's largest weight sits on an objective
        that is also maximal in ``w_true``. Ties in the true vector count
        as a match.
    rmse : float
        Root mean squared error between the two vectors.
    rho : float
        Spearman rank correlation. Measures whether the priority order agrees.
    tau : float
        Kendall rank correlation. Same question, counted on pairwise orderings.
    cos_sim : float
        Cosine similarity. Closer to 1 means the two vectors are proportional.
    emd : float
        1D Wasserstein distance between the two weight vectors. Small values
        mean the mass on each objective is close.
    """
    w_true = np.asarray(w_true, dtype=float)
    w_estimated = np.asarray(w_estimated, dtype=float)

    # A match on the top priority, allowing several objectives tied for first.
    maximal_objectives = np.where(w_true == np.max(w_true))[0]
    equal_maxs = w_estimated.argmax() in maximal_objectives

    rmse = math.sqrt(mse(w_true, w_estimated))
    rho, _ = spearmanr(w_true, w_estimated)
    tau, _ = kendalltau(w_true, w_estimated)
    cos_sim = cosine_similarity(
        w_true.reshape(1, -1), w_estimated.reshape(1, -1)
    )[0, 0]
    emd = wasserstein_distance(w_true, w_estimated)
    return equal_maxs, rmse, rho, tau, cos_sim, emd


def summarize_recovery(weights_true, weights_estimated, ndigits=None):
    """Score a batch of preference estimates and average each score.

    Parameters
    ----------
    weights_true, weights_estimated : sequence of array-like
        Paired row by row. Both sequences must have the same length.
    ndigits : int, optional
        Decimal places for the means. ``None`` leaves the raw mean, which is
        what the unknown-objective driver stores.

    Returns
    -------
    per_instance : list of tuple
        One :func:`analyze_instance` result per row.
    means : list of float
        Column means, in the same order as :func:`analyze_instance`.
    """
    per_instance = [
        analyze_instance(weights_true[n], weights_estimated[n])
        for n in range(len(weights_estimated))
    ]
    columns = list(zip(*per_instance))
    means = []
    for column in columns:
        mean = np.mean(column)
        means.append(mean.round(ndigits) if ndigits is not None else mean)
    return per_instance, means


def regroup(W, X, u=None):
    """Order clusters and drop empty or duplicate preference vectors.

    Clusters are sorted by the scalar score ``s @ w``, with
    ``s = (1, 2, ..., K)``. That puts similar catalogues in a stable order
    so two runs that discover the same clusters are comparable.

    Then:

    * a cluster with no assigned instance has its weight row set to 0;
    * if two surviving rows are equal entrywise, their assignments are
      merged into the first row and the second row is zeroed.

    Parameters
    ----------
    W : ndarray, shape (L, K)
        Cluster weight vectors.
    X : ndarray, shape (N, L)
        Binary assignments of instances to clusters.
    u : ndarray, shape (G, L, m), optional
        Dual multipliers. Reordered with the clusters when provided.

    Returns
    -------
    W_re, X_re : ndarray
        Reordered weights and assignments, with dropped clusters zeroed.
    u_ordered : ndarray, only if ``u`` was given
        Dual multipliers in the same cluster order. Rows that were later
        merged are not deleted; only ``W`` and ``X`` are edited for merges.
    n_effective : int
        Number of clusters that still have at least one instance.
    dropped : list of int
        Indices (in the reordered indexing) of clusters that ended empty.
    """
    n_clusters, n_objectives = W.shape
    n_instances, _ = X.shape

    dropped = []
    # Score used only to sort. The values 1..K break ties deterministically.
    score_weights = np.arange(1, n_objectives + 1)
    order = np.argsort([score_weights @ w for w in W])

    W_ordered = np.zeros(W.shape)
    X_ordered = np.zeros(X.shape)
    if u is not None:
        u_ordered = np.zeros(u.shape)

    for new_index, old_index in enumerate(order):
        W_ordered[new_index, :] = W[old_index, :]
        if u is not None:
            u_ordered[:, new_index, :] = u[:, old_index, :]
        assigned = X[:, old_index] == 1
        X_ordered[assigned, new_index] = 1

    W_re = W_ordered.copy()
    X_re = X_ordered.copy()

    # Empty clusters do not represent a recovered preference.
    for cluster in range(n_clusters):
        if X_re[:, cluster].sum() == 0:
            W_re[cluster, :] = np.zeros(n_objectives)
            dropped.append(cluster)

    # Identical weight vectors describe the same preference: keep one cluster.
    for cluster in range(n_clusters):
        for other in range(n_clusters):
            if cluster == other:
                continue
            same_weights = all(
                math.isclose(left, right)
                for left, right in zip(W_re[cluster, :], W_re[other, :])
            )
            if same_weights:
                X_re[:, cluster] += X_re[:, other]
                X_re[:, other] = np.zeros(n_instances)
                W_re[other, :] = np.zeros(n_objectives)

    n_effective = int(np.sum(X_re.sum(axis=0) > 0))
    if u is None:
        return W_re, X_re, n_effective, dropped
    return W_re, X_re, u_ordered, n_effective, dropped
