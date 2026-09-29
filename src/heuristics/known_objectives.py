"""Alternating heuristic for the known-objective model.

The exact model chooses assignments and preferences jointly. This heuristic
alternates two easier problems:

1. With assignments held fixed, optimize each cluster's preference (and the
   dual multipliers of its feasible regions) on its own.
2. With preferences held fixed, reassign every instance to the cluster with
   the smallest duality gap.

The loop stops when the objective repeats the previous iteration, or after
``n_iters_heuristic`` passes. The best iteration is the warm start of the
exact solve.
"""

import random

import gurobipy as gp
import numpy as np
from sklearn.cluster import KMeans


def initialize_instance_weights(decision, A, rhs, C, timelimit, silence):
    """Inverse problem for a single instance: the preference that best fits ``decision``.

    Solves ``min_w  u (A z - b)`` over ``w`` on the simplex and dual multipliers
    ``u <= 0`` with ``u A = w C``. The optimal ``w`` is the starting preference
    of this instance, before any clustering.

    Returns ``(model, w)`` with ``w`` an array of length K, or ``(model, None)``
    if the solve does not finish optimally.
    """
    n_objectives, _ = C.shape
    n_constraints, _ = A.shape

    model = gp.Model()
    if silence:
        model.setParam("OutputFlag", 0)
    if timelimit is not None:
        model.setParam(gp.GRB.Param.TimeLimit, timelimit)

    dual = model.addMVar(
        shape=n_constraints,
        lb=-float("inf"),
        ub=0.0,
        vtype=gp.GRB.CONTINUOUS,
        name="u",
    )
    weights = model.addMVar(
        shape=n_objectives, lb=0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="W"
    )
    model.addConstr(
        gp.quicksum(weights[k] for k in range(n_objectives)) == 1,
        name="weights' sum is one",
    )
    model.addConstr(dual @ A == weights @ C, name="dual_constr")
    model.setObjective(dual @ (A @ decision - rhs), sense=gp.GRB.MINIMIZE)
    model.update()
    model.optimize()

    if model.status == gp.GRB.Status.TIME_LIMIT:
        print("NOT ENOUGH TIME LIMIT")
    if model.status == gp.GRB.OPTIMAL:
        return model, weights.X
    print("unable to find initialization for w_n")
    return model, None


def kmeans_assignments(n_clusters, preferences, seed):
    """Hard assignments from k-means++ on the per-instance preferences.

    Returns a binary matrix of shape ``(N, n_clusters)``.
    """
    n_instances, _ = preferences.shape
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init="auto",
        init="k-means++",
    ).fit(preferences)

    assignments = np.zeros((n_instances, n_clusters))
    for instance, label in enumerate(kmeans.labels_):
        assignments[instance, label] = 1
    return assignments


def optimize_weights_given_assignments(
    assigned, C, N, dataset, A, list_bn, groups_N, considered_groups, timelimit, silence
):
    """Best preference for one cluster, given which instances belong to it.

    ``assigned[n]`` is 1 when instance ``n`` is in the cluster. Instances
    outside the cluster contribute nothing to the objective. The same
    stationarity constraint as the full model is imposed, once per feasible
    region.

    Returns ``(model, w, u)`` on an optimal solve. On any other status returns
    ``(model, None)``.
    """
    n_objectives, _ = C.shape
    n_regions = len(considered_groups)

    model = gp.Model()
    if silence:
        model.setParam("OutputFlag", 0)
    if timelimit is not None:
        model.setParam(gp.GRB.Param.TimeLimit, timelimit)

    dual = model.addMVar(
        shape=(n_regions, A.shape[0]),
        lb=-float("inf"),
        ub=0.0,
        vtype=gp.GRB.CONTINUOUS,
        name="u",
    )
    weights = model.addMVar(
        shape=n_objectives, lb=0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="W"
    )
    model.addConstr(
        gp.quicksum(weights[k] for k in range(n_objectives)) == 1,
        name="weights' sum is one",
    )
    model.addConstrs(
        (dual[region, :] @ A == weights @ C for region in range(n_regions)),
        name="dual_constr",
    )

    def _gap(instance):
        region = considered_groups.index(groups_N[instance])
        residual = A @ dataset[instance, :] - list_bn[instance]
        return assigned[instance] * dual[region, :] @ residual

    model.setObjective(
        gp.quicksum(_gap(n) for n in range(N)), sense=gp.GRB.MINIMIZE
    )
    model.update()
    model.optimize()

    if model.status == gp.GRB.Status.TIME_LIMIT:
        print("NOT ENOUGH TIME LIMIT")
    if model.status == gp.GRB.OPTIMAL:
        return model, weights.X, dual.X
    print("could not optimize for w_l")
    return model, None


def cluster_assignments(
    u, dataset, A, list_bn, groups_N, considered_groups, delete_cluster=None
):
    """Assign each instance to the cluster with the smallest duality gap.

    Parameters
    ----------
    u : ndarray, shape (G, L, m)
        Dual multipliers of every (feasible region, cluster) pair.
    delete_cluster : sequence of int, optional
        Clusters that must not receive instances. They are given an infinite
        gap, which is how empty clusters are kept empty at test time.

    Returns
    -------
    assignments : ndarray, shape (N, L)
        Binary matrix. Each row has a single 1.
    total_gap : float
        Sum of the chosen gaps. This is the heuristic objective.
    """
    if delete_cluster is None:
        delete_cluster = []

    n_instances = dataset.shape[0]
    n_clusters = u.shape[1]
    assignments = np.zeros((n_instances, n_clusters))
    total_gap = 0.0
    blocked = set(delete_cluster)
    for instance in range(n_instances):
        region = considered_groups.index(groups_N[instance])
        residual = A @ dataset[instance, :] - list_bn[instance]
        gaps = [
            u[region, cluster, :] @ residual
            if cluster not in blocked
            else float("inf")
            for cluster in range(n_clusters)
        ]
        total_gap += np.min(gaps)
        assignments[instance, int(np.argmin(gaps))] = 1
    return assignments, total_gap


def heuristic(
    seed,
    C,
    N,
    L,
    dataset_train,
    A,
    list_bn_train,
    groups_N_train,
    considered_groups,
    init_type="optimal",
    n_iters_heuristic=50,
    timelimit=60.0,
    silence=True,
):
    """Run the alternating heuristic and return every iteration.

    Parameters
    ----------
    init_type : {"optimal", "random"}
        ``optimal`` solves a per-instance inverse problem and clusters those
        preferences with k-means. ``random`` draws a cluster index uniformly
        for each instance.
    n_iters_heuristic : int
        Maximum number of alternation passes. The loop also stops early when
        the objective equals the previous pass.
    timelimit : float
        Time limit, in seconds, of each small LP inside one pass.

    Returns
    -------
    history : list of dict
        ``history[0]`` is the initialization (no weights yet). Later entries
        have keys ``W``, ``u``, ``X`` and ``obj_val``.
    best_obj_val : float
        Smallest objective seen.
    best_iter : int
        Index in ``history`` of that objective. The initialization occupies
        index 0, so the first real pass is index 1.
    """
    n_objectives, _ = C.shape
    np.random.seed(seed)
    random.seed(seed)

    print("HEURISTIC INITIALIZATION: ", init_type)
    if init_type == "optimal":
        preferences = np.array([
            initialize_instance_weights(
                dataset_train[n, :],
                A,
                list_bn_train[n],
                C,
                timelimit,
                silence,
            )[1]
            for n in range(N)
        ])
        fixed_assignments = kmeans_assignments(L, preferences, seed)
    if init_type == "random":
        fixed_assignments = np.zeros((N, L))
        for instance in range(N):
            fixed_assignments[instance, random.randint(0, L - 1)] = 1
    print("X_init composition = \n", fixed_assignments.sum(axis=0))

    print("TWO STAGE HEURISTIC: \n")
    # Sentinel so the first pass has a previous objective to compare against.
    history = [{"W": None, "u": None, "X": fixed_assignments, "obj_val": float("inf")}]
    best_obj_val, best_iter = float("inf"), -float("inf")
    iteration, improving = 0, True
    while (iteration <= n_iters_heuristic) and improving:
        iteration += 1
        print("ITERATION ", iteration)

        # Stage 1: one LP per cluster, assignments held fixed.
        new_W = np.zeros((L, n_objectives))
        new_u = np.zeros((len(considered_groups), L, A.shape[0]))
        for cluster in range(L):
            _, weights, dual = optimize_weights_given_assignments(
                fixed_assignments[:, cluster],
                C,
                N,
                dataset_train,
                A,
                list_bn_train,
                groups_N_train,
                considered_groups,
                timelimit,
                silence,
            )
            new_W[cluster, :] = weights
            new_u[:, cluster, :] = dual

        # Stage 2: reassign every instance given the new preferences.
        new_X, obj_val = cluster_assignments(
            new_u,
            dataset_train,
            A,
            list_bn_train,
            groups_N_train,
            considered_groups,
        )
        print("First step:")
        print("new W: \n", new_W.round(2))
        print("Second step:")
        print("new X: \n", new_X.sum(axis=0))
        print("obj val: ", obj_val)

        # Stop when this pass reproduces the previous objective exactly.
        # A strict improvement updates the incumbent; a worse pass is kept
        # in the history but does not move ``best_iter``.
        if obj_val == history[-1]["obj_val"]:
            print("EARLY STOP: solution not improved")
            improving = False
        if obj_val < best_obj_val:
            best_obj_val, best_iter = obj_val, iteration

        history.append({"W": new_W, "u": new_u, "X": new_X, "obj_val": obj_val})
        fixed_assignments = new_X

    return history, best_obj_val, best_iter
