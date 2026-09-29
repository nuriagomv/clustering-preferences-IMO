"""Alternating heuristic for the unknown-objective model.

Preferences stay on the catalogue. Each start picks ``L`` catalogue rows and
alternates:

1. Assignments fixed: estimate the cost matrix ``C`` (and the duals).
2. Costs fixed: re-solve the assignment of instances to catalogue rows.

Both stages are the full formulation with the other block frozen, so each
pass is a MIP restricted to one group of variables. With ``multistart=True``
every combination of ``L`` catalogue rows is tried and the best objective is
kept. With ``multistart=False`` a single combination is drawn at random.
"""

import os
import random
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src.models.unknown_objectives import problem_fUNknown


def random_assignments(chosen_clusters, n_instances, n_catalogue):
    """Assign every instance uniformly to one of ``chosen_clusters``.

    Columns outside ``chosen_clusters`` stay at zero, so the first pass only
    estimates costs for the selected catalogue rows. Returns a binary matrix
    of shape ``(n_instances, n_catalogue)``.
    """
    assignments = np.zeros((n_instances, n_catalogue))
    for instance in range(n_instances):
        cluster = random.sample(chosen_clusters, 1)[0]
        assignments[instance, cluster] = 1
    print("X_init composition = \n", assignments.sum(axis=0))
    return assignments


def _alternating_passes(task):
    """One multistart trajectory. ``task`` is the tuple built in :func:`heuristic`."""
    (
        chosen_clusters,
        silence,
        n_instances,
        n_catalogue,
        n_iters_heuristic,
        predef_W,
        n_active,
        dataset_train,
        A,
        list_bn_train,
        groups_N_train,
        considered_groups,
        timelimit,
        known_objs,
        lambd,
    ) = task

    fixed_assignments = random_assignments(chosen_clusters, n_instances, n_catalogue)
    print("TWO STAGE HEURISTIC: \n")
    history = [{
        "C": None,
        "c0": None,
        "u": None,
        "X": fixed_assignments,
        "obj_val": float("inf"),
    }]
    best_obj_val, best_iter = float("inf"), -float("inf")
    iteration, improving = 0, True
    while (iteration <= n_iters_heuristic) and improving:
        iteration += 1
        print("ITERATION ", iteration)

        # Stage 1: costs and duals, assignments frozen.
        _, costs, intercept, _, dual, _, _ = problem_fUNknown(
            predef_W,
            n_instances,
            n_active,
            dataset_train,
            A,
            list_bn_train,
            groups_N_train,
            considered_groups,
            timelimit=timelimit,
            silence=silence,
            predef_X=fixed_assignments,
            known_objs=known_objs,
            lambd=lambd,
        )
        new_C, new_c0, new_u = costs.X, intercept.X, dual.X

        # Stage 2: assignments, costs and duals frozen.
        assignment_model, _, _, new_X, _, _, _ = problem_fUNknown(
            predef_W,
            n_instances,
            n_active,
            dataset_train,
            A,
            list_bn_train,
            groups_N_train,
            considered_groups,
            timelimit=timelimit,
            silence=silence,
            predef_u=new_u,
            predef_C=(new_C, new_c0),
            known_objs=known_objs,
            lambd=lambd,
        )
        new_X = new_X.X
        obj_val = assignment_model.ObjVal

        print("First step:")
        print("new C: \n", new_C.round(2))
        print("Second step:")
        print("new X: \n", new_X.sum(axis=0))
        print("obj val: ", obj_val)

        if obj_val == history[-1]["obj_val"]:
            print("EARLY STOP: solution not improved")
            improving = False
        if obj_val < best_obj_val:
            best_obj_val, best_iter = obj_val, iteration

        history.append({
            "C": new_C,
            "c0": new_c0,
            "u": new_u,
            "X": new_X,
            "obj_val": obj_val,
        })
        fixed_assignments = new_X

    return history, best_obj_val, best_iter


def heuristic(
    seed,
    predef_W,
    known_objs,
    lambd,
    N,
    L,
    dataset_train,
    A,
    list_bn_train,
    groups_N_train,
    considered_groups,
    n_iters_heuristic=50,
    timelimit=60.0,
    silence=True,
    multistart=True,
):
    """Run the heuristic from one or every catalogue subset of size ``L``.

    Returns the same triple as
    :func:`src.heuristics.known_objectives.heuristic`: the history of the
    chosen start, its best objective, and the iteration at which that
    objective was found.
    """
    n_catalogue, _ = predef_W.shape
    np.random.seed(seed)
    random.seed(seed)

    # Each combination is one legal support: exactly L rows of the catalogue.
    starts = [
        (
            chosen_clusters,
            silence,
            N,
            n_catalogue,
            n_iters_heuristic,
            predef_W,
            L,
            dataset_train,
            A,
            list_bn_train,
            groups_N_train,
            considered_groups,
            timelimit,
            known_objs,
            lambd,
        )
        for chosen_clusters in combinations(range(n_catalogue), L)
    ]

    if multistart:
        # Workers are fresh interpreters. Put the repository root on
        # PYTHONPATH so they can import this package.
        repo_root = str(Path(__file__).resolve().parents[2])
        current = os.environ.get("PYTHONPATH", "")
        if repo_root not in current.split(os.pathsep):
            os.environ["PYTHONPATH"] = (
                repo_root + os.pathsep + current if current else repo_root
            )
        n_jobs = os.cpu_count()
        results = Parallel(n_jobs=n_jobs)(
            delayed(_alternating_passes)(start) for start in starts
        )
        best_start = int(np.argmin([best_obj for (_, best_obj, _) in results]))
        return results[best_start]

    start = random.sample(starts, 1)[0]
    return _alternating_passes(start)
