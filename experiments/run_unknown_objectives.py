"""Unknown-objective experiments on the sustainable-diet instances.

The preference of each cluster is chosen from :func:`src.catalogues.weight_catalogue`.
The cost matrix ``C`` is estimated. The objective mixes the duality gap with a
regression fit to the observed objective values, with weight ``lambd`` on the gap.

Run from the repository root::

    python experiments/run_unknown_objectives.py

One pickle per configuration is written to ``outputs/unknown_diet/``.
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.catalogues import weight_catalogue
from src.data import diet
from src.heuristics.known_objectives import cluster_assignments
from src.heuristics.unknown_objectives import heuristic
from src.metrics import regroup, summarize_recovery
from src.models.unknown_objectives import problem_fUNknown
from src.paths import OUTPUT_DIR, already_computed, save_path


np.set_printoptions(suppress=True)

OUTPUT = OUTPUT_DIR / "unknown_diet"


def _observed_objective_values(costs, decisions, n_instances, n_objectives):
    """``known_objs[n][k] = c^k · z^n`` for the first ``n_instances`` decisions.

    These are the targets of the regression term. ``decisions`` is the full
    generated sample; only the prefix of length ``n_instances`` is read.
    """
    observed = {}
    for instance in range(n_instances):
        observed[instance] = {}
        for objective in range(n_objectives):
            observed[instance][objective] = costs[objective, :] @ decisions[instance, :]
    return observed


def main():
    """Sweep the grid below. Configurations already saved in ``OUTPUT`` are skipped."""
    n_test = 100
    train_sizes = [25]
    n_objectives_grid = [2, 3]
    dimensions = [20]
    # Weight of the duality gap. The regression term receives ``1 - lambd``.
    lambdas = [0.75]
    perturbations = [1]
    cluster_sizes = [2, 3]
    time_limit = 3600.0
    multistart = False
    absolute_gap = 1e-4
    seeds = [0, 1, 2]

    for seed in seeds:
        for n_train in train_sizes:
            for n_objectives in n_objectives_grid:
                for dimension in dimensions:
                    for n_perturbed in perturbations:
                        print(
                            "DATA CONFIGURATION: (seed, N, K, d, n_vars_perturbed)=",
                            (seed, n_train, n_objectives, dimension, n_perturbed),
                        )
                        # Pool size is the literal ``100 + N_test`` from the
                        # original driver (200 with the grid above), not
                        # ``n_train + n_test``. Train is the first ``n_train``
                        # rows; test is the last ``n_test`` rows.
                        instances = diet.get_diet_instances(
                            100 + n_test,
                            n_objectives,
                            dimension,
                            n_perturbed,
                            seed,
                        )
                        (
                            _data,
                            _foods,
                            _objective_names,
                            _constraint_names,
                            _constraints,
                            considered_groups,
                            costs_true,
                            preferences,
                            constraint_matrix,
                            groups,
                            right_hand_sides,
                            decisions,
                        ) = instances

                        preferences_train = preferences[:n_train, :]
                        decisions_train = decisions[:n_train, :]
                        rhs_train = right_hand_sides[:n_train]
                        groups_train = groups[:n_train]
                        preferences_test = preferences[-n_test:, :]
                        decisions_test = decisions[-n_test:, :]
                        rhs_test = right_hand_sides[-n_test:]
                        groups_test = groups[-n_test:]

                        for n_clusters in cluster_sizes:
                            for lambd in lambdas:
                                print("\n-------------\n")
                                print("NUMBER OF CLUSTERS REQUIRED L =", n_clusters)
                                run_name = str((
                                    "seed", seed,
                                    "N", n_train,
                                    "K", n_objectives,
                                    "d", dimension,
                                    "n_pert", n_perturbed,
                                    "L", n_clusters,
                                ))
                                fragment = (
                                    "soloutput-fUNKNOWN-" + str(lambd) + "-" + run_name
                                )
                                if already_computed(OUTPUT, fragment):
                                    print("ALREADY DONE")
                                    continue
                                print("NOT ALREADY DONE")

                                # Regression targets: c^k · z^n on the training rows.
                                observed = _observed_objective_values(
                                    costs_true,
                                    decisions,
                                    n_train,
                                    n_objectives,
                                )
                                predefined_weights = weight_catalogue(n_objectives)

                                heuristic_start = time.time()
                                heuristic_history, heuristic_objective, best_iteration = heuristic(
                                    seed,
                                    predefined_weights,
                                    observed,
                                    lambd,
                                    n_train,
                                    n_clusters,
                                    decisions_train,
                                    constraint_matrix,
                                    rhs_train,
                                    groups_train,
                                    considered_groups,
                                    multistart=multistart,
                                    silence=True,
                                )
                                heuristic_time = time.time() - heuristic_start
                                incumbent = heuristic_history[best_iteration]
                                costs_warm = incumbent["C"]
                                intercept_warm = incumbent["c0"]
                                assignments_warm = np.abs(incumbent["X"])
                                dual_warm = incumbent["u"]
                                print("Time in heuristic: ", heuristic_time)
                                print(
                                    "Best solution found at iteration: ",
                                    best_iteration,
                                    ", with objval: ",
                                    heuristic_objective,
                                )
                                print("new C: \n", costs_warm.round(2))
                                print("new X: \n", assignments_warm.sum(axis=0))
                                warmstart = (
                                    assignments_warm,
                                    (costs_warm, intercept_warm),
                                    dual_warm,
                                )

                                solve_start = time.time()
                                (
                                    model,
                                    costs,
                                    intercept,
                                    assignments,
                                    dual,
                                    solution_log,
                                    _progress,
                                ) = problem_fUNknown(
                                    predefined_weights,
                                    n_train,
                                    n_clusters,
                                    decisions_train,
                                    constraint_matrix,
                                    rhs_train,
                                    groups_train,
                                    considered_groups,
                                    timelimit=time_limit,
                                    warmstart=warmstart,
                                    call_callback=True,
                                    mipgapabs=absolute_gap,
                                    known_objs=observed,
                                    lambd=lambd,
                                )
                                solve_time = time.time() - solve_start
                                weights = predefined_weights

                                print("time to sol: ", round(solve_time, 2), "s.")
                                if model.status == 2:
                                    gap = 0.0
                                else:
                                    gap = solution_log[-1]["my gap (%)"]
                                solver_objective = model.objVal
                                print(
                                    "objective val from solver: ",
                                    solver_objective,
                                    ", gap: ",
                                    gap,
                                )

                                try:
                                    regrouped = regroup(weights.X, assignments.X)
                                except Exception:
                                    regrouped = regroup(weights, assignments.X)
                                weights_re, assignments_re, n_effective, dropped = regrouped
                                print("W:\n", np.round(weights_re, 3))
                                print(
                                    "cluster composition, sum X by columns: ",
                                    assignments_re.sum(axis=0),
                                )
                                print("efective number of clusters: ", n_effective)
                                print("ESTIMATED COSTS: ")
                                print(costs.X.round(2))
                                print("WITH INTERCEPTS: ", intercept.X.round(2))
                                print("ORIGINAL COSTS: ")
                                print(costs_true.round(2))

                                # One cosine per objective, then the mean. This is the
                                # recovery score for C, separate from the preference scores.
                                similarities = [
                                    cosine_similarity(
                                        costs_true[k, :].reshape(1, -1),
                                        costs.X[k, :].reshape(1, -1),
                                    )[0, 0]
                                    for k in range(n_objectives)
                                ]
                                mean_similarity = np.mean(similarities)
                                print(
                                    "COSINE SIMILARITIES IN OBJS: ",
                                    similarities,
                                    " WITH MEAN=",
                                    mean_similarity,
                                )

                                train_estimates = [
                                    weights_re[assignments_re[n, :].argmax(), :]
                                    for n in range(n_train)
                                ]
                                train_metrics, train_means = summarize_recovery(
                                    preferences_train, train_estimates
                                )
                                (
                                    train_equal_maxs,
                                    train_rmse,
                                    train_rho,
                                    train_tau,
                                    train_cos_sim,
                                    train_emd,
                                ) = train_means
                                print(
                                    "MEAN TRAIN performance metrics: "
                                    "(equal_maxs, rmse, rho, tau, cos_sim, emd): ",
                                    train_equal_maxs,
                                    train_rmse,
                                    train_rho,
                                    train_tau,
                                    train_cos_sim,
                                    train_emd,
                                )

                                test_assignments, _ = cluster_assignments(
                                    dual.X,
                                    decisions_test,
                                    constraint_matrix,
                                    rhs_test,
                                    groups_test,
                                    considered_groups,
                                    delete_cluster=dropped,
                                )
                                test_clusters = [int(np.argmax(row)) for row in test_assignments]
                                test_estimates = [
                                    weights_re[test_clusters[n], :] for n in range(n_test)
                                ]
                                test_metrics, test_means = summarize_recovery(
                                    preferences_test, test_estimates
                                )
                                (
                                    test_equal_maxs,
                                    test_rmse,
                                    test_rho,
                                    test_tau,
                                    test_cos_sim,
                                    test_emd,
                                ) = test_means
                                print(
                                    "MEAN TEST performance metrics: "
                                    "(equal_maxs, rmse, rho, tau, cos_sim, emd): ",
                                    test_equal_maxs,
                                    test_rmse,
                                    test_rho,
                                    test_tau,
                                    test_cos_sim,
                                    test_emd,
                                )

                                saved = {
                                    "seed": seed,
                                    "N": n_train,
                                    "n_vars_perturbed": n_perturbed,
                                    "instances": instances,
                                    "K": n_objectives,
                                    "d": dimension,
                                    "L": n_clusters,
                                    "lambda": lambd,
                                    "heuristic": heuristic_history,
                                    "best_obj_val": heuristic_objective,
                                    "best_iter": best_iteration,
                                    "time_in_heur": heuristic_time,
                                    "warmstart": warmstart,
                                    "optimality": model.status == 2,
                                    "time": round(solve_time, 3),
                                    "gap": gap,
                                    "obj_val": solver_objective,
                                    "print_W": str(weights_re.round(2)),
                                    "W": np.round(weights_re, 3),
                                    "C": np.round(costs.X, 2),
                                    "c0": np.round(intercept.X, 2),
                                    "X": assignments_re.astype(int),
                                    "X_composition": assignments_re.sum(axis=0),
                                    "L_efective": n_effective,
                                    "delete_clusters": dropped,
                                    "clusters_test": test_clusters,
                                    "mean_sim_obj": mean_similarity,
                                    "sims": similarities,
                                    "TRAIN_metrics": train_metrics,
                                    "TRmean_equal_maxs": train_equal_maxs,
                                    "TRmean_rmse": train_rmse,
                                    "TRmean_rho": train_rho,
                                    "TRmean_tau": train_tau,
                                    "TRmean_cos_sim": train_cos_sim,
                                    "TRmean_emd": train_emd,
                                    "TEST_metrics": test_metrics,
                                    "TSmean_equal_maxs": test_equal_maxs,
                                    "TSmean_rmse": test_rmse,
                                    "TSmean_rho": test_rho,
                                    "TSmean_tau": test_tau,
                                    "TSmean_cos_sim": test_cos_sim,
                                    "TSmean_emd": test_emd,
                                }
                                destination = save_path(OUTPUT, fragment + ".pkl")
                                with open(destination, "wb") as handle:
                                    pickle.dump(
                                        saved, handle, protocol=pickle.HIGHEST_PROTOCOL
                                    )
                                print("Saved:", destination)


if __name__ == "__main__":
    main()
