"""Known-objective experiments on the sustainable-diet instances.

Costs ``C`` are treated as observed. For each configuration the script

1. builds diet instances and splits off a test set of ``N_TEST`` rows,
2. warm-starts with the alternating heuristic (unless the catalogue mode is on),
3. solves :func:`src.models.known_objectives.problem_fknown`,
4. scores the recovered preferences in and out of sample,
5. writes one pickle under ``outputs/known_diet/``.

Edit the grid at the top of :func:`main` to change what is run.
Run from the repository root::

    python experiments/run_known_objectives.py
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.catalogues import weight_catalogue
from src.data import diet
from src.heuristics.known_objectives import cluster_assignments, heuristic
from src.metrics import regroup, summarize_recovery
from src.models.known_objectives import problem_fknown
from src.paths import OUTPUT_DIR, already_computed, save_path


np.set_printoptions(suppress=True)

OUTPUT = OUTPUT_DIR / "known_diet"


def _instability(preference, decisions, costs, constraint_matrix, right_hand_sides):
    """Gap between each observed decision and the true optimum of ``preference @ C``.

    The forward problem is solved without perturbations, so the gap is zero
    when the stored decision was already optimal for that preference.
    """
    weighted_cost = preference @ costs
    gaps = []
    for row in range(len(decisions)):
        optimal = diet.original_problem(
            weighted_cost, constraint_matrix, right_hand_sides[row], 0
        )
        gaps.append(weighted_cost @ (decisions[row, :] - optimal))
    return gaps


def main():
    """Sweep the grid below. Configurations already saved in ``OUTPUT`` are skipped."""
    # ---------------------------------------------------------------- grid
    n_test = 100
    train_sizes = [100]
    n_objectives_grid = [3]
    dimensions = [20]
    perturbations = [0]  # 0: optimal decisions; >0: that many foods fixed >= 1
    # Catalogue mode fixes W to weight_catalogue(K) instead of optimizing it.
    interpretability_grid = [False]
    # Cluster counts that are actually solved for this driver.
    cluster_sizes = [5, 6]
    time_limit = 3600.0
    seeds = [2]

    for seed in seeds:
        for n_train in train_sizes:
            for n_objectives in n_objectives_grid:
                for dimension in dimensions:
                    for n_perturbed in perturbations:
                        print(
                            "DATA CONFIGURATION: (seed, N, K, d, n_vars_perturbed)=",
                            (seed, n_train, n_objectives, dimension, n_perturbed),
                        )
                        # Train and test are one sample, then sliced, so they
                        # share C, A and the food subset.
                        instances = diet.get_diet_instances(
                            n_train + n_test,
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
                            costs,
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

                            for interpretability in interpretability_grid:
                                print("INTERPRETABILITY: ", interpretability)
                                fragment = str(interpretability) + "-" + run_name
                                if already_computed(OUTPUT, fragment):
                                    print("ALREADY DONE")
                                    continue
                                print("NOT ALREADY DONE")

                                if interpretability:
                                    predefined_weights = weight_catalogue(n_objectives)
                                    if predefined_weights is None:
                                        print("PREDEF NOT DEFINED FOR THIS CASE OF K")
                                    warmstart = None
                                    heuristic_history = None
                                    heuristic_objective = None
                                    best_iteration = None
                                    heuristic_time = None
                                else:
                                    predefined_weights = None
                                    heuristic_start = time.time()
                                    heuristic_history, heuristic_objective, best_iteration = heuristic(
                                        seed,
                                        costs,
                                        n_train,
                                        n_clusters,
                                        decisions_train,
                                        constraint_matrix,
                                        rhs_train,
                                        groups_train,
                                        considered_groups,
                                        init_type="optimal",
                                    )
                                    heuristic_time = time.time() - heuristic_start
                                    # Reorder the warm start the same way the
                                    # final solution is reordered before scoring.
                                    incumbent = heuristic_history[best_iteration]
                                    weights_warm, assignments_warm, dual_warm, _, _ = regroup(
                                        incumbent["W"],
                                        np.abs(incumbent["X"]),
                                        incumbent["u"],
                                    )
                                    print("Time in heuristic: ", heuristic_time)
                                    print(
                                        "Best solution found at iteration: ",
                                        best_iteration,
                                        ", with objval: ",
                                        heuristic_objective,
                                    )
                                    print("new W: \n", weights_warm.round(2))
                                    print("new X: \n", assignments_warm.sum(axis=0))
                                    warmstart = (assignments_warm, weights_warm, dual_warm)

                                solve_start = time.time()
                                (
                                    model,
                                    weights,
                                    assignments,
                                    dual,
                                    solution_log,
                                    progress_loss,
                                ) = problem_fknown(
                                    costs,
                                    n_train,
                                    n_clusters,
                                    decisions_train,
                                    constraint_matrix,
                                    rhs_train,
                                    groups_train,
                                    considered_groups,
                                    break_sym=False,
                                    linearize_product=False,
                                    timelimit=time_limit,
                                    predef_W=predefined_weights,
                                    warmstart=warmstart,
                                )
                                solve_time = time.time() - solve_start

                                print("time to sol: ", round(solve_time, 2), "s.")
                                # Status 2 is optimal. Otherwise report the relative gap in percent.
                                if model.status == 2:
                                    gap = 0.0
                                else:
                                    gap = model.MIPGap * 100
                                solver_objective = model.objVal
                                print(
                                    "objective val from solver: ",
                                    solver_objective,
                                    ", gap: ",
                                    gap,
                                )

                                # Catalogue mode returns W as a plain array, so .X fails.
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

                                train_estimates = [
                                    weights_re[assignments_re[n, :].argmax(), :]
                                    for n in range(n_train)
                                ]
                                train_metrics, train_means = summarize_recovery(
                                    preferences_train, train_estimates, ndigits=2
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

                                # Test instances are assigned with the duals of the
                                # solve, ignoring clusters that regroup() emptied.
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
                                    preferences_test, test_estimates, ndigits=2
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

                                # How much worse the estimated preference is, as an
                                # objective gap, than the preference that generated the data.
                                train_gap = {"real": [], "estimated": [], "dif": []}
                                for n in range(n_train):
                                    paired = {
                                        "estimated": train_estimates[n],
                                        "real": preferences_train[n, :],
                                    }
                                    for label in ("estimated", "real"):
                                        train_gap[label].extend(
                                            _instability(
                                                paired[label],
                                                decisions_train[n : n + 1, :],
                                                costs,
                                                constraint_matrix,
                                                rhs_train[n : n + 1],
                                            )
                                        )
                                    train_gap["dif"].append(
                                        train_gap["real"][-1] - train_gap["estimated"][-1]
                                    )
                                print(
                                    "real instab - estimated instab: (mean,std) ",
                                    (np.mean(train_gap["dif"]), np.std(train_gap["dif"])),
                                )

                                test_gap = {"real": [], "estimated": [], "dif": []}
                                for n in range(n_test):
                                    paired = {
                                        "estimated": test_estimates[n],
                                        "real": preferences_test[n, :],
                                    }
                                    for label in ("estimated", "real"):
                                        test_gap[label].extend(
                                            _instability(
                                                paired[label],
                                                decisions_test[n : n + 1, :],
                                                costs,
                                                constraint_matrix,
                                                rhs_test[n : n + 1],
                                            )
                                        )
                                    test_gap["dif"].append(
                                        test_gap["real"][-1] - test_gap["estimated"][-1]
                                    )
                                print(
                                    "real instab - estimated instab: (mean,std) ",
                                    (np.mean(test_gap["dif"]), np.std(test_gap["dif"])),
                                )

                                # Keys match the historical pickle schema, including
                                # the spelling of ``L_efective``.
                                saved = {
                                    "seed": seed,
                                    "N": n_train,
                                    "n_vars_perturbed": n_perturbed,
                                    "instances": instances,
                                    "K": n_objectives,
                                    "d": dimension,
                                    "interpretability": interpretability,
                                    "L": n_clusters,
                                    "heuristic": heuristic_history,
                                    "best_obj_val": heuristic_objective,
                                    "best_iter": best_iteration,
                                    "time_in_heur": heuristic_time,
                                    "warmstart": warmstart,
                                    "optimality": model.status == 2,
                                    "time": round(solve_time, 3),
                                    "gap": gap,
                                    "solution_log": solution_log,
                                    "progress_loss": progress_loss,
                                    "obj_val": solver_objective,
                                    "print_W": str(weights_re.round(2)),
                                    "W": np.round(weights_re, 3),
                                    "X": assignments_re.astype(int),
                                    "X_composition": assignments_re.sum(axis=0),
                                    "L_efective": n_effective,
                                    "delete_clusters": dropped,
                                    "clusters_test": test_clusters,
                                    "TRcomparison_instabs": train_gap,
                                    "TRcomparison_instabs_mean": np.mean(train_gap["dif"]),
                                    "TRAIN_metrics": train_metrics,
                                    "TRmean_equal_maxs": train_equal_maxs,
                                    "TRmean_rmse": train_rmse,
                                    "TRmean_rho": train_rho,
                                    "TRmean_tau": train_tau,
                                    "TRmean_cos_sim": train_cos_sim,
                                    "TRmean_emd": train_emd,
                                    "TScomparison_instabs": test_gap,
                                    "TScomparison_instabs_mean": np.mean(test_gap["dif"]),
                                    "TEST_metrics": test_metrics,
                                    "TSmean_equal_maxs": test_equal_maxs,
                                    "TSmean_rmse": test_rmse,
                                    "TSmean_rho": test_rho,
                                    "TSmean_tau": test_tau,
                                    "TSmean_cos_sim": test_cos_sim,
                                    "TSmean_emd": test_emd,
                                }
                                destination = save_path(
                                    OUTPUT, "soloutput-" + fragment + ".pkl"
                                )
                                with open(destination, "wb") as handle:
                                    pickle.dump(
                                        saved, handle, protocol=pickle.HIGHEST_PROTOCOL
                                    )
                                print("Saved:", destination)


if __name__ == "__main__":
    main()
