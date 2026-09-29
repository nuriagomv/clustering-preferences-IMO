"""Known-objective experiments on synthetic forward problems.

Same solve as :mod:`experiments.run_known_objectives`, with the diet generator
replaced by :func:`src.data.synthetic.generate_synthetic_instances`. Train and
test instances are drawn together so they share ``C``, ``A`` and the group
labels.

Run from the repository root::

    python experiments/run_known_objectives_synthetic.py

Pickles go to ``outputs/known_synthetic/``.
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.synthetic import (
    generate_synthetic_instances,
    solve_forward_problem,
    to_diet_tuple,
)
from src.heuristics.known_objectives import cluster_assignments, heuristic
from src.metrics import regroup, summarize_recovery
from src.models.known_objectives import problem_fknown
from src.paths import OUTPUT_DIR, already_computed, save_path


np.set_printoptions(suppress=True)

OUTPUT = OUTPUT_DIR / "known_synthetic"


def _instability(preference, decisions, costs, constraint_matrix, right_hand_sides):
    """Objective gap of each decision against the unperturbed forward optimum."""
    weighted_cost = preference @ costs
    gaps = []
    for row in range(len(decisions)):
        optimal = solve_forward_problem(
            weighted_cost,
            constraint_matrix,
            right_hand_sides[row],
            n_vars_perturbed=0,
        )
        gaps.append(weighted_cost @ (decisions[row, :] - optimal))
    return gaps


def main():
    """Sweep the grid below. Configurations already saved in ``OUTPUT`` are skipped."""
    n_test = 100
    train_sizes = [100]
    n_objectives_grid = [5]
    dimensions = [20]
    perturbations = [1]  # 0 would store optimal decisions
    cluster_sizes = [5, 8, 10, 12]  # the generator's true number of clusters is 10
    interpretability_grid = [False]
    n_clusters_true = 10
    n_groups = 5
    n_core_constraints = 8
    noise_concentration = 50.0
    time_limit = 3600.0
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
                        perturbed = bool(n_perturbed > 0)
                        synthetic = generate_synthetic_instances(
                            N=n_train + n_test,
                            K=n_objectives,
                            d=dimension,
                            m_core=n_core_constraints,
                            n_groups=n_groups,
                            n_clusters=n_clusters_true,
                            perturbed=perturbed,
                            n_vars_perturbed=n_perturbed,
                            noise_concentration=noise_concentration,
                            seed=seed,
                            verbose=True,
                        )
                        instances = to_diet_tuple(synthetic)
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

                        # In catalogue mode the ground-truth centroids are the catalogue.
                        catalogue = synthetic["cluster_centroids"]

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
                                fragment = (
                                    "synth-fKNOWN-"
                                    + str(interpretability)
                                    + "-"
                                    + run_name
                                )
                                if already_computed(OUTPUT, fragment):
                                    print("ALREADY DONE -- SKIPPING")
                                    continue
                                print("NOT ALREADY DONE")

                                if interpretability:
                                    predefined_weights = catalogue
                                    warmstart = None
                                    heuristic_history = None
                                    heuristic_objective = None
                                    best_iteration = None
                                    heuristic_time = None
                                else:
                                    predefined_weights = None
                                    heuristic_start = time.time()
                                    (
                                        heuristic_history,
                                        heuristic_objective,
                                        best_iteration,
                                    ) = heuristic(
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
                                    incumbent = heuristic_history[best_iteration]
                                    (
                                        weights_warm,
                                        assignments_warm,
                                        dual_warm,
                                        _,
                                        _,
                                    ) = regroup(
                                        incumbent["W"],
                                        np.abs(incumbent["X"]),
                                        incumbent["u"],
                                    )
                                    print("Time in heuristic: ", heuristic_time)
                                    print(
                                        "Best heuristic iter: ",
                                        best_iteration,
                                        ", obj val: ",
                                        heuristic_objective,
                                    )
                                    print("heuristic W:\n", weights_warm.round(2))
                                    print(
                                        "heuristic X (col sums): ",
                                        assignments_warm.sum(axis=0),
                                    )
                                    warmstart = (
                                        assignments_warm,
                                        weights_warm,
                                        dual_warm,
                                    )

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

                                try:
                                    regrouped = regroup(weights.X, assignments.X)
                                except Exception:
                                    regrouped = regroup(weights, assignments.X)
                                weights_re, assignments_re, n_effective, dropped = regrouped
                                print("W:\n", np.round(weights_re, 3))
                                print(
                                    "cluster composition (sum X by cols): ",
                                    assignments_re.sum(axis=0),
                                )
                                print("efective number of clusters: ", n_effective)

                                train_estimates = [
                                    weights_re[assignments_re[n, :].argmax(), :]
                                    for n in range(n_train)
                                ]
                                train_metrics, train_means = summarize_recovery(
                                    preferences_train, train_estimates, ndigits=3
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
                                    "MEAN TRAIN metrics "
                                    "(equal_maxs, rmse, rho, tau, cos_sim, emd):",
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
                                    preferences_test, test_estimates, ndigits=3
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
                                    "MEAN TEST metrics "
                                    "(equal_maxs, rmse, rho, tau, cos_sim, emd):",
                                    test_equal_maxs,
                                    test_rmse,
                                    test_rho,
                                    test_tau,
                                    test_cos_sim,
                                    test_emd,
                                )

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
                                    "TRAIN real - estimated instab (mean,std):",
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
                                    "TEST  real - estimated instab (mean,std):",
                                    (
                                        np.mean(test_gap["dif"]),
                                        np.std(test_gap["dif"]),
                                    ),
                                )

                                saved = {
                                    "seed": seed,
                                    "N": n_train,
                                    "n_vars_perturbed": n_perturbed,
                                    "perturbed": perturbed,
                                    "instances": instances,
                                    "synthetic_meta": {
                                        "cluster_centroids": synthetic["cluster_centroids"],
                                        "cluster_assignments_train": synthetic[
                                            "cluster_assignments"
                                        ][:n_train],
                                        "cluster_assignments_test": synthetic[
                                            "cluster_assignments"
                                        ][-n_test:],
                                        "n_clusters_true": n_clusters_true,
                                        "n_groups": n_groups,
                                        "m_core": n_core_constraints,
                                        "noise_concentration": noise_concentration,
                                    },
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
                                destination = save_path(OUTPUT, fragment + ".pkl")
                                with open(destination, "wb") as handle:
                                    pickle.dump(
                                        saved, handle, protocol=pickle.HIGHEST_PROTOCOL
                                    )
                                print("Saved:", destination)


if __name__ == "__main__":
    main()
