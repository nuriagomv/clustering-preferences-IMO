"""Inverse clustering model when the cost vectors ``C`` are known.

The mixed-integer program assigns each observed decision to a cluster and
picks one preference vector per cluster. A cluster is optimal for an instance
when the weighted cost ``w C`` admits dual multipliers that certify the
observed decision. The objective sums, over instances, the duality gap of the
cluster they are assigned to.
"""

import gurobipy as gp
import numpy as np


def problem_fknown(
    C,
    N,
    L,
    dataset,
    A,
    list_bn,
    groups_N,
    considered_groups,
    timelimit=None,
    silence=False,
    break_sym=False,
    linearize_product=False,
    predef_W=None,
    predef_u=None,
    predef_X=None,
    warmstart=None,
):
    """Build and solve the known-objective clustering model.

    Parameters
    ----------
    C : ndarray, shape (K, d)
        Known cost of each objective. Row ``k`` is the cost vector of
        objective ``k``.
    N, L : int
        Number of instances and number of clusters requested.
    dataset : ndarray, shape (N, d)
        Observed decisions, one row per instance.
    A : ndarray, shape (m, d)
        Shared constraint matrix in the form ``A z <= b``.
    list_bn : sequence of ndarray
        Right-hand side of instance ``n``.
    groups_N : sequence
        Feasible-region label of instance ``n``. Instances that share a label
        share a block of dual multipliers.
    considered_groups : sequence
        The distinct labels, in the order used to index the dual variable.
    timelimit : float, optional
        Gurobi time limit in seconds.
    silence : bool
        Suppress the Gurobi log.
    break_sym : bool
        Order clusters by ``sum_k k * W[l, k]`` so permutations of the same
        partition are not all enumerated.
    linearize_product : bool
        Replace the product of the assignment and the dual vector by an
        auxiliary continuous variable. Off by default; the product is left
        for Gurobi to handle.
    predef_W, predef_u, predef_X : optional
        Fix the corresponding decision instead of optimizing it. Used by the
        alternating heuristic. If ``predef_W`` has more rows than ``L``, the
        extra rows are a catalogue and at most ``L`` of them may be used.
    warmstart : tuple (X, W, u), optional
        Gurobi MIP start. Ignored for any block that was passed in predefined.

    Returns
    -------
    model : gurobipy.Model
    W, X, u : MVar or ndarray
        Preferences, assignments, dual multipliers. Predefined blocks are
        returned as the arrays that were passed in; optimized blocks are
        Gurobi variables (read ``.X`` after the solve).
    solution_log : list of dict
        One entry per feasible solution found by the callback.
    progress_logs : list of tuple
        ``(runtime, best objective, best bound)`` samples, about every 5 seconds.
    """
    n_objectives, _ = C.shape
    n_constraints, _ = A.shape
    if predef_W is not None:
        # ``L`` is the maximum number of catalogue rows we may turn on.
        # The variable blocks are sized to the full catalogue.
        catalogue_limit = L
        L, n_objectives = predef_W.shape

    solution_log = []
    count_checks = 0
    progress_logs = [(0.0, float("inf"), 0.0)]

    def my_callback(model, where):
        """Record incumbents, and stop after a long stretch with no progress.

        On each new feasible solution the weights and the cluster sizes are
        printed and stored. During the search, the incumbent and the bound
        are sampled every 5 seconds. After 100 samples with neither improving,
        the solve is terminated.
        """
        global count_checks

        if where == gp.GRB.Callback.MIPSOL:
            obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
            best_bound = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
            if obj_val != 0:
                mip_gap = abs(best_bound - obj_val) / abs(obj_val)
            else:
                mip_gap = float("inf")

            var_values = {
                var.VarName: model.cbGetSolution(var) for var in model.getVars()
            }
            weights = np.array(
                [value for name, value in var_values.items() if "W" in name]
            ).reshape((L, n_objectives))
            assignments = np.array(
                [value for name, value in var_values.items() if "X" in name]
            ).reshape((N, L))
            print(
                f"Feasible Solution Found - Time: {runtime:.2f}s, "
                f"Objective: {obj_val:.4f}, Gap: {mip_gap:.4%}, \n"
                f"Solution W_feas: \n{weights.round(3)}\n "
                f"Cluster composition: {assignments.sum(axis=0)}"
            )
            solution_log.append({
                "Time (s)": runtime,
                "Objective": obj_val,
                "Gap (%)": mip_gap * 100,
                "W_feas": weights,
                "X_feas": assignments,
                **var_values,
            })

        elif where == gp.GRB.Callback.MIP:
            runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
            if runtime - progress_logs[-1][0] > 5:
                best_sol = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                best_bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
                # Any improvement of the incumbent or the bound resets the counter.
                if best_sol < progress_logs[-1][1]:
                    count_checks = 0
                elif best_bound > progress_logs[-1][2]:
                    count_checks = 0
                else:
                    count_checks += 1
                progress_logs.append((runtime, best_sol, best_bound))
                if count_checks > 100:
                    print("STOPPING BECAUSE THERE IS NO IMPROVEMENT")
                    model.terminate()

    model = gp.Model()
    if silence:
        model.setParam("OutputFlag", 0)

    # X[n, l] = 1 iff instance n is explained by cluster l.
    if predef_X is None:
        X = model.addMVar(shape=(N, L), vtype=gp.GRB.BINARY, name="X")
        model.addConstrs(
            (gp.quicksum(X[n, l] for l in range(L)) == 1 for n in range(N)),
            name="all instances belong only to one cluster",
        )
    else:
        X = predef_X

    # W[l, :] is the preference of cluster l. It lives on the simplex.
    if predef_W is None:
        W = model.addMVar(
            shape=(L, n_objectives),
            lb=0,
            ub=1.0,
            vtype=gp.GRB.CONTINUOUS,
            name="W",
        )
        model.addConstrs(
            (
                gp.quicksum(W[l, k] for k in range(n_objectives)) == 1
                for l in range(L)
            ),
            name="weights' sum is one",
        )
        if break_sym:
            model.addConstrs(
                (
                    gp.quicksum(k * W[l, k] for k in range(n_objectives))
                    <= gp.quicksum(k * W[l + 1, k] for k in range(n_objectives))
                    for l in range(L - 1)
                ),
                name="symmetry break",
            )
    else:
        W = predef_W
        if L > catalogue_limit:
            # At most ``catalogue_limit`` rows of the catalogue may receive an instance.
            active = model.addMVar(shape=L, vtype=gp.GRB.BINARY, name="Y")
            model.addConstrs(
                (active[l] >= X[n, l] for n in range(N) for l in range(L)),
                name="catalogue_active",
            )
            model.addConstr(
                gp.quicksum(active[l] for l in range(L)) <= catalogue_limit,
                name="chosen_from_catalogue",
            )

    # Dual multipliers of A z <= b, one block per (feasible region, cluster).
    # Sign convention: u <= 0, and stationarity is u A = w C.
    n_regions = len(considered_groups)
    if predef_u is None:
        u = model.addMVar(
            shape=(n_regions, L, n_constraints),
            lb=-float("inf"),
            ub=0.0,
            vtype=gp.GRB.CONTINUOUS,
            name="u",
        )
        model.addConstrs(
            (
                u[region, l, :] @ A == W[l, :] @ C
                for region in range(n_regions)
                for l in range(L)
            ),
            name="dual_constr",
        )
    else:
        u = predef_u

    # Optional McCormick-style copy of the product X[n, l] * u[region(n), l, :].
    if linearize_product:
        product = model.addMVar(
            shape=(N, L, n_constraints),
            lb=-float("inf"),
            ub=0.0,
            vtype=gp.GRB.CONTINUOUS,
            name="v",
        )
        model.addConstrs(
            (
                product[n, l, i]
                == X[n, l] * u[considered_groups.index(groups_N[n]), l, i]
                for n in range(N)
                for l in range(L)
                for i in range(n_constraints)
            ),
            name="realocate_product",
        )

    def _gap(cluster, instance):
        """Duality gap of assigning ``instance`` to ``cluster``.

        With the stationarity constraint this equals
        ``X * (w C z - u b)``, i.e. the violation of optimality of the
        observed decision under that cluster's preference.
        """
        residual = A @ dataset[instance, :] - list_bn[instance]
        region = considered_groups.index(groups_N[instance])
        if linearize_product:
            return product[instance, cluster, :] @ residual
        return X[instance, cluster] * u[region, cluster, :] @ residual

    model.setObjective(
        gp.quicksum(_gap(l, n) for l in range(L) for n in range(N)),
        sense=gp.GRB.MINIMIZE,
    )

    if warmstart is not None:
        X.Start, W.Start, u.Start = warmstart

    model.update()
    if timelimit is not None:
        model.setParam(gp.GRB.Param.TimeLimit, timelimit)

    # The callback reads W and X by name, so it is only attached when both
    # of those blocks are real Gurobi variables.
    if (predef_W is None) and (predef_X is None):
        model.optimize(my_callback)
    else:
        model.optimize()

    return model, W, X, u, solution_log, progress_logs
