"""Inverse clustering model when the cost vectors ``C`` are unknown.

Preferences are taken from a catalogue (``predef_W``). The solver estimates
the cost matrix ``C`` and the cluster assignments together. The objective is
a convex combination of two terms:

* the same duality-gap loss used in the known-objective model;
* a squared error between ``C z^n`` and observed objective values, when those
  values are provided in ``known_objs``.

``lambda = 1`` uses only the duality gap. ``lambda = 0`` uses only the
regression term.
"""

import gurobipy as gp
import numpy as np


def problem_fUNknown(
    predef_W,
    N,
    L,
    dataset,
    A,
    list_bn,
    groups_N,
    considered_groups,
    timelimit=None,
    silence=False,
    linearize_product=False,
    predef_C=None,
    known_objs=None,
    lambd=1.0,
    predef_u=None,
    predef_X=None,
    warmstart=None,
    call_callback=False,
    mipgapabs=None,
):
    """Build and solve the unknown-objective clustering model.

    Parameters
    ----------
    predef_W : ndarray, shape (L_catalogue, K)
        Preference catalogue. If it has more rows than ``L``, at most ``L``
        rows may be selected.
    N, L : int
        Number of instances, and maximum number of active catalogue rows.
    dataset, A, list_bn, groups_N, considered_groups
        Same roles as in :func:`src.models.known_objectives.problem_fknown`.
    predef_C : tuple (C, c0), optional
        Fix the cost matrix and the intercepts. Used by the second stage of
        the heuristic, which only re-assigns instances.
    known_objs : dict, optional
        ``known_objs[n][k]`` is the value of objective ``k`` at the decision
        of instance ``n``. Missing means the regression term is zero.
    lambd : float
        Weight on the duality-gap term. The regression term gets ``1 - lambd``.
    warmstart : tuple (X, (C, c0), u), optional
        MIP start for the assignments, the costs, the intercepts and the duals.
    call_callback : bool
        Attach the incumbent / stagnation callback. The callback also stops
        the solve once its own gap measure drops to ``1e-4``.
    mipgapabs : float, optional
        Gurobi ``MIPGapAbs`` parameter.

    Returns
    -------
    model, C, c0, X, u, solution_log, progress_logs
        Optimized blocks are Gurobi variables; predefined blocks are the
        arrays that were passed in.
    """
    _, dimension = dataset.shape
    n_constraints, _ = A.shape

    catalogue_limit = L
    L, n_objectives = predef_W.shape

    solution_log = []
    count_checks = 0
    progress_logs = [(0.0, float("inf"), 0.0)]

    def my_callback(model, where):
        """Record incumbents and stop on a small gap or on stagnation.

        The printed gap is ``|bound - obj| / (|obj| + 1)``, which stays
        informative when the objective is near zero. The solve is cut once
        that quantity is at most ``1e-4``, and also after 100 consecutive
        five-second samples with no improvement.
        """
        global count_checks

        if where == gp.GRB.Callback.MIPSOL:
            obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
            best_bound = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
            print("best_bound:", best_bound)
            print("obj_val:", obj_val)
            if obj_val != 0:
                mip_gap = abs(best_bound - obj_val) / abs(obj_val)
            else:
                mip_gap = float("inf")
            my_mip_gap = abs(best_bound - obj_val) / (abs(obj_val) + 1)

            var_values = {
                var.VarName: model.cbGetSolution(var) for var in model.getVars()
            }
            costs = np.array(
                [value for name, value in var_values.items() if "C" in name]
            ).reshape((n_objectives, dimension))
            assignments = np.array(
                [value for name, value in var_values.items() if "X" in name]
            ).reshape((N, L))
            print(
                f"Feasible Solution Found - Time: {runtime:.2f}s, "
                f"Objective: {obj_val:.4f}, Gap: {my_mip_gap:.4%}, \n"
                f"Solution C_feas: \n{costs.round(3)}\n "
                f"Cluster composition: {assignments.sum(axis=0)}"
            )
            solution_log.append({
                "Time (s)": runtime,
                "Objective": obj_val,
                "Gap (%)": mip_gap * 100,
                "my gap (%)": my_mip_gap * 100,
                "C_feas": costs,
                "X_feas": assignments,
                **var_values,
            })
            if my_mip_gap <= 1e-4:
                model.terminate()

        elif where == gp.GRB.Callback.MIP:
            runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
            if runtime - progress_logs[-1][0] > 5:
                best_sol = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                best_bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
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
    if mipgapabs is not None:
        model.setParam("MIPGapAbs", mipgapabs)
    if silence:
        model.setParam("OutputFlag", 0)

    # C[k, j] is the unknown cost of food j under objective k.
    # The L1 normalization sum_j |C[k, j]| = 1 is linearized with P and S:
    # S[k, j] = 1 picks the positive part, S[k, j] = 0 picks the negative part,
    # and 2 P[k, j] - C[k, j] equals |C[k, j]|.
    if predef_C is None:
        C = model.addMVar(
            shape=(n_objectives, dimension),
            lb=-1,
            ub=1.0,
            vtype=gp.GRB.CONTINUOUS,
            name="C",
        )
        c0 = model.addMVar(
            shape=n_objectives,
            lb=-float("inf"),
            ub=+float("inf"),
            vtype=gp.GRB.CONTINUOUS,
            name="c0",
        )
        sign = model.addMVar(
            shape=(n_objectives, dimension), vtype=gp.GRB.BINARY, name="S"
        )
        positive_part = model.addMVar(
            shape=(n_objectives, dimension),
            lb=0.0,
            ub=1.0,
            vtype=gp.GRB.CONTINUOUS,
            name="P",
        )
        model.addConstrs(
            (
                2 * positive_part[k, j] - C[k, j] >= 0
                for k in range(n_objectives)
                for j in range(dimension)
            ),
            name="define_absval",
        )
        model.addConstrs(
            (
                gp.quicksum(
                    2 * positive_part[k, j] - C[k, j] for j in range(dimension)
                )
                == 1
                for k in range(n_objectives)
            ),
            name="normalization",
        )
        model.addConstrs(
            (
                positive_part[k, j] <= sign[k, j]
                for k in range(n_objectives)
                for j in range(dimension)
            ),
            name="define_P1",
        )
        model.addConstrs(
            (
                positive_part[k, j] <= C[k, j] + (1 - sign[k, j])
                for k in range(n_objectives)
                for j in range(dimension)
            ),
            name="define_P2",
        )
        model.addConstrs(
            (
                positive_part[k, j] >= C[k, j] - (1 - sign[k, j])
                for k in range(n_objectives)
                for j in range(dimension)
            ),
            name="define_P3",
        )
    else:
        C, c0 = predef_C

    if predef_X is None:
        X = model.addMVar(shape=(N, L), vtype=gp.GRB.BINARY, name="X")
        model.addConstrs(
            (gp.quicksum(X[n, l] for l in range(L)) == 1 for n in range(N)),
            name="all instances belong only to one cluster",
        )
    else:
        X = predef_X

    W = predef_W
    if L > catalogue_limit:
        active = model.addMVar(shape=L, vtype=gp.GRB.BINARY, name="Y")
        model.addConstrs(
            (active[l] >= X[n, l] for n in range(N) for l in range(L)),
            name="catalogue_active",
        )
        model.addConstr(
            gp.quicksum(active[l] for l in range(L)) <= catalogue_limit,
            name="chosen_from_catalogue",
        )

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
        """Duality gap of assigning ``instance`` to ``cluster``. See the known-objective model."""
        residual = A @ dataset[instance, :] - list_bn[instance]
        region = considered_groups.index(groups_N[instance])
        if linearize_product:
            return product[instance, cluster, :] @ residual
        return X[instance, cluster] * u[region, cluster, :] @ residual

    def _regression(known_objs):
        """Squared error of predicting the observed objective values by ``C z``.

        The intercept ``c0`` is a decision variable (the warm start sets it)
        but it is not part of this residual: the fitted value is ``C_k · z^n``.
        """
        if known_objs is None:
            return 0
        return gp.quicksum(
            (known_objs[n][k] - C[k, :] @ dataset[n, :]) ** 2
            for n in known_objs.keys()
            for k in range(n_objectives)
        )

    regression = _regression(known_objs)
    model.setObjective(
        lambd * gp.quicksum(_gap(l, n) for l in range(L) for n in range(N))
        + (1 - lambd) * regression,
        sense=gp.GRB.MINIMIZE,
    )
    model.addConstr(regression >= 0)

    model.update()
    if timelimit is not None:
        model.setParam(gp.GRB.Param.TimeLimit, timelimit)

    if warmstart is not None:
        X_start, (C_start, c0_start), u_start = warmstart
        X.Start = X_start
        C.Start = C_start
        c0.Start = c0_start
        u.Start = u_start

    if call_callback:
        model.optimize(my_callback)
    else:
        model.optimize()

    return model, C, c0, X, u, solution_log, progress_logs
