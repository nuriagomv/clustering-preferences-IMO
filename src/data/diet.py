"""Diet instances built from the sustainable-diet workbook.

Each instance is one person's lunch: a decision vector ``z`` (servings of
``d`` food groups), a shared constraint matrix ``A``, a group-dependent
right-hand side ``b^n``, and a private preference ``w^n`` over ``K``
nutritional objectives. The observed decision is the optimum of the weighted
sum ``w^n @ C``, optionally pushed away from optimality by fixing a few
servings away from zero.
"""

import random

import gurobipy as gp
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.paths import DIET_WORKBOOK


np.set_printoptions(suppress=True)


def original_problem(linear_obj, A, b, n_vars_perturbed):
    """Solve one forward diet problem, with optional suboptimal decisions.

    The linear program is

        min_z  linear_obj @ z
        s.t.   A z <= b
               z >= 0

    When ``n_vars_perturbed > 0``, that many coordinates of ``z`` are forced
    to be at least 1 before the solve. Those coordinates are chosen at random,
    so the returned decision is feasible but generally not optimal for the
    unperturbed problem. This is the suboptimal-decision protocol used in the
    experiments.

    Returns the optimal ``z``, or ``None`` if Gurobi does not close the solve
    with status optimal.
    """
    _, dimension = A.shape

    model = gp.Model()
    model.setParam("OutputFlag", 0)

    lower_bound = np.zeros(dimension)
    if n_vars_perturbed > 0:
        # Freeze a random subset of foods at a serving of at least 1.
        for food in random.sample(range(dimension), n_vars_perturbed):
            lower_bound[food] = 1.0

    servings = model.addMVar(
        shape=dimension,
        lb=lower_bound,
        ub=float("inf"),
        vtype=gp.GRB.CONTINUOUS,
        name="z",
    )
    model.setObjective(linear_obj @ servings, sense=gp.GRB.MINIMIZE)
    model.addConstr(A @ servings <= b)
    model.update()
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        return servings.X
    return None


def _canonical_rhs(constraints, group, dimension):
    """Right-hand side for one demographic group, in canonical ``A z <= b`` form.

    The workbook stores nutrient lower bounds. Together with the box
    ``0 <= z <= 2`` they become

        b = [-b_nutrient,  0_d,  2 * 1_d]
    """
    nutrient_lower_bounds = np.array(constraints.loc[group, :])
    upper_bounds = np.repeat(2, repeats=dimension)
    return np.concatenate([-nutrient_lower_bounds, np.zeros(dimension), upper_bounds])


def get_diet_instances(N, K, d, n_vars_perturbed, seed, preferences="one_or_two"):
    """Build ``N`` diet instances that share costs ``C`` and constraints ``A``.

    Parameters
    ----------
    N : int
        Number of instances. Drivers usually pass ``N_train + N_test`` and
        split the returned arrays afterwards.
    K : int
        Number of objectives. ``K = 2`` uses fat and sugars. ``K = 3`` uses
        protein, fat and sugars.
    d : int
        Number of food groups kept. They are a random subset of the groups
        in the workbook.
    n_vars_perturbed : int
        Passed through to :func:`original_problem`. ``0`` stores optimal
        decisions; a positive value stores suboptimal ones.
    seed : int
        Seed for NumPy and the ``random`` module.
    preferences : {"one_or_two", "random", "strict"}
        How ``w^n`` is drawn. ``one_or_two`` puts most of the weight on one
        or two randomly chosen objectives. ``random`` draws uniform weights.
        ``strict`` puts weight 1 on a single objective and a small residual
        on the others.

    Returns
    -------
    tuple
        ``data, chosen_foods, list_of_objectives, list_of_constraints,
        constraints, considered_groups, C, W, A, groups_N, list_bn, dataset``.

        ``C`` has shape ``(K, d)`` and each row sums in absolute value to 1.
        ``W`` has shape ``(N, K)`` and each row sums to 1. ``dataset`` has
        shape ``(N, d)``.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Food groups are the median nutrient profile of the rows that share a
    # "my regrouping" label. A subset of size d is what the model sees.
    data = pd.read_excel(DIET_WORKBOOK, sheet_name="food nutritional values")
    nutrient_columns = [
        "my regrouping",
        "ENERGY (Kcal)",
        "PROTEIN (g)",
        "CALCIUM (mg)",
        "IRON (mg)",
        "VIT B12 (μg)",
        "CARB (g)",
        "FIBRES (g)",
        "FAT (g)",
        "SUGARS (g)",
        "CO2eq (g)",
        "H2O (lt)",
        "N (g)",
    ]
    data = data.loc[:, nutrient_columns]
    data = data.groupby("my regrouping").median()
    data = data.iloc[random.sample(range(data.shape[0]), d), :]
    chosen_foods = list(data.index)

    # Protein is a nutrient to maximize. Negating its column turns every
    # objective into a minimization cost, which is what the forward LP expects.
    if K == 2:
        list_of_objectives = ["FAT (g)", "SUGARS (g)"]
    if K == 3:
        list_of_objectives = ["PROTEIN (g)", "FAT (g)", "SUGARS (g)"]
    print("LIST OF OBJECTIVES: ", list_of_objectives)

    raw_costs = [
        -data.loc[:, name] if name == "PROTEIN (g)" else data.loc[:, name]
        for name in list_of_objectives
    ]
    C = np.array([cost / np.abs(cost).sum() for cost in raw_costs])

    print("OBJECTIVES POINTING TO DIFFERENT DIRECTIONS?")
    for i in range(K - 1):
        for j in range(i + 1, K):
            similarity = cosine_similarity(
                C[i, :].reshape(1, -1), C[j, :].reshape(1, -1)
            )[0, 0]
            print(
                list_of_objectives[i],
                " and ",
                list_of_objectives[j],
                ": cos_sim=",
                round(similarity, 3),
            )

    # Private preferences. Each row is normalized onto the simplex.
    W = np.zeros((N, K))
    for n in range(N):
        if preferences == "random":
            raw_weight = [random.random() for _ in range(K)]
        if preferences == "strict":
            goal = random.randint(0, K - 1)
            raw_weight = [
                1.0 if k == goal else random.uniform(0, 0.1) for k in range(K)
            ]
        if preferences == "one_or_two":
            # ``None`` is a dummy draw: sampling two items from K labels plus
            # one dummy yields either one or two real objectives.
            candidates = list(range(K)) + [None]
            goals = random.sample(candidates, k=2)
            raw_weight = np.array([
                random.uniform(5, 10) if k in goals else random.uniform(0, 1)
                for k in range(K)
            ])
        W[n, :] = raw_weight / np.sum(raw_weight)
    print("Original preferences (W):\n", np.round(W, 2))
    print("W mean: ", W.mean(axis=0).round(2))
    print("W std: ", W.std(axis=0).round(2))

    constraints = pd.read_excel(
        DIET_WORKBOOK, sheet_name="group lunch lower bounds", index_col=0
    )
    list_of_constraints = list(constraints.columns)
    considered_groups = constraints.index.to_list()

    # Nutrient lower bounds, nonnegativity, and an upper bound of 2 per food:
    #     A = [-N ;  -I ;  I],   so A z <= b encodes N z >= b_nutrient.
    nutrient_matrix = np.array(data.loc[:, list_of_constraints]).T
    _, dimension = nutrient_matrix.shape
    A = np.concatenate([-nutrient_matrix, -np.eye(dimension), np.eye(dimension)])

    # Weighted-sum costs, one per instance, then the forward solve.
    weighted_costs = np.array([
        np.array([W[n, k] * C[k, :] for k in range(K)]).sum(axis=0)
        for n in range(N)
    ])
    dataset = np.zeros((N, dimension))
    groups_N = []
    list_bn = []
    for n in range(N):
        groups_N.append(random.choice(considered_groups))
        rhs = _canonical_rhs(constraints, groups_N[0], dimension)
        list_bn.append(rhs)
        # A perturbation can make the LP infeasible. Resample until it is not.
        decision = None
        while decision is None:
            decision = original_problem(
                weighted_costs[n, :], A, rhs, n_vars_perturbed
            )
        dataset[n, :] = decision

    print("DATASET: \n", dataset.round(2))
    print("DATASET mean: ", dataset.mean(axis=0).round(2))
    print("DATASET std: ", dataset.std(axis=0).round(2))

    return (
        data,
        chosen_foods,
        list_of_objectives,
        list_of_constraints,
        constraints,
        considered_groups,
        C,
        W,
        A,
        groups_N,
        list_bn,
        dataset,
    )
