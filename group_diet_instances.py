import numpy as np
import pandas as pd
import gurobipy as gp
import random
from sklearn.metrics.pairwise import cosine_similarity

# Set print options to suppress scientific notation
np.set_printoptions(suppress=True)
    

def original_problem(linear_obj, A, b, n_vars_perturbed):
    """
    solves the following problem:

    min_z linear_obj@z
    s.t.  A·z <= b

    where n_vars_perturbed variables (<=d) in z can be perturbed
    """
    
    m,d = A.shape
    
    model_n = gp.Model()
    model_n.setParam('OutputFlag', 0)
    
    min=np.zeros(d)#min, max = np.zeros(d), np.repeat(2,d)
    if n_vars_perturbed > 0:
        #we perturb solutions by fixing some of the variables
        for j in random.sample(list(range(d)), n_vars_perturbed):
            #fix_to = np.random.uniform(0.5, 1.5)
            min[j] = 1.#min[j], max[j] = fix_to, fix_to
    z = model_n.addMVar(shape=d,
                        lb=min, ub=float('inf'),#max,
                        vtype=gp.GRB.CONTINUOUS, name="z")
    
    model_n.setObjective(linear_obj@z,
                        sense=gp.GRB.MINIMIZE)
    
    model_n.addConstr(A@z<=b)

    model_n.update()
    model_n.optimize()

    if model_n.status == 2:
        return z.X
    else:
        return None
    """
        z = z.X
    else:
        if n_vars_perturbed > 0: 
            if model_n.status == gp.GRB.INFEASIBLE:
                model_n.feasRelaxS(relaxobjtype=0,
                                   minrelax=False, vrelax=True, crelax=False)
                model_n.optimize()
                z = z.X
                feas = ((A@z<=b).sum() == m) and ((z >= 0).sum() == d)
                if not feas:
                    print("unable to relax and obtain feasibility")
                    z = None
        else:
            print("model status for unperturbed problem: ", model_n.status)
            z = None
    return z
    """
    

def get_diet_instances(N, K, d, n_vars_perturbed, seed, preferences='one_or_two'):

    # Reproducible experiments
    np.random.seed(seed)
    random.seed(seed) 
    
    data = pd.read_excel("sustainable_diet_plan_data.xlsx", sheet_name="food nutritional values")
    data = data.loc[:,['my regrouping', 
                       'ENERGY (Kcal)', 'PROTEIN (g)', 'CALCIUM (mg)', 'IRON (mg)', 'VIT B12 (μg)', 'CARB (g)', 'FIBRES (g)', 
                       'FAT (g)', 'SUGARS (g)', 
                       'CO2eq (g)', 'H2O (lt)', 'N (g)']]
    data = data.groupby('my regrouping').median()
    data = data.iloc[random.sample(range(data.shape[0]), d),:]
    chosen_foods = list(data.index)

    #list_of_objectives = ['FAT (g)', 'SUGARS (g)', 'CO2eq (g)', 'H2O (lt)', 'N (g)']
    #list_of_objectives = random.sample(list_of_objectives, K)
    if K== 2:
        list_of_objectives = ['FAT (g)', 'SUGARS (g)']
    if K==3:
        list_of_objectives = ['PROTEIN (g)', 'FAT (g)', 'SUGARS (g)']
        #list_of_objectives = ['CO2eq (g)', 'H2O (lt)', 'N (g)']
        
    print("LIST OF OBJECTIVES: ",list_of_objectives)
    #linear objectives normalized
    data.loc[:,list_of_objectives]
    C = [-data.loc[:,o] if o=='PROTEIN (g)' else data.loc[:,o] for o in list_of_objectives]
    C = np.array([c/np.abs(c).sum() for c in C])
    #np.array([data.loc[:,o]/abs(np.sum(data.loc[:,o])) for o in list_of_objectives])
    #np.linalg.norm(C,ord=1, axis=1)

    print("OBJECTIVES POINTING TO DIFFERENT DIRECTIONS?")
    for i in range(K-1):
        for j in range(i+1,K):
            a = C[i,:]
            b = C[j,:]
            print(list_of_objectives[i]," and ", list_of_objectives[j], ": cos_sim=",
                    round(cosine_similarity(np.array(a).reshape(1, -1), np.array(b).reshape(1, -1))[0][0], 3))


    #preferences
    W = np.zeros((N,K))
    for n in range(N):
        if preferences == 'random':
            w = [random.random() for _ in range(K)]
        if preferences == 'strict':
            goal = random.randint(0,K-1)
            w = [1. if k==goal else random.uniform(0,0.1) for k in range(K)]
        if preferences == 'one_or_two':
            all_goals = list(range(K)) + [None] 
            goals = random.sample(all_goals, k=2)
            w = np.array([random.uniform(5,10) if k in goals 
                            else random.uniform(0,1) 
                          for k in range(K)])
        W[n,:] = w/np.sum(w)
    print("Original preferences (W):\n", np.round(W,2))
    print("W mean: ",W.mean(axis=0).round(2))
    print("W std: ",W.std(axis=0).round(2))
    
    constraints = pd.read_excel("sustainable_diet_plan_data.xlsx", sheet_name="group lunch lower bounds", index_col=0)
    list_of_constraints = list(constraints.columns)
    considered_groups = constraints.index.to_list()
    #matrix of constraints
    A = np.array(data.loc[:,list_of_constraints]).transpose()
    _,d = A.shape
    A = np.concatenate([-A,-np.eye(d), np.eye(d)])
    m,_ = A.shape
    #RHS
    def get_b_n(group):
        b_n = np.array(constraints.loc[group,:])
        return np.concatenate([-b_n,np.zeros(d), np.repeat(2,repeats=d)])

    #get dataset of feasible solutions
    linear_objs = np.array([np.array([W[n,k]*C[k,:] for k in range(K)]).sum(axis=0) 
                            for n in range(N)])
    dataset = np.zeros((N,d))
    groups_N = []
    list_bn = []
    for n in range(N):
        groups_N.append(random.choice(considered_groups))
        b_n = get_b_n(groups_N[0])
        list_bn.append(b_n)
        feas_perturbed_flag = True
        while feas_perturbed_flag:
            z = original_problem(linear_objs[n,:], A, b_n, n_vars_perturbed)
            feas_perturbed_flag = (z is None)
        dataset[n,:] = z
    print("DATASET: \n",dataset.round(2))
    print("DATASET mean: ",dataset.mean(axis=0).round(2))
    print("DATASET std: ",dataset.std(axis=0).round(2))

    return data, chosen_foods, list_of_objectives, list_of_constraints, constraints, considered_groups, C, W, A, groups_N, list_bn, dataset
