import gurobipy as gp
from sklearn.cluster import KMeans #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
import numpy as np
from formulation_fknown import problem_fknown
import group_diet_instances as diet
import auxfuncs as f
import pandas as pd
import random


def init_optimal_wn(zn, An, bn, C, timelimit, silence):

    K,d = C.shape
    m,_ = An.shape

    model = gp.Model()
    if silence:
        model.setParam('OutputFlag', 0)
    if timelimit is not None:
        model.setParam(gp.GRB.Param.TimeLimit, timelimit)

    u = model.addMVar(shape=m, lb=-float('inf'), ub=0., vtype=gp.GRB.CONTINUOUS, name="u")
    
    w = model.addMVar(shape=K, lb=0, ub=1., vtype=gp.GRB.CONTINUOUS, name="W")
    model.addConstr(gp.quicksum(w[k] for k in range(K)) == 1, name="weights' sum is one")

    model.addConstr(u@An == w@C, name="dual_constr")
    
    model.setObjective(u@(An@zn - bn),sense=gp.GRB.MINIMIZE)

    model.update()
    model.optimize()

    if model.status == gp.GRB.Status.TIME_LIMIT:
        print("NOT ENOUGH TIME LIMIT")
    if model.status == 2:
        return model, w.X
    else:
        print("unable to find initialization for w_n")
        return model, None
    

def traditional_kmeans(L,Wn,seed):
    N,_ = Wn.shape
    kmeans = KMeans(n_clusters=L, random_state=seed, 
                    n_init="auto", init="k-means++").fit(Wn)

    fixed_X = np.zeros((N,L))
    for n in range(N):
        fixed_X[n,kmeans.labels_[n]] = 1

    return fixed_X


def optimizeforW_Xfixed(X_l, C, N, dataset, A, list_bn, groups_N, considered_groups, timelimit, silence):

    K,d = C.shape
    m,_ = A.shape
    
    model = gp.Model()
    if silence:
        model.setParam('OutputFlag', 0)
    if timelimit is not None:
        model.setParam(gp.GRB.Param.TimeLimit, timelimit)
    
    different_Z_cal = len(considered_groups)
    
    u = model.addMVar(shape=(different_Z_cal, m),
                      lb=-float('inf'), ub=0., vtype=gp.GRB.CONTINUOUS, name="u")
    
    w_l = model.addMVar(shape=K,
                        lb=0, ub=1., vtype=gp.GRB.CONTINUOUS, name="W")
    model.addConstr(gp.quicksum(w_l[k] for k in range(K)) == 1, name="weights' sum is one")

    model.addConstrs((u[feas_reg,:]@A == w_l@C 
                      for feas_reg in range(different_Z_cal)), name="dual_constr")
    
    def term(n):
        return X_l[n] * u[considered_groups.index(groups_N[n]),:] @ (A@dataset[n,:] - list_bn[n])
    
    model.setObjective(gp.quicksum(term(n) for n in range(N)), sense=gp.GRB.MINIMIZE)
    
    model.update()
    model.optimize()

    if model.status == gp.GRB.Status.TIME_LIMIT:
        print("NOT ENOUGH TIME LIMIT")
    if model.status == 2:
        return model, w_l.X, u.X
    else:
        print("could not optimize for w_l")
        return model, None


def cluster_assignments(u, dataset, A, list_bn, groups_N, considered_groups, delete_cluster = []):

    N = dataset.shape[0]
    L = u.shape[1]
    new_X = np.zeros((N,L))
    obj_val = 0
    for n in range(N):
        z_n = dataset[n,:]
        vals = [ u[considered_groups.index(groups_N[n]),l,:] @ (A@dataset[n,:] - list_bn[n]) if l not in delete_cluster else float('inf') for l in range(L)]
        obj_val += np.min(vals)
        assignment = np.argmin(vals)
        new_X[n,assignment] = 1

    return new_X, obj_val

def heuristic(seed, C, N, L, dataset_train, A, list_bn_train,  groups_N_train, considered_groups,
              init_type='optimal',
              n_iters_heuristic = 50, timelimit = 60., silence = True):
    
    K,_ = C.shape
    # Reproducible experiments
    np.random.seed(seed)
    random.seed(seed) 

    #INITIALIZATION
    print("HEURISTIC INITIALIZATION: ", init_type)

    if init_type == 'optimal':
        Wn = np.array([init_optimal_wn(dataset_train[n,:], A, list_bn_train[n], C, timelimit, silence)[1] for n in range(N)])
        fixed_X = traditional_kmeans(L,Wn,seed)
    
    if init_type == 'random':
        fixed_X = np.zeros((N,L))
        for n in range(N):
            assignment = random.randint(0,L-1)
            fixed_X[n,assignment] = 1
    print("X_init composition = \n", fixed_X.sum(axis=0))

    print("TWO STAGE HEURISTIC: \n")
    heuristic_outputs = []
    heuristic_outputs.append( {'W': None, 'u': None, 'X': fixed_X, 'obj_val': float('inf')} )
    best_obj_val, best_iter = float('inf'), -float('inf')
    i, flag = 0, True
    while (i<=n_iters_heuristic) and flag:
        i+=1
        print("ITERATION ", i)

        #####################################
        #FIRST STAGE
        new_W = np.zeros((L,K))
        new_u = np.zeros((len(considered_groups),L,A.shape[0]))
        for l in range(L):
            _, w_l, u_l = optimizeforW_Xfixed(fixed_X[:,l], C, N, dataset_train, A, list_bn_train,  groups_N_train, considered_groups, timelimit,silence)
            new_W[l,:] = w_l
            new_u[:,l,:] = u_l
        """
        #I need symmetry breaking
        model1, W, _, u, _, _ = problem_fknown(C, N, L, dataset_train, A, list_bn_train,  groups_N_train, considered_groups,
                                               timelimit=timelimit,
                                               break_sym=True,
                                               silence = silence,
                                               predef_X = fixed_X)
        new_W, new_u = W.X, u.X
        """
        
        #####################################
        #SECOND STAGE
        new_X, obj_val = cluster_assignments(new_u, dataset_train, A, list_bn_train, groups_N_train, considered_groups)
        """
        model2, _, new_X, _, _, _ = problem_fknown(C, N, L, dataset_train, A, list_bn_train,  groups_N_train, considered_groups,
                                                   timelimit=timelimit,
                                                   silence = silence,
                                                   predef_W=new_W, predef_u = new_u)
        new_X = new_X.X
        obj_val = model2.ObjVal
        """
        print("First step:")
        print("new W: \n", new_W.round(2))

        print("Second step:")
        print("new X: \n", new_X.sum(axis=0))
        print("obj val: ", obj_val)

        if obj_val == heuristic_outputs[-1]['obj_val']:#obj_val >= best_obj_val:
            print("EARLY STOP: solution not improved")
            flag = False
        if obj_val < best_obj_val:#else:
            best_obj_val, best_iter = obj_val, i
        
        heuristic_output = {'W': new_W, 'u': new_u, 'X': new_X, 'obj_val': obj_val}

        heuristic_outputs.append( heuristic_output )

        fixed_X = new_X

    return heuristic_outputs, best_obj_val, best_iter
