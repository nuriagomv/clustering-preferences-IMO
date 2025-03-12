import gurobipy as gp
from sklearn.cluster import KMeans #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
import numpy as np
from formulation_fUNknown import problem_fUNknown
import group_diet_instances as diet
import auxfuncs as f
import pandas as pd
import random
from heuristic_fknown import cluster_assignments
from itertools import combinations
from joblib import Parallel, delayed
import os


def random_init(chosen_clusters,N,L_prev):

    fixed_X = np.zeros((N,L_prev))
    for n in range(N):
        assignment = random.sample(chosen_clusters,1)
        fixed_X[n,assignment] = 1
    print("X_init composition = \n", fixed_X.sum(axis=0))
    return fixed_X


def iterative_process(input):

    chosen_clusters, silence, N,L_prev, n_iters_heuristic, predef_W, L, dataset_train, A, list_bn_train,  groups_N_train, considered_groups, timelimit, known_objs, lambd= input
    
    #INITIALIZATION
    fixed_X = random_init(chosen_clusters,N,L_prev)
    print("TWO STAGE HEURISTIC: \n")
    heuristic_outputs = []
    heuristic_outputs.append( {'C': None, 'c0':None,
                               'u': None, 'X': fixed_X, 
                               'obj_val': float('inf')} )
    best_obj_val, best_iter = float('inf'), -float('inf')
    i, flag = 0, True
    while (i<=n_iters_heuristic) and flag:
        i+=1
        print("ITERATION ", i)

        #####################################
        #FIRST STAGE
        
        #I need symmetry breaking
        model1, C, c0, _, u, _, _ = problem_fUNknown(predef_W,
                                                     N, L, dataset_train, A, list_bn_train,  groups_N_train, considered_groups,
                                                     timelimit=timelimit, silence=silence,
                                                     predef_X = fixed_X, 
                                                     known_objs =  known_objs, lambd=lambd)
        new_C,new_c0, new_u = C.X, c0.X, u.X
        
        
        #####################################
        #SECOND STAGE
        """
        delete_cluster = [l for l in range(L_prev) if fixed_X[:,l].sum()==0]
        new_X, obj_val = cluster_assignments(new_u, dataset_train, A, list_bn_train, groups_N_train, considered_groups,
                                             delete_cluster = delete_cluster)
        """
        model2, _,_, new_X, _, _, _ = problem_fUNknown(predef_W,
                                                       N, L, dataset_train, A, list_bn_train,  groups_N_train, considered_groups,
                                                       timelimit=timelimit, silence=silence,
                                                       predef_u = new_u, predef_C=(new_C,new_c0),
                                                       known_objs =  known_objs, lambd=lambd)
        new_X = new_X.X
        obj_val = model2.ObjVal
        
        print("First step:")
        print("new C: \n", new_C.round(2))

        print("Second step:")
        print("new X: \n", new_X.sum(axis=0))
        print("obj val: ", obj_val)

        if obj_val == heuristic_outputs[-1]['obj_val']:#obj_val >= best_obj_val:
            print("EARLY STOP: solution not improved")
            flag = False
        if obj_val < best_obj_val:#else:
            best_obj_val, best_iter = obj_val, i
        
        heuristic_output = {'C': new_C, 'c0': new_c0, 'u': new_u, 'X': new_X, 'obj_val': obj_val}

        heuristic_outputs.append( heuristic_output )

        fixed_X = new_X

    return heuristic_outputs, best_obj_val, best_iter


def heuristic(seed, predef_W, known_objs, lambd,
              N, L, dataset_train, A, list_bn_train,  groups_N_train, considered_groups,
              n_iters_heuristic = 50, timelimit = 60., 
              silence = True,
              multistart = True):
        
    L_prev,K = predef_W.shape
    # Reproducible experiments
    np.random.seed(seed)
    random.seed(seed) 

    inputs = [(chosen_clusters, silence, N,L_prev, n_iters_heuristic, predef_W, L, dataset_train, A, list_bn_train,  groups_N_train, considered_groups, timelimit, known_objs, lambd) 
              for chosen_clusters in list(combinations(list(range(L_prev)),L))]
    if multistart:
        #import multiprocessing
        #num_cores = multiprocessing.cpu_count()
        num_cores = os.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(iterative_process)(input) for input in inputs)
        best_random = np.argmin([best_obj_val for (heuristic_outputs, best_obj_val, best_iter) in results])
        heuristic_outputs, best_obj_val, best_iter = results[best_random]
    else:
        input = random.sample(inputs,1)[0]
        heuristic_outputs, best_obj_val, best_iter = iterative_process(input)
    
    return (heuristic_outputs, best_obj_val, best_iter)
    
