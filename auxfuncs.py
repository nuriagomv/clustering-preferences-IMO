import numpy as np
from sklearn.metrics import mean_squared_error as mse
import math
import random
from scipy.stats import spearmanr, kendalltau, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity


def analyze_instance(w_orig, w_est):
    
    #identify first priority of the objectives
    equal_maxs = w_est.argmax() in np.where(w_orig == np.max(w_orig))[0]
    #rmse
    rmse = math.sqrt(mse(w_orig,w_est))
    #ranking
    rho, _ = spearmanr(w_orig, w_est)
    tau, _ = kendalltau(w_orig, w_est)
    #proportionality
    cos_sim = cosine_similarity(np.array(w_orig).reshape(1, -1), np.array(w_est).reshape(1, -1))[0][0]
    #spacing between weights
    emd = wasserstein_distance(w_orig, w_est)

    return equal_maxs, rmse, rho, tau, cos_sim, emd


def regroup(W,X, u = None):

    L,K = W.shape
    N,_ = X.shape

    delete_cluster = []

    s = np.array(range(1,K+1))
    order = np.argsort([s@w for w in W])
    W_ordered = np.zeros(W.shape)
    X_ordered = np.zeros(X.shape)
    if u is not None:
        u_ordered = np.zeros(u.shape)
    for l in range(L):
        W_ordered[l,:] = W[order[l],:]
        if u is not None:
            u_ordered[:,l,:] = u[:,order[l],:]
        for n in range(N):
            if X[n,order[l]] ==1:
                X_ordered[n,l] = 1
    
    W_re, X_re = W_ordered.copy(), X_ordered.copy()
    for l in range(L):
        if X_re[:,l].sum() == 0:
            W_re[l,:] = np.zeros(K)
            delete_cluster.append(l)
    for (l,ltild) in [(l,ltild) for l in range(L) for ltild in range(L) if l!=ltild]:
        if np.array([math.isclose(wl,wltild) for (wl,wltild) in zip(W_re[l,:], W_re[ltild,:])]).all():
            X_re[:,l] += X_re[:,ltild]
            X_re[:,ltild] = np.zeros(N)
            W_re[ltild,:] = np.zeros(K)
    L_efective = np.sum(X_re.sum(axis = 0) > 0)

    if u is None:
        return W_re, X_re, L_efective, delete_cluster
    else:
        return W_re, X_re, u_ordered, L_efective, delete_cluster


"""
def data_through_gaps(N,K,seed):

    # Set print options to suppress scientific notation
    np.set_printoptions(suppress=True)
    # Reproducible experiments
    #experiments are reproducible
    np.random.seed(seed)
    random.seed(seed)

    min_goals = list(range(int(K/2)))
    max_goals = list(range(int(K/2),K))
    noise = min_goals.pop(random.choice(min_goals))
    all_goals = min_goals+max_goals
    
    G = np.zeros((N,K))
    goals_list = []
    for n in range(N):
        case = random.choices([min_goals,max_goals,all_goals],
                              weights=[0.3,0.3,0.4], k = 1)[0]
        goals = random.sample(case, k=min(int(K/2),len(min_goals)))
        goals_list.append(goals)
        if np.sum([k in min_goals for k in goals]) in [0,len(goals)]:
            print("all objectives are maximizing or minimizing")
            g = np.array([random.uniform(50,100) if k == noise else 
                          (random.uniform(0,1) if k in goals else random.uniform(5,10))
                          for k in range(K)])
        else:
            #all gaps are going to be high because you cannot optimize both
            g = np.array([random.uniform(10,20) if k == noise else random.uniform(3,9)
                          for k in range(K)])
        G[n,:] = g

    return G, noise, min_goals, max_goals, goals_list
"""
