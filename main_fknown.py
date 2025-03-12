from formulation_fknown import problem_fknown
import group_diet_instances as diet
import numpy as np
import time
import pandas as pd
import pickle
import auxfuncs as f
import math
import os
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import imageio
from heuristic_fknown import heuristic, cluster_assignments
                                    

# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# Set print options to suppress scientific notation
np.set_printoptions(suppress=True)

#parameters
N_test= 100
list_N = [100]
list_K = [2]
list_d = [20]
list_n_vars_perturbed = [0,1] #<= d, number of variables
list_interpretability = [False]
timelimit = 3600.

for seed in [2]:
    for N in list_N:
        for K in list_K:
            for d in list_d:
                for n_vars_perturbed in list_n_vars_perturbed:
                    
                    ###################################################
                    #diet problem
                    print("DATA CONFIGURATION: (seed, N, K, d, n_vars_perturbed)=", (seed,N,K,d, n_vars_perturbed))
                    instances = diet.get_diet_instances(100+N_test, K, d, n_vars_perturbed, seed)
                    data, chosen_foods, list_of_objectives, list_of_constraints, constraints, considered_groups, C, W_orig, A, groups_N, list_bn, dataset = instances
                    
                    W_orig_train, dataset_train, list_bn_train, groups_N_train = W_orig[:N,:], dataset[:N,:], list_bn[:N], groups_N[:N]
                    W_orig_test, dataset_test, list_bn_test, groups_N_test = W_orig[-N_test:,:], dataset[-N_test:,:], list_bn[-N_test:], groups_N[-N_test:]
                    
                    #number of clusters
                    if K ==2:
                        L_max = K +1
                    if K>2:
                        L_max = K + math.comb(K, math.ceil(K/2))
                    list_L = list(range(2,L_max+1))
                    
                    for L in list_L:
                        
                        print("\n-------------\n")
                        print("NUMBER OF CLUSTERS REQUIRED L =", L)
                        
                        name = str(("seed",seed,"N",N,"K",K,"d",d,"n_pert",n_vars_perturbed,"L",L))

                        ###################################################
                        
                        for interpretability in list_interpretability:
                            # solve problem
                            print("INTERPRETABILITY: ", interpretability)

                            #CHECK IF ALREADY DONE
                            already_done = np.any([(str(interpretability)+'-'+name) in string for string in os.listdir(r"./collecting outputs")])
                            if not already_done:
                                print("NOT ALREADY DONE")

                                if interpretability:
                                    if K==2:
                                        predef_W = np.array([[1., 0.],
                                                            [0., 1.],
                                                            [.25, .75],
                                                            [.75, .25],
                                                            [.5, .5]])
                                    elif K==3:
                                        predef_W = np.array([[1., 0., 0.],
                                                            [0., 1., 0.],
                                                            [0., 0., 1.],
                                                            [.5, .5, 0.],
                                                            [.25, .75, 0.],
                                                            [.75, .25, 0.],
                                                            [.5, 0., .5],
                                                            [.25, 0., .75],
                                                            [.75, 0., .25],
                                                            [0., .5, .5],
                                                            [0., .25, .75],
                                                            [0., .75, .25],
                                                            [1/3, 1/3, 1/3]])
                                    else:
                                        print("PREDEF NOT DEFINED FOR THIS CASE OF K")
                                        predef_W = None
                                    warmstart, heur_outputs, obj_val_heur, best_iter, time_in_heur = None, None, None, None, None
                                else:
                                    predef_W = None
                                    # heuristic solution
                                    inicio_heur = time.time()
                                    heur_outputs, obj_val_heur, best_iter = heuristic(seed, C, N, L, dataset_train, A, list_bn_train,  groups_N_train, considered_groups,
                                                                                      init_type='optimal')
                                    time_in_heur = time.time() - inicio_heur
                                    
                                    #HACER QUE WARM STAR SATISFAGA SYMMETRY BREAKING
                                    W_heur, X_heur, u_heur,_ , _ = f.regroup(heur_outputs[best_iter]['W'],
                                                                            np.abs(heur_outputs[best_iter]['X']),
                                                                            heur_outputs[best_iter]['u'])
                                    
                                    #print( pd.DataFrame(heuristic_outputs) )
                                    print("Time in heuristic: ", time_in_heur)
                                    print("Best solution found at iteration: ", best_iter, ", with objval: ", obj_val_heur)
                                    print("new W: \n", W_heur.round(2))
                                    print("new X: \n", X_heur.sum(axis=0))

                                    warmstart = (X_heur,W_heur,u_heur)
                                
                                inicio = time.time()
                                output  = problem_fknown(C, N, L, dataset_train, A, list_bn_train,  groups_N_train, considered_groups,
                                                        break_sym=False, linearize_product=False,
                                                        timelimit=timelimit, predef_W = predef_W,
                                                        warmstart = warmstart)
                                time_sol = time.time() - inicio
                                
                                model, W, X, u, solution_log, progress_loss = output
                                
                                print("time to sol: ", round(time_sol,2), "s.")
                                gap = 0. if model.status == 2 else model.MIPGap*100
                                obj_val_solver = model.objVal
                                print("objective val from solver: ", obj_val_solver, ", gap: ", gap)
                                
                                try:
                                    W_re, X_re, L_efective, delete_cluster = f.regroup(W.X,X.X)
                                except:
                                    W_re, X_re, L_efective, delete_cluster = f.regroup(W,X.X) #used predef_W != None for interpretability
                                print("W:\n", np.round(W_re,3))
                                #print("X:\n", X_re.astype(int))
                                print("cluster composition, sum X by columns: ", X_re.sum(axis = 0))
                                print("efective number of clusters: ", L_efective)

                                """
                                #depict clusters
                                if K in [2,3]:
                                    colors = ["b","g","r","m","y","k","c"]
                                    clust = dict([(l,colors.pop()) for l in range(7)])
                                    if K==2:
                                        fig, ax = plt.subplots()
                                    if K==3:
                                        fig = plt.figure()
                                        ax = fig.add_subplot(projection='3d')
                                        ax.view_init(elev=30, azim=50)
                                    fig.suptitle(formulation)
                                    ax.set_xlim((0,1))
                                    ax.set_ylim((0,1))
                                    if K==3:
                                        ax.set_zlim((0,1))
                                    for n in range(N):
                                        if K==2:
                                            ax.scatter(W_orig[n,0], W_orig[n,1], color=clust[X_re[n,:].argmax()])
                                        if K==3:
                                            ax.scatter(W_orig[n,0], W_orig[n,1], W_orig[n,2], color=clust[X_re[n,:].argmax()])
                                    for l in range(W_re.shape[0]):
                                        w_l = W_re[l,:]
                                        if w_l.sum()!= 0.:
                                            if K==2:
                                                ax.scatter(w_l[0],w_l[1], s=75,marker="X", color=clust[l],label="cluster_"+str(l+1))
                                            if K ==3:
                                                ax.scatter(w_l[0],w_l[1],w_l[2], s=75,marker="X", color=clust[l],label="cluster_"+str(l+1))
                                    fig.legend()
                                    
                                    # Save rotation frames
                                    #filenames = []
                                    #for angle in range(0, 360, 10):  # Rotate every 10 degrees
                                        #   ax.view_init(elev=30, azim=angle)
                                        #  filename = f"frame_{angle}.png"
                                        # plt.savefig(filename)
                                        #filenames.append(filename)

                                    # Create GIF
                                    #images = [imageio.imread(f) for f in filenames]
                                    #imageio.mimsave('3d_plot_rotation.gif', images, duration=0.1)
                                    
                                    plt.savefig(name+"-"+formulation+"-TRAINscatter.svg")
                                    plt.close()
                                    #plt.show()
                                """
                                
                                ####################
                                TRAIN_metrics = zip(*[f.analyze_instance(W_orig_train[n,:],
                                                                        W_re[X_re[n,:].argmax(),:]) 
                                                        for n in range(N)])
                                TRAIN_equal_maxs, TRAIN_rmse, TRAIN_rho, TRAIN_tau, TRAIN_cos_sim, TRAIN_emd = TRAIN_metrics
                                TRmean_equal_maxs, TRmean_rmse, TRmean_rho, TRmean_tau, TRmean_cos_sim, TRmean_emd = [np.mean(m).round(2) for m in [TRAIN_equal_maxs, TRAIN_rmse, TRAIN_rho, TRAIN_tau, TRAIN_cos_sim, TRAIN_emd]]
                                print("MEAN TRAIN performance metrics: (equal_maxs, rmse, rho, tau, cos_sim, emd): ",
                                    TRmean_equal_maxs, TRmean_rmse, TRmean_rho, TRmean_tau, TRmean_cos_sim, TRmean_emd)

                                
                                #EVALUATING OUT SAMPLE
                                X_test, _ = cluster_assignments(u.X, dataset_test, A, list_bn_test, groups_N_test, considered_groups,
                                                                delete_cluster = delete_cluster)
                                clusters_test = [np.argmax(x) for x in X_test]
                                
                                TEST_metrics = zip(*[f.analyze_instance(W_orig_test[n,:], 
                                                                        W_re[clusters_test[n],:]) 
                                                        for n in range(N_test)])
                                TEST_equal_maxs, TEST_rmse, TEST_rho, TEST_tau, TEST_cos_sim, TEST_emd = TEST_metrics
                                TSmean_equal_maxs, TSmean_rmse, TSmean_rho, TSmean_tau, TSmean_cos_sim, TSmean_emd = [np.mean(m).round(2) for m in [TEST_equal_maxs, TEST_rmse, TEST_rho, TEST_tau, TEST_cos_sim, TEST_emd]]
                                print("MEAN TEST performance metrics: (equal_maxs, rmse, rho, tau, cos_sim, emd): ",
                                    TSmean_equal_maxs, TSmean_rmse, TSmean_rho, TSmean_tau, TSmean_cos_sim, TSmean_emd)
                                
                                #real vs. estimated instability
                                comparison_instabs = {'real': [], 'estimated': [], 'dif': []}
                                for n in range(N):
                                    W_s = {'estimated':W_re[X_re[n,:].argmax(),:], 'real': W_orig_train[n,:]}
                                    for vect in ['estimated','real']:
                                        W_chosen = W_s[vect]
                                        linear_obj = W_chosen@C
                                        inst = linear_obj@(dataset_train[n,:] - diet.original_problem(linear_obj, A, list_bn_train[n], 0))
                                        comparison_instabs[vect].append(inst)
                                    comparison_instabs['dif'].append( comparison_instabs['real'][-1] - comparison_instabs['estimated'][-1])
                                print("real instab - estimated instab: (mean,std) ", (np.mean(comparison_instabs['dif']),np.std(comparison_instabs['dif'])))
                                
                                TEST_comparison_instabs = {'real': [], 'estimated': [], 'dif': []}
                                for n in range(N_test):
                                    W_s = {'estimated':W_re[clusters_test[n],:], 'real': W_orig_test[n,:]}
                                    for vect in ['estimated','real']:
                                        W_chosen = W_s[vect]
                                        linear_obj = W_chosen@C
                                        inst = linear_obj@(dataset_test[n,:] - diet.original_problem(linear_obj, A, list_bn_test[n], 0))
                                        TEST_comparison_instabs[vect].append(inst)
                                    TEST_comparison_instabs['dif'].append( TEST_comparison_instabs['real'][-1] - TEST_comparison_instabs['estimated'][-1])
                                print("real instab - estimated instab: (mean,std) ", (np.mean(TEST_comparison_instabs['dif']),np.std(TEST_comparison_instabs['dif'])))
                                
                                """
                                fig1, ax1 = plt.subplots()
                                ax1.set_title('TEST COMPARISON INSTABS')
                                ax1.boxplot(TEST_comparison_instabs['dif'])
                                plt.savefig(name+"-"+formulation+"-TESTboxplot.svg")
                                plt.close()
                                #plt.show()
                                """

                                sol_outputs = {'seed': seed,
                                            'N': N,
                                            'n_vars_perturbed': n_vars_perturbed,
                                            'instances': instances,
                                            'K': K,
                                            'd': d,
                                            'interpretability': interpretability,
                                            'L': L, 
                                            'heuristic': heur_outputs,
                                            'best_obj_val': obj_val_heur, 
                                            'best_iter': best_iter,
                                            'time_in_heur': time_in_heur,
                                            'warmstart': warmstart,
                                            #'output': output,
                                            'optimality': model.status == 2,
                                            'time': round(time_sol,3),
                                            'gap': gap,
                                            'solution_log': solution_log, 
                                            'progress_loss': progress_loss,
                                            'obj_val': obj_val_solver,
                                            'print_W': str(W_re.round(2)),
                                            'W': np.round(W_re,3),
                                            'X': X_re.astype(int),
                                            'X_composition': X_re.sum(axis = 0),
                                            'L_efective': L_efective,
                                            'delete_clusters': delete_cluster,
                                            'clusters_test': clusters_test,
                                            'TRcomparison_instabs': comparison_instabs,
                                            'TRcomparison_instabs_mean': np.mean(comparison_instabs['dif']),
                                            'TRAIN_metrics': TRAIN_metrics,
                                            'TRmean_equal_maxs': TRmean_equal_maxs,
                                            'TRmean_rmse': TRmean_rmse,
                                            'TRmean_rho': TRmean_rho,
                                            'TRmean_tau': TRmean_tau,
                                            'TRmean_cos_sim': TRmean_cos_sim,
                                            'TRmean_emd': TRmean_emd,
                                            'TScomparison_instabs': TEST_comparison_instabs,
                                            'TScomparison_instabs_mean': np.mean(TEST_comparison_instabs['dif']),
                                            'TEST_metrics': TEST_metrics,
                                            'TSmean_equal_maxs': TSmean_equal_maxs,
                                            'TSmean_rmse': TSmean_rmse,
                                            'TSmean_rho': TSmean_rho,
                                            'TSmean_tau': TSmean_tau,
                                            'TSmean_cos_sim': TSmean_cos_sim,
                                            'TSmean_emd': TSmean_emd
                                            }

                                with open('soloutput-'+str(interpretability)+'-'+name+'.pkl', 'wb') as handle:
                                    pickle.dump(sol_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
