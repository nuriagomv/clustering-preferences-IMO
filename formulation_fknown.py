import gurobipy as gp
import numpy as np
import time


def problem_fknown(C, N, L, dataset, A, list_bn, groups_N, considered_groups, timelimit=None, 
                   silence = False, break_sym=False, linearize_product=False,
                   predef_W = None, predef_u = None, predef_X = None, warmstart = None):

    K,d = C.shape
    m,_ = A.shape
    if predef_W is not None:
        L_upto = L
        L,K = predef_W.shape
    
    # List to store feasible solutions
    solution_log = []
    count_checks = 0
    progress_logs = [(0., float('inf'), 0.)]
    saved_best_time, saved_best_objval, saved_best_bound = None, float('inf'), 0.
    def my_callback(model, where):

        global count_checks

        if where == gp.GRB.Callback.MIPSOL:  # When a new feasible solution is found

            obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)  # Objective value
            runtime = model.cbGet(gp.GRB.Callback.RUNTIME)  # Time when found
            gap = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST)  # Best bound so far
            best_bound = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)  # Current best bound
            mip_gap = abs(best_bound - obj_val) / abs(obj_val) if obj_val != 0 else float('inf')  # Compute MIP gap
            
            # Extract variable values
            var_values = {var.VarName: model.cbGetSolution(var) for var in model.getVars()}
            
            # Print information dynamically
            #var_str = ", ".join([f"{k}={v:.4f}" for k, v in var_values.items() if "W" in k])
            #print(f"Feasible Solution Found - Time: {runtime:.2f}s, Objective: {obj_val:.4f}, Gap: {mip_gap:.4%}, Solution: {var_str}")
            W_feas = np.array([v for k, v in var_values.items() if "W" in k]).reshape((L,K))
            X_feas = np.array([v for k, v in var_values.items() if "X" in k]).reshape((N,L))
            print(f"Feasible Solution Found - Time: {runtime:.2f}s, Objective: {obj_val:.4f}, Gap: {mip_gap:.4%}, \nSolution W_feas: \n{W_feas.round(3)}\n Cluster composition: {X_feas.sum(axis=0)}")
            
            # Store in a structured format
            solution_log.append({"Time (s)": runtime, 
                                 "Objective": obj_val, 
                                 "Gap (%)": mip_gap * 100,
                                 "W_feas": W_feas,
                                 "X_feas": X_feas,
                                 **var_values  # Merge variable values into the dictionary
                                 })
            
        elif where == gp.GRB.Callback.MIP: # General MIP progress

            runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
            if runtime - progress_logs[-1][0] >5:
                #print("ENTRO EN SEGUNDO BUCLE COMPROBACION: ", runtime - progress_logs[-1][0])
                best_sol = model.cbGet(gp.GRB.Callback.MIP_OBJBST)  # Best found feasible solution
                best_bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)  # Current best bound
                
                # Stop criteria
                if best_sol < progress_logs[-1][1]:
                    #print("improving obj val")
                    count_checks = 0
                elif best_bound > progress_logs[-1][2]:
                    #print("improving bound")
                    count_checks = 0
                else:
                    count_checks +=1
                    #print("not improving")
                progress_logs.append((runtime,best_sol,best_bound))
                
                if count_checks > 100 :
                    print("STOPPING BECAUSE THERE IS NO IMPROVEMENT")
                    model.terminate()  # Force stop Gurobi


    model = gp.Model()

    if silence:
        model.setParam('OutputFlag', 0)
    
    if predef_X is None:
        X = model.addMVar(shape=(N,L), vtype=gp.GRB.BINARY, name="X")
        model.addConstrs((gp.quicksum(X[n,l] for l in range(L)) == 1
                          for n in range(N)),
                          name="all instances belong only to one cluster")
    else:
        X = predef_X
    
    if predef_W is None:
        W = model.addMVar(shape=(L,K), lb=0, ub=1., vtype=gp.GRB.CONTINUOUS, name="W")
        model.addConstrs((gp.quicksum(W[l,k] for k in range(K)) == 1
                          for l in range(L)),
                          name="weights' sum is one")
        if break_sym:
            model.addConstrs((gp.quicksum(k*W[l,k] for k in range(K)) <= gp.quicksum(k*W[l+1,k] for k in range(K))
                              for l in range(L-1)), name="symmetry break")
    else:
        W = predef_W
        if L > L_upto:
            #INTERPRETABILITY FROM CATALOGUE 
            None #COMPLETE CASE INTERPRETABILITY
            y = model.addMVar(shape=(L), vtype=gp.GRB.BINARY, name="Y")
            model.addConstrs((y[l] >= X[n,l] for n in range(N) for l in range(L)), name="catalogue_active")
            model.addConstr(gp.quicksum(y[l] for l in range(L))<=L_upto, name="chosen_from_catalogue")
    
    different_Z_cal = len(considered_groups)
    if predef_u is None:
        u = model.addMVar(shape=(different_Z_cal,L,m),
                          lb=-float('inf'), ub=0., vtype=gp.GRB.CONTINUOUS, name="u")
        model.addConstrs((u[feas_reg,l,:]@A == W[l,:]@C
                    for feas_reg in range(different_Z_cal) for l in range(L)), name="dual_constr")
    else:
        u = predef_u
       
    #COMPUTATIONAL ENHANCENMENT ?
    if linearize_product:
        v = model.addMVar(shape=(N,L,m),
                        lb=-float('inf'), ub=0., vtype=gp.GRB.CONTINUOUS, name="v")
        model.addConstrs((v[n,l,i] == X[n,l] * u[considered_groups.index(groups_N[n]),l,i]
                            for n in range(N) for l in range(L) for i in range(m)), name="realocate_product")
    
    def term(l,n):
        if linearize_product:
            return v[n,l,:] @ (A@dataset[n,:] - list_bn[n])
        else:
            return X[n,l] * u[considered_groups.index(groups_N[n]),l,:] @ (A@dataset[n,:] - list_bn[n])

    model.setObjective(gp.quicksum(term(l,n) for l in range(L) for n in range(N)),
                       sense=gp.GRB.MINIMIZE)

    if warmstart is not None:
        X.Start, W.Start, u.Start = warmstart
    
    model.update()

    if timelimit is not None:
        model.setParam(gp.GRB.Param.TimeLimit, timelimit)
    
    if (predef_W is None) and (predef_X is None):
        model.optimize(my_callback)
    else:
        model.optimize()
    
    return model, W, X, u, solution_log, progress_logs
