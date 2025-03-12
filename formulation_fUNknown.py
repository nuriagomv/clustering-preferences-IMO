import gurobipy as gp
import numpy as np
import time


def problem_fUNknown(predef_W, 
                     N, L, dataset, A, list_bn, groups_N, considered_groups, 
                     timelimit=None, silence = False, linearize_product=False,
                     predef_C = None, known_objs = None, lambd = 1.,
                     predef_u = None, predef_X = None,
                     warmstart = None,
                     call_callback=False,
                     mipgapabs = None):

    _,d = dataset.shape
    m,_ = A.shape
    
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
            print("best_bound:", best_bound)
            print("obj_val:",obj_val)
            mip_gap = (abs(best_bound - obj_val) / abs(obj_val)) if obj_val != 0 else float('inf')  # Compute MIP gap
            my_mip_gap = (abs(best_bound - obj_val) / (abs(obj_val)+1)) #case almost 0
            
            # Extract variable values
            var_values = {var.VarName: model.cbGetSolution(var) for var in model.getVars()}
            
            # Print information dynamically
            #var_str = ", ".join([f"{k}={v:.4f}" for k, v in var_values.items() if "W" in k])
            #print(f"Feasible Solution Found - Time: {runtime:.2f}s, Objective: {obj_val:.4f}, Gap: {mip_gap:.4%}, Solution: {var_str}")
            C_feas = np.array([v for k, v in var_values.items() if "C" in k]).reshape((K,d))
            X_feas = np.array([v for k, v in var_values.items() if "X" in k]).reshape((N,L))
            print(f"Feasible Solution Found - Time: {runtime:.2f}s, Objective: {obj_val:.4f}, Gap: {my_mip_gap:.4%}, \nSolution C_feas: \n{C_feas.round(3)}\n Cluster composition: {X_feas.sum(axis=0)}")
            
            # Store in a structured format
            solution_log.append({"Time (s)": runtime, 
                                 "Objective": obj_val, 
                                 "Gap (%)": mip_gap * 100,
                                 "my gap (%)": my_mip_gap *100,
                                 "C_feas": C_feas,
                                 "X_feas": X_feas,
                                 **var_values  # Merge variable values into the dictionary
                                 })
            if my_mip_gap <= 1e-4:
                model.terminate()
            
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

    if mipgapabs is not None:
        model.setParam("MIPGapAbs", mipgapabs)

    if silence:
        model.setParam('OutputFlag', 0)
    
    if predef_C is None:
        C = model.addMVar(shape=(K,d), lb=-1, ub=1., vtype=gp.GRB.CONTINUOUS, name="C")
        c0 = model.addMVar(shape=K, 
                           lb=-float('inf'), ub=+float('inf'),
                           vtype=gp.GRB.CONTINUOUS, name="c0")
        S = model.addMVar(shape=(K,d),
                          vtype=gp.GRB.BINARY, name="S")
        P = model.addMVar(shape=(K,d),
                          lb=0., ub=1.,
                          vtype=gp.GRB.CONTINUOUS, name="P")
        model.addConstrs((2*P[k,j] - C[k,j] >= 0 
                          for k in range(K) for j in range(d)), name="define_absval")
        model.addConstrs((gp.quicksum(2*P[k,j] - C[k,j] for j in range(d)) == 1
                          for k in range(K)), name="normalization")

        model.addConstrs((P[k,j] <= S[k,j]
                        for k in range(K) for j in range(d)), name="define_P1")
        model.addConstrs((P[k,j] <=  C[k,j] + (1-S[k,j])
                        for k in range(K) for j in range(d)), name="define_P2")
        model.addConstrs((P[k,j] >=  C[k,j] - (1-S[k,j])
                        for k in range(K) for j in range(d)), name="define_P3")
    else:
        C,c0 = predef_C

    if predef_X is None:
        X = model.addMVar(shape=(N,L), vtype=gp.GRB.BINARY, name="X")
        model.addConstrs((gp.quicksum(X[n,l] for l in range(L)) == 1
                          for n in range(N)),
                          name="all instances belong only to one cluster")
    else:
        X = predef_X
    
    W = predef_W
    if L > L_upto:
        #INTERPRETABILITY FROM CATALOGUE 
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
        
    def linreg(known_objs):
        if known_objs is None:
            return 0
        else:
            return gp.quicksum( (known_objs[n][k] - C[k,:]@dataset[n,:])**2#(known_objs[n][k] - c0[k] - C[k,:]@dataset[n,:])**2
                               for n in known_objs.keys() for k in range(K))

    model.setObjective(lambd * gp.quicksum(term(l,n) for l in range(L) for n in range(N)) + (1-lambd)*linreg(known_objs),
                       sense=gp.GRB.MINIMIZE)
    #model.addConstrs((term(l,n) >= 0 for l in range(L) for n in range(N)))
    model.addConstr(linreg(known_objs) >= 0)
    

    model.update()

    if timelimit is not None:
        model.setParam(gp.GRB.Param.TimeLimit, timelimit)
    
    if warmstart is not None:
        (X_heur,(C_heur,c0_heur),u_heur) = warmstart
        X.Start, C.Start, c0.Start, u.Start = X_heur, C_heur, c0_heur, u_heur
    
    if call_callback:
        model.optimize(my_callback)
    else:
        model.optimize()
    
    return model, C,c0, X, u, solution_log, progress_logs
