import numpy as np
import csv
import redis
import pickle
from util import Veh_parameter,HDV_dynamics,measure_mixed_traffic,\
    Tstep,total_time_step,ID,pos_cav,n_vehicle,n_cav,hdv_type,acel_noise,\
    T,Tini,N

# 是否进行物理增强
Phy_flag = 1

# 对每个以CAV为首的子系统进行初始化
def init_sub():
    # redis settings
    host = '8.130.125.99'
    redis_port = 6379
    db = 2
    password = 'chlpw1039'
    rs = redis.Redis(host=host, port=redis_port, password=password,db=db)

    # cav parameter
    cav_id = np.arange(0,n_cav)
    # 通信端口
    Port = np.arange(3389,3389+n_cav)

    # Construction of this subsystem
    n_vehicle_sub = [[] for _ in range(n_cav)]
    ID_sub = [[] for _ in range(n_cav)]
    for i in range(n_cav):
        if cav_id[i] != n_cav-1:
            n_vehicle_sub[i] = pos_cav[cav_id[i]+1]-pos_cav[cav_id[i]] # number of vehicles in the subsystem
            ID_sub[i] = ID[pos_cav[cav_id[i]]:pos_cav[cav_id[i]+1]]
        else:
            n_vehicle_sub[i] = n_vehicle - pos_cav[cav_id[i]]
            ID_sub[i] = ID[pos_cav[cav_id[i]]:]

    # Vehicle Parameter Initialization
    s_st = 5
    s_star = 20
    v_max = 30
    v_star = 15
    data = []
    with open('data/hdv_ovm_random_moderate.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
            
    header = data[0]  # 表头在第1行
    data = np.array(data[1:])

    alpha_total   = np.array([eval(i) for i in data[:,header.index('alpha')]])
    beta_total    = np.array([eval(i) for i in data[:,header.index('beta')]])
    s_go_total    = np.array([eval(i) for i in data[:,header.index('s_go')]])
    System_para_mod_total = Veh_parameter(hdv_type,alpha_total.squeeze(),beta_total.squeeze(),s_st,s_go_total.squeeze(),v_max,s_star,v_star)
    Subsystem_para_mod = [[] for _ in range(n_cav)]

    for i in range(n_cav):
        alpha   = np.array([eval(i) for i in data[pos_cav[cav_id[i]]:pos_cav[cav_id[i]]+n_vehicle_sub[i],header.index('alpha')]])
        beta    = np.array([eval(i) for i in data[pos_cav[cav_id[i]]:pos_cav[cav_id[i]]+n_vehicle_sub[i],header.index('beta')]])
        s_go    = np.array([eval(i) for i in data[pos_cav[cav_id[i]]:pos_cav[cav_id[i]]+n_vehicle_sub[i],header.index('s_go')]])
        Subsystem_para_mod[i] = Veh_parameter(hdv_type,alpha.squeeze(),beta.squeeze(),s_st,s_go.squeeze(),v_max,s_star,v_star)

    # Initial Data's Hankel Matrix
    # load trajectory data (precollected data: 300 time steps)
    # calculate KKTvert and Hzvert
    weight_v        = 1     # weight coefficient for velocity error
    weight_s        = 0.5   # weight coefficient for spacing error   
    weight_u        = 0.1   # weight coefficient for control input
    K = [[] for _ in range(n_cav)]
    P = [[] for _ in range(n_cav)]
    Qi = [[] for _ in range(n_cav)]
    Ri = [[] for _ in range(n_cav)]
    Qi_stack = [[] for _ in range(n_cav)]
    Ri_stack = [[] for _ in range(n_cav)]
    
    # matrix for physics-augmented calculation
    weight_delta1   = 0.5*1e12
    weight_delta2   = 0.5*1e12  # weight coefficient for physics laws
    M1 = weight_delta1 * np.eye(N-1)
    M2 = weight_delta2 * np.eye(N-1)
    F = [[] for _ in range(n_cav)]
    Q_delta1 = [[] for _ in range(n_cav)]
    Q_delta2 = [[] for _ in range(n_cav)]
    Bf = np.hstack((np.eye(N-1),np.zeros([N-1,1])))  # Bf is a N-1*N matrix, Bf*v=v[0:-1]
    Af = np.hstack((np.zeros([N-1,1]),np.eye(N-1)))  # Af is a N-1*N matrix, Af*v=v[1:]
    DM = Af - Bf                                     # Diff_matrix DM*v = v[1:] - v[0:-1]

    for i in range(n_cav):
        Ui_temp = np.genfromtxt('data/precollected_moderate/Ui_'+ str(cav_id[i]) + '_moderate.csv', delimiter=",", skip_header=0)
        Uip = Ui_temp[0:Tini,:]
        Uif = Ui_temp[Tini:,:]
        rs.mset({f'Uip_in_CAV_{cav_id[i]}':pickle.dumps(Uip)})
        rs.mset({f'Uif_in_CAV_{cav_id[i]}':pickle.dumps(Uif)})

        Ei_temp = np.genfromtxt('data/precollected_moderate/Ei_'+ str(cav_id[i]) + '_moderate.csv', delimiter=",", skip_header=0)
        Eip = Ei_temp[0:Tini,:]
        Eif = Ei_temp[Tini:,:]
        rs.mset({f'Eip_in_CAV_{cav_id[i]}':pickle.dumps(Eip)})
        rs.mset({f'Eif_in_CAV_{cav_id[i]}':pickle.dumps(Eif)})

        Yi_temp = np.genfromtxt('data/precollected_moderate/Yi_'+ str(cav_id[i]) + '_moderate.csv', delimiter=",", skip_header=0)
        linenum = int(Tini*(n_vehicle_sub[i]+1))
        Yip = Yi_temp[0:linenum,:]
        Yif = Yi_temp[linenum:,:]
        rs.mset({f'Yip_in_CAV_{cav_id[i]}':pickle.dumps(Yip)})
        rs.mset({f'Yif_in_CAV_{cav_id[i]}':pickle.dumps(Yif)})

        # Initial dual variables
        g_initial = np.zeros(T-Tini-N+1)
        mu_initial = np.zeros(T-Tini-N+1)
        eta_initial = np.zeros(N)
        phi_initial = np.zeros(N)
        theta_initial = np.zeros(N)

        rs.mset({f'g_initial_in_CAV_{cav_id[i]}':pickle.dumps(g_initial)})
        rs.mset({f'mu_initial_in_CAV_{cav_id[i]}':pickle.dumps(mu_initial)})
        rs.mset({f'eta_initial_in_CAV_{cav_id[i]}':pickle.dumps(eta_initial)})
        rs.mset({f'phi_initial_in_CAV_{cav_id[i]}':pickle.dumps(phi_initial)})
        rs.mset({f'theta_initial_in_CAV_{cav_id[i]}':pickle.dumps(theta_initial)})

        lambda_g        = 10    # penalty on ||g||_2^2 in objective
        lambda_y        = 1e4   # penalty on ||sigma_y||_2^2 in objective
        rho             = 1     # penality parameter in ADMM
        lambda_gi = lambda_g/n_cav
        lambda_yi = lambda_y
        rs.mset({f'rho_in_CAV_{cav_id[i]}':pickle.dumps(rho)})
        rs.mset({f'lambda_gi_in_CAV_{cav_id[i]}':pickle.dumps(lambda_gi)})
        rs.mset({f'lambda_yi_in_CAV_{cav_id[i]}':pickle.dumps(lambda_yi)})
    
        K[i] = np.kron(np.eye(N),np.append(np.zeros(int(n_vehicle_sub[i]-1)),[1,0]))
        P[i] = np.kron(np.eye(N),np.append(np.zeros(int(n_vehicle_sub[i])),[1]))

        Qi[i] = np.diagflat(np.append(weight_v*np.ones(int(n_vehicle_sub[i])),[weight_s]))
        Qi_stack[i] = np.kron(np.eye(N),Qi[i])
        Ri[i] = weight_u
        Ri_stack[i] = np.kron(np.eye(N),Ri[i])
        rs.mset({f'Qi_in_CAV_{cav_id[i]}':pickle.dumps(Qi_stack[i])})
        rs.mset({f'Ri_in_CAV_{cav_id[i]}':pickle.dumps(Ri_stack[i])})

        # physics augmented
        F[i] = np.kron(np.eye(N),np.append([1],np.zeros(int(n_vehicle_sub[i])))) 
        Q_delta1[i] = DM@P[i]@Yif - Tstep*Bf@(Eif-F[i]@Yif)
        Q_delta2[i] = DM@F[i]@Yif - Tstep*Bf@Uif

        if Phy_flag == 0:
            # KKT_vert
            if cav_id[i] == 0: # the first subsystem has different Hgi
                Hg = Yif.T@Qi_stack[i]@Yif+Uif.T@Ri_stack[i]@Uif+\
                    lambda_gi*np.eye(T-Tini-N+1)+lambda_yi*Yip.T@Yip+\
                    rho/2*(np.eye(T-Tini-N+1)+Yif.T@P[i].T@P[i]@Yif+Uif.T@Uif)
                Aeqg = np.vstack((Uip,Eip,Eif))
            else:
                Hg = Yif.T@Qi_stack[i]@Yif+Uif.T@Ri_stack[i]@Uif+\
                    lambda_gi*np.eye(T-Tini-N+1)+lambda_yi*Yip.T@Yip+\
                    rho/2*(np.eye(T-Tini-N+1)+Eif.T@Eif+Yif.T@P[i].T@P[i]@Yif+Uif.T@Uif)
                Aeqg = np.vstack((Uip,Eip))

            Phi = np.vstack((np.hstack((Hg,Aeqg.T)),np.hstack((Aeqg,np.zeros([Aeqg.shape[0],Aeqg.shape[0]])))))
            KKT_vert = np.linalg.inv(Phi)
        else:
            if cav_id[i] == 0: # the first subsystem has different Hgi
                Hg = Yif.T@Qi_stack[i]@Yif+Uif.T@Ri_stack[i]@Uif+\
                    lambda_gi*np.eye(T-Tini-N+1)+lambda_yi*Yip.T@Yip+\
                    Q_delta2[i].T@M2@Q_delta2[i]+\
                    rho/2*(np.eye(T-Tini-N+1)+Yif.T@P[i].T@P[i]@Yif+Uif.T@Uif)
                Aeqg = np.vstack((Uip,Eip,Eif))
            else:
                Hg = Yif.T@Qi_stack[i]@Yif+Uif.T@Ri_stack[i]@Uif+\
                    lambda_gi*np.eye(T-Tini-N+1)+lambda_yi*Yip.T@Yip+\
                    Q_delta1[i].T@M1@Q_delta1[i] + Q_delta2[i].T@M2@Q_delta2[i]+\
                    rho/2*(np.eye(T-Tini-N+1)+Eif.T@Eif+Yif.T@P[i].T@P[i]@Yif+Uif.T@Uif)
                Aeqg = np.vstack((Uip,Eip))
    
            Phi = np.vstack((np.hstack((Hg,Aeqg.T)),np.hstack((Aeqg,np.zeros([Aeqg.shape[0],Aeqg.shape[0]])))))
            KKT_vert = np.linalg.inv(Phi)

        rs.mset({f'KKT_vert_in_CAV_{cav_id[i]}':pickle.dumps(KKT_vert)})


        # Hz_vert
        if cav_id[i] != n_cav-1:
            Hz = rho/2*np.eye(T-Tini-N+1)+rho/2*Yif.T@K[i].T@K[i]@Yif
        else:
            Hz = rho/2*np.eye(T-Tini-N+1)
        Hz_vert = np.linalg.inv(Hz)
        rs.mset({f'Hz_vert_in_CAV_{cav_id[i]}':pickle.dumps(Hz_vert)})
    
    S_total = np.zeros([total_time_step+Tini,n_vehicle+1,3])
    S_total[0,0,0] = 0 # initial position of the platoon
    for i in range(1,n_vehicle+1):
        S_total[0,i,0] = S_total[0,i-1,0] - s_star # initial position
    
    S_total[0,:,1] = v_star * np.ones(n_vehicle+1) # initial velocity = v*

    # Initial trajectory
    uini = np.zeros(Tini)
    eini = np.zeros(Tini)
    yini = np.zeros([n_vehicle+n_cav,Tini])
    Su = np.zeros(total_time_step+Tini)
    Su[0:Tini] = uini

    for k in range(Tini-1):
        # update acceleration
        acel = HDV_dynamics(S_total[k,:,:],System_para_mod_total)+\
            acel_noise*(-1+2*np.random.random())
        
        S_total[k,0,2] = 0                # the head vehicle has 0 acc
        S_total[k,1:,2] = acel            # all the vehicles using HDV model
        S_total[k,pos_cav+1,2] = uini[k]          # the CAVs

        S_total[k+1,:,1] = S_total[k,:,1] + Tstep*S_total[k,:,2]
        S_total[k+1,0,1] = eini[k] + v_star           # velocity of the head vehicles
        S_total[k+1,:,0] = S_total[k,:,0] + Tstep*S_total[k,:,1]  # update position

        yini[:,k] = measure_mixed_traffic(S_total[k,1:,1],S_total[k,:,0],ID,v_star,s_star)

    k = k+1
    yini[:,k] = measure_mixed_traffic(S_total[k,1:,1],S_total[k,:,0],ID,v_star,s_star)



    args_list = [[] for _ in range(n_cav)]
    for i in range(n_cav):
        if i == 0:
            ei_ini = eini
        else:
            ei_ini = S_total[0:Tini,pos_cav[i],1] - v_star
        
        yi_ini = np.zeros([n_vehicle_sub[i]+1,Tini])
        if i != n_cav-1:
            yi_ini[0:-1,:] = yini[pos_cav[i]:pos_cav[i+1],:]
        else:
            yi_ini[0:-1,:] = yini[pos_cav[i]:n_vehicle,:]
        yi_ini[-1,:] = yini[n_vehicle+i,:]

        S = np.zeros([total_time_step+Tini,n_vehicle_sub[i]+1,3])
        if i == 0:
            S = S_total[:,0:n_vehicle_sub[i]+1,:]
        else:
            S = S_total[:,pos_cav[i]:pos_cav[i]+n_vehicle_sub[i]+1,:]

        args_list[i] = [Port[i],ID_sub[i],cav_id[i],n_vehicle_sub[i],Subsystem_para_mod[i],S,uini,ei_ini,yi_ini,Su]

    return args_list