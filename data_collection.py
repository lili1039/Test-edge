import numpy as np
import csv
import random
import math
from util import Veh_parameter,Hankel_matrix,HDV_dynamics,measure_mixed_traffic,T,Tini,N,Tstep,total_time_step

if __name__=="__main__":
    # how many data sets to collect?
    # data_total_number = 100
    data_total_number = 1
    # Type for HDV car-following model
    hdv_type        = 1    # 1.OVM    2.IDM
    # Uncertainty for HDV behavior
    acel_noise      = 0.1  # A white noise signal on HDV's original acceleration
    # Data set
    data_str        = '1'  # 1. random ovm  2. manual ovm  3. homogeneous ovm

    # Parameters in mixed traffic
    ID = np.array([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0])  # ID of vehicle types 1: CAV  0: HDV
    # ID_str      = ''.join(str(i) for i in ID)

    pos_cav     = np.where(ID == 1)[0]         # position of CAVs
    # pos_cav = [ 0  3  6  9 12]
    n_vehicle   = len(ID)                      # number of vehicles
    n_cav       = len(pos_cav)                 # number of CAVs
    n_hdv       = n_vehicle-n_cav              # number of HDVs

    v_star      = 15                           # Equilibrium velocity
    # s_star      = 20                           # Equilibrium spacing for CAV
    acel_max = 2                               # max accelaration
    dcel_max = -5                              # min accelaration

    # Velocity error and spacing error of the CAVs are measurable, 
    # and the velocity error of the HDVs are measurable.

    # size in DeePC-LCC
    n_ctr = 2*n_vehicle         # number of state variables
    m_ctr = n_cav               # number of input variables u(t)
    p_ctr = n_vehicle + n_cav   # number of output variables

    ni_vehicle = np.zeros(n_cav) # number of vehicles in each LCC subsystem

    for i_cav in range(n_cav-1):
        ni_vehicle[i_cav] = int(pos_cav[i_cav+1] - pos_cav[i_cav])

    ni_vehicle[n_cav-1] = int(n_vehicle - pos_cav[-1])

    # Scenario initialization
    s_st = 5
    v_max = 30
    s_star = 20
    data = []
    with open('data/hdv_ovm_random_moderate.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
            
    header = data[0]  # 表头在第1行
    data = np.array(data[1:])

    alpha   = np.array([eval(i) for i in data[:,header.index('alpha')]])
    beta    = np.array([eval(i) for i in data[:,header.index('beta')]])
    s_go    = np.array([eval(i) for i in data[:,header.index('s_go')]])

    HDVpara_moderate = Veh_parameter(hdv_type,alpha.squeeze(),beta.squeeze(),s_st,s_go.squeeze(),v_max,s_star,v_star)

    # There is one head vehicle at the very beginning
    # S: all times all cars(including the head), distance error/ velocity error/ accelaration
    S = np.zeros([total_time_step,n_vehicle+1,3]) 
    S[0,0,0] = 0
    for i in range(1,n_vehicle+1):
        S[0,i,0] = S[0,i-1,0] - s_star # initial position

    S[0,:,1] = v_star * np.ones(n_vehicle+1) # initial velocity = v*

    # Data collection
    # persistently exciting input data
    ud = -1+2*np.random.random([m_ctr,T]) # 生成-1到1的 维度为CAV数*数据长度 的随机数数组
    ed = -1+2*np.random.random(T)  # -1到1
    yd = np.zeros([p_ctr,T]) 

    # generate output data
    for k in range(T-1):
        acel = HDV_dynamics(S[k,:,:],HDVpara_moderate) - \
            acel_noise*(-1+2*np.random.random([1,n_vehicle]))

        S[k,0,2] = 0                # the head vehicle has 0 acc
        S[k,1:,2] = acel            # all the vehicles using HDV model
        S[k,pos_cav+1,2] = ud[:,k]  # the CAVs

        S[k+1,:,1] = S[k,:,1] + Tstep*S[k,:,2]
        S[k+1,0,1] = ed[k] + v_star             # velocity of the head vehicles, which is random
        S[k+1,:,0] = S[k,:,0] + Tstep*S[k,:,1]  # update position

        yd[:,k] = measure_mixed_traffic(S[k,1:,1],S[k,:,0],ID,v_star,s_star)

    k = k+1
    yd[:,k] = measure_mixed_traffic(S[k,1:,1],S[k,:,0],ID,v_star,s_star)

    # construct distributed data
    ui_d = []
    yi_d = []
    ei_d = []

    for i in range(n_cav):
        ui_d.append(np.array([ud[i,:]]))
        
        if i != n_cav-1:
            yi_d.append(yd[pos_cav[i]:pos_cav[i+1],:])
        else:
            yi_d.append(yd[pos_cav[i]:n_vehicle,:])
        
        # add one row: the spacing error of the CAV in the subsystem
        yi_d[i] = np.append(yi_d[i],np.array([yd[n_vehicle+i,:]]),axis = 0)

        if i == 0:                          # the first subsystem
            ei_d.append(np.array([ed]))     # velocity error of the head
        else:
            ei_d.append(np.array([yd[pos_cav[i]-1,:]])) # velocity of the veh before the subsystem

    # data Hankel matrices
    # for distributed DeePLCC
    Ui = [] # 创建一个列表
    Uip = []
    Uif = []
    Ei = []
    Eip = []
    Eif = []
    Yi = []
    Yip = []
    Yif = []

    for i in range(n_cav):
        Ui.append(Hankel_matrix(ui_d[i],Tini+N))
        Ei.append(Hankel_matrix(ei_d[i],Tini+N))
        Yi.append(Hankel_matrix(yi_d[i],Tini+N))

        np.savetxt('data/precollected_moderate/Ui_'+ str(i) + '_moderate.csv', Ui[i], fmt='%.6f', delimiter=',')
        np.savetxt('data/precollected_moderate/Ei_'+ str(i) + '_moderate.csv', Ei[i], fmt='%.6f', delimiter=',')
        np.savetxt('data/precollected_moderate/Yi_'+ str(i) + '_moderate.csv', Yi[i], fmt='%.6f', delimiter=',')
