import numpy as np
import matplotlib.pyplot as plt
import websockets
import pickle
import time
import asyncio
# import aioping
import random
import math

# cost coefficient
weight_v        = 1     # weight coefficient for velocity error
weight_s        = 0.5   # weight coefficient for spacing error   
weight_u        = 0.1   # weight coefficient for control input

# 参数 
server_ip = '8.130.125.99' # server ip为公网地址，port为通信端口（一个端口对应一个docker）
# Parameter in Simulation
total_time       = 10   # Total Simulation Time 40
Tstep            = 0.05 # Time Step
total_time_step  = int(total_time/Tstep)
# Parameters in mixed traffic
ID = np.array([1,0,0,1,0,0,1,0,0,1,0,0,1,0,0])  # ID of vehicle types 1: CAV  0: HDV
pos_cav     = np.where(ID == 1)[0]         # position of CAVs: pos_cav = [ 0  3  6  9 12]
n_vehicle   = len(ID)                      # number of vehicles
n_cav       = len(pos_cav)                 # number of CAVs
n_hdv       = n_vehicle-n_cav              # number of HDVs
# Type for HDV car-following model
hdv_type        = 1    # 1.OVM    2.IDM
# Uncertainty for HDV behavior
acel_noise      = 0.01  # A white noise signal on HDV's original acceleration
acel_noise_CAV  = 0.05  # A relative noise on CAV's designed acceleration 
# DeeP-LCC Parameters
T = 300             # length of data samples 300
Tini = 20           # length of past data
N = 50              # length of predicted horizon
# max and min accelaration
acel_max = 2 # 2
dcel_max = -5

delay = 0
delay_step = int(delay/Tstep)

class Veh_parameter:
    def __init__(self,type,alpha,beta,s_st,s_go,v_max,s_star,v_star):
        self.size = len(alpha) # number of cars
        self.type = type # car following model: 1. OVM 2. IDM
        self.alpha = alpha
        self.beta = beta
        self.s_st = s_st
        self.s_go = s_go
        self.v_max = v_max
        self.s_star = s_star
        self.v_star = v_star

# Generate a Hankel matrix of order L
def Hankel_matrix(u,L):
    m = u.shape[0]      # the dimension of signal
    T = u.shape[1]      # the length of data
    U = np.zeros([m*L,T-L+1])

    for i in range(L):
        U[i*m:(i+1)*m,:] = u[:,i:(T-L+1+i)]

    return U

# car-following model
# INPUT: S:state of all the vehicles(pos/velo/acc) parameter: para in the model
# OUTPUT: acel of cars
def HDV_dynamics(S,parameter): # 传进来的S是二维的，即k时间的数据
    num_vehicle = S.shape[0] - 1
    # the second dimension size = number of cars (exclude the head one)
    
    if parameter.type == 1: # OVM
        V_diff = S[0:-1,1] - S[1:,1] # the velocity error with former car
        D_diff = S[0:-1,0] - S[1:,0] # the pos error with former car
        
        for i in range(num_vehicle):
            if D_diff[i] > parameter.s_go[i]:
                D_diff[i] = parameter.s_go[i]
            elif D_diff[i] < parameter.s_st:
                D_diff[i] = parameter.s_st
        
        np.seterr(divide = 'ignore') # 当s=sst时，vd的计算过程会出现分母=0的情况，此时忽略报错，将结果记为inf，仍可以参与后续的计算
        acel = parameter.alpha*(parameter.v_max/2*(1-np.cos(math.pi*(D_diff-parameter.s_st)/(parameter.s_go-parameter.s_st)))-S[1:,1])\
                                +parameter.beta*V_diff
        
        # acceleration saturation
        acel = np.where(acel > acel_max, acel_max, acel)
        acel = np.where(acel < dcel_max, dcel_max, acel)

        # SD and ADAS to prevent crash
        acel_sd = (S[1:,1]**2-S[0:-1,1]**2)/2/D_diff
        acel = np.where(acel_sd > abs(dcel_max), dcel_max, acel)

    elif parameter.type == 2: #IDM
        T_gap = 1
        a = 1
        b = 1
        delta = 4

        V_diff = S[0:-1,1] - S[1:,1] # the velocity error with former car
        D_diff = S[0:-1,0] - S[1:,0] # the pos error with former car

        np.seterr(divide = 'ignore')
        acel = a*(1-(S[0:-1,1]/parameter.v_max)**delta-((parameter.s_st+T_gap*S[1:,1]-V_diff*S[1:,1]/2/a**0.5/b**0.5)/D_diff)**2)
        
        # acceleration saturation
        acel = np.where(acel > acel_max, acel_max, acel)
        acel = np.where(acel < dcel_max, dcel_max, acel)

        # SD and ADAS to prevent crash
        acel_sd = (S[1:,1]**2-S[0:-1,1]**2)/2/D_diff
        acel = np.where(acel_sd > abs(dcel_max), dcel_max, acel)

    return acel


# measure output yd
# vel:      velocity of each vehicle
# pos:      position of each vehicle    
# ID:       ID of vehicle types     1: CAV  0: HDV
# v_star:   equilibrium velocity
# s_star:   equilibrium spacing

# Velocity error and spacing error of the CAVs are measurable, 
# and the velocity error of the HDVs are measurable.

def measure_mixed_traffic(vel,pos,ID,v_star,s_star):
    pos_cav     = np.where(ID == 1)[0]  # position of CAVs
    spacing = pos[0:-1] - pos[1:]       # spacing between cars
    y = np.append(vel-v_star,spacing[pos_cav]-s_star)
    
    return y

# =========================================================================
#                   Distributed DeeP-LCC Formulation
#
# Uip --> Eif:                  data hankel matrices
# ui_ini --> ei_ini:            past trajectory before time t
# lambda_gi & lambda_yi:        penalty for the regularization in the cost function
# u_limit & s_limit:            box constraint for control and spacing
# rho:                          penality value for ADMM (Augmented Lagrangian)
# mu_initial --> theta_initial: initial value for variables in ADMM
# KKT_vert:                     pre-calculated inverse matrix for the KKT system
# Hz_vert:                      pre-calculated value of Hz^-1
# =========================================================================

# 连接云: 给每个容器传入子系统的参数信息和初始数据
async def ConnectCloud(Port,k,cav_id,n_vehicle_sub,uini,eini,yini):
    compute_success = False

    # 发送的消息
    msg_send = [cav_id,n_vehicle_sub,uini,eini,yini,k]
    msg_bytes_send = pickle.dumps(msg_send)

    for i in range(5): # 尝试5次
        try:
            # async with websockets.connect(f"ws://{server_ip}:{Port}") as websocket:
            websocket = await asyncio.wait_for(websockets.connect(f"ws://{server_ip}:{Port}"),timeout=50)
            # 发送ini数据
            await websocket.send(msg_bytes_send)
            # 接收握手数据
            msg_bytes_recv = await websocket.recv()
            msg_recv = pickle.loads(msg_bytes_recv)
            # 接收到服务器返回的echo
            if msg_recv == True:
                print(f"Subsystem {cav_id} is loaded.")
            
            # 接收控制输入
            msg_bytes_recv = await websocket.recv()
            msg_recv = pickle.loads(msg_bytes_recv)
            u_cloud         = msg_recv[0]
            real_iter_num   = msg_recv[1]
            use_time        = msg_recv[2]
            y_prediction    = msg_recv[3]
            cost            = msg_recv[4]
            flag = 1
            break

        except asyncio.TimeoutError:
            print(f"Subsystem {cav_id} connect timeout!")
            flag = 0
            # await websocket.close()

    if flag == 0:
        return False,None,None,None,None,None
    else:
        compute_success = True
        await websocket.close()
        return compute_success,u_cloud,real_iter_num,use_time,y_prediction,cost


# 本地进程间通信，发送信息
def SendMessage(msg_send, child_conn):
    msg_bytes_send = pickle.dumps(msg_send)
    child_conn.send(msg_bytes_send)
    return True

# 本地进程间通信，接收信息
def ReceiveMessage(parent_conn):
    msg_bytes_recv = parent_conn.recv()
    msg_recv = pickle.loads(msg_bytes_recv)
    return msg_recv


def RunSimulation(Port,ID_sub,cav_id,n_vehicle_sub,Subsystem_para_mod,S,
                  uini,eini,yini,Su,child_conn,parent_conn):
    # 对于每个子系统，在得到初始数据（以上输入参数）后开始仿真
    # 计算Tini-1:total_time_step-1 的控制量u 测量到total_time_step-1步的输出后停止仿真

    v_star = Subsystem_para_mod.v_star
    s_star = Subsystem_para_mod.s_star
    sine_amp = 4

    # problem size
    m = uini.ndim         # the size of control input of each subsystem
    p = yini.shape[0]     # the size of output of each subsystem

    # operation data for analyse
    computation_time = np.zeros(total_time_step)
    iteration_num    = np.zeros(total_time_step)
    y_error          = np.zeros(total_time_step)
    CAV_vel_error    = np.zeros(total_time_step)
    CAV_spa_error    = np.zeros(total_time_step)

    # real-world cost
    cost = np.zeros(total_time_step)


    # 给云发送初始信息
    # for k in range(Tini-1,total_time_step-1):
    for k in range(total_time_step):
        compute_success,u_cloud,real_iter_num,use_time,y_prediction,cost_this = asyncio.run(ConnectCloud(Port,k,cav_id,n_vehicle_sub,uini,eini,yini))
        if compute_success == True:
            print(f"Step {k+1}: CAV {cav_id} get cloud result.",u_cloud)
            print('iter_num = ',real_iter_num,', use_time = ',use_time)
            computation_time[k] = use_time
            iteration_num[k] = real_iter_num
 

        # # update accelaration
        acel = HDV_dynamics(S[k+Tini-1,:,:],Subsystem_para_mod)+\
            acel_noise*(-1+2*np.random.random(n_vehicle_sub))

        S[k+Tini-1,1:,2] = acel            # all the vehicles using HDV model

        # calculate control input via distributed DeeP-LCC in the cloud
        # apply control input
        # for safety reason, use HDV model to judge whether some CAVs need to brake
        if acel[0] == dcel_max:
            Su[k+Tini-1] = dcel_max
            S[k+Tini-1,1,2] = Su[k+Tini-1-delay_step]
        else:
            Su[k+Tini-1] = u_cloud*(1+acel_noise_CAV*random.uniform(-1,1))
            S[k+Tini-1,1,2] =  Su[k+Tini-1-delay_step]# a certain percentage of noise

        # update velocity
        S[k+Tini,:,1] = S[k+Tini-1,:,1] + Tstep*S[k+Tini-1,:,2]

        # 除了最后一个子系统，其他子系统将末车速度信息发送给后一个子系统
        if cav_id!=n_cav-1:
            last_veh_vel = S[k+Tini,-1,1]
            SendMessage(last_veh_vel,child_conn)

        # 更新S[~,0,1]
        # perturbation for the head vehicle: sin wave
        # 第一个子系统需要用sin函数更新S[~,0,:]这个参数
        # 其他子系统需要用进程间通信来获得这个参数
        if cav_id == 0:
            S[k+Tini,0,1] = v_star + sine_amp*math.sin(2*math.pi/(10/Tstep)*(k))
        else:
            S[k+Tini,0,1] = ReceiveMessage(parent_conn)
                
        # update position
        S[k+Tini,:,0] = S[k+Tini-1,:,0] + Tstep*S[k+Tini-1,:,1]

        # record output
        y = measure_mixed_traffic(S[k+Tini,1:,1],S[k+Tini,:,0],ID_sub,v_star,s_star)
        y_error[k] = np.linalg.norm(y_prediction - y)
        CAV_vel_error[k] = y_prediction[0] - y[0]
        CAV_spa_error[k] = y_prediction[-1] - y[-1]
        e_input = S[k+Tini,0,1] - v_star # 前一个子系统末车的速度误差

        # update yini and eini and uini
        eini = np.append(eini[1:],e_input)
        uini = np.append(uini[1:],u_cloud)
        yini = np.hstack((yini[:,1:],y.reshape([p,1])))

        # calculate cost
        cost[k] = cost_this

    return S,computation_time,iteration_num,cost,y_error,CAV_vel_error,CAV_spa_error
