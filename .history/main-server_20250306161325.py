from concurrent import futures
# import multiprocessing
from util import n_cav,RunSimulation,n_vehicle,total_time_step,pos_cav,Tstep,total_time,Tini
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import multiprocessing
import subsystem_init

def StartTest():
    pass

def StartProcess(executor):
    # 提交一个任务给执行器。这个任务就是调用 StartTest 函数。
    future = executor.submit(StartTest)
    # 通过 future.result() 来获取执行结果
    future.result()

if __name__ == "__main__":
    # args_list = [subsystem0.init_sub(),subsystem1.init_sub(),subsystem2.init_sub(),
    #              subsystem3.init_sub(),subsystem4.init_sub()]
    args_list = subsystem_init.init_sub()
    print("finish init")
    result_list = [[] for _ in range(n_cav)]

    # child发送 parent接收
    parent_conn0, child_conn5 = None,None
    parent_conn1, child_conn0 = multiprocessing.Pipe()
    parent_conn2, child_conn1 = multiprocessing.Pipe()
    parent_conn3, child_conn2 = multiprocessing.Pipe()
    parent_conn4, child_conn3 = multiprocessing.Pipe()
    parent_conn5, child_conn4 = multiprocessing.Pipe()

    with futures.ProcessPoolExecutor(max_workers=n_cav) as executor:
        StartProcess(executor)
        start_time = time.perf_counter()
        future0 = executor.submit(RunSimulation,*args_list[0],child_conn = child_conn0,parent_conn = parent_conn0)
        future1 = executor.submit(RunSimulation,*args_list[1],child_conn = child_conn1,parent_conn = parent_conn1)
        future2 = executor.submit(RunSimulation,*args_list[2],child_conn = child_conn2,parent_conn = parent_conn2)
        future3 = executor.submit(RunSimulation,*args_list[3],child_conn = child_conn3,parent_conn = parent_conn3)
        future4 = executor.submit(RunSimulation,*args_list[4],child_conn = child_conn4,parent_conn = parent_conn4)

        # 得到最终结果
        result_list[0] = future0.result()
        result_list[1] = future1.result()
        result_list[2] = future2.result()
        result_list[3] = future3.result()
        result_list[4] = future4.result()
    
    # result of all vehs
    result_S = np.zeros([total_time_step,n_vehicle+1,3])
    computation_time = np.zeros(total_time_step)
    iteration_num    = np.zeros(total_time_step)
    cost    = np.zeros(total_time_step)
    y_error = np.zeros(total_time_step)

    # for i in range(n_cav):
    #     computation_time = computation_time + result_list[i][1]
    #     iteration_num = iteration_num + result_list[i][2]
    #     cost = cost + result_list[i][3]
    #     y_error = y_error + result_list[i][4]
    #     np.savetxt('result/CAV_vel_error_'+str(i)+'.csv',result_list[i][5],fmt='%.6f', delimiter=',')
    #     np.savetxt('result/CAV_spa_error_'+str(i)+'.csv',result_list[i][6],fmt='%.6f', delimiter=',')

    computation_time = computation_time/n_cav # average computation time of each subsystem
    iteration_num = iteration_num/n_cav       # average iteration num of each subsystem
    cost = cost
    y_error = y_error/(n_cav**0.5)

    np.savetxt('result/computation_time.csv', computation_time, fmt='%.6f', delimiter=',')
    np.savetxt('result/iteration_num.csv', iteration_num, fmt='%.6f', delimiter=',')
    np.savetxt('result/cost.csv', cost, fmt='%.6f', delimiter=',')
    np.savetxt('result/y_error.csv', y_error, fmt='%.6f', delimiter=',')
    

    for i in range(total_time_step):
        result_S[i] = np.vstack((result_list[0][0][i+Tini],result_list[1][0][i+Tini][1:,:],
                                 result_list[2][0][i+Tini][1:,:],result_list[3][0][i+Tini][1:,:],
                                 result_list[4][0][i+Tini][1:,:]))

    for i in range(n_vehicle+1):
        np.savetxt('result/Veh_vel/Vel_'+str(i)+'.csv', result_S[:,i,1], fmt='%.6f', delimiter=',')
        np.savetxt('result/Veh_pos/Pos_'+str(i)+'.csv', result_S[:,i,0], fmt='%.6f', delimiter=',')

