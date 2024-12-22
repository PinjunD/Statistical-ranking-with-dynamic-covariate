from .generator import MultipleComparison
from .algorithm import *
import os

def simulation(n, N, v, m_lower, m_upper,
                name = 'NURHM',
                repeat_time = 1,
                ):
    current_dir = os.getcwd()
    data_dir = current_dir + '\\' + 'synthesis' + '\\'

    d = len(v)
    for nn in n:
        for _ in range(repeat_time):
            print('-'*10)
            H = MultipleComparison(nn,N,v,m_lower=m_lower,m_upper=m_upper,Type = name)
            print(H.T)
            u_true, v_true = H.u,H.v
            u_estimation, v_estimation = AM(H.T,H.X,nn,d,u=u_true,v=v_true)
            u_infty = max(abs(u_estimation-u_true))
            v_infty = max(abs(v_estimation-v_true))
            print(f'n:{nn},u_infty:{u_infty},v_infty:{v_infty}')
            with open(data_dir+name+'.txt', 'a') as file:
                file.write(str(nn) + ',' + str(u_infty) + ',' + str(v_infty) + '\n')