import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import generator,algorithm,basicfun
from matplotlib import rc
from joblib import Parallel,delayed
rc('text', usetex=True)
rc('font', family='serif')
pd.set_option('future.no_silent_downcasting', True)
sz = 36
Figure_name = os.path.basename(__file__)[:-3]

def simulation(n, N, v, m_lower, m_upper,
                name = 'NURHM',
                repeat_time = 10,
                ):
    current_dir = os.getcwd()
    data_dir = current_dir + '\\' +'data' + '\\' + 'synthesis' + '\\'

    d = len(v)
    for nn in n:
        for _ in range(repeat_time):
            print('-'*10)
            H = generator.MultipleComparison(nn,N,v,m_lower=m_lower,m_upper=m_upper,Type = name)
            u_true, v_true = H.u,H.v
            u_estimation, v_estimation = algorithm.AM(H.T,H.X,nn,d,u_initial=u_true,v_initial=v_true)
            u_infty = max(abs(u_estimation-u_true))
            v_infty = max(abs(v_estimation-v_true))
            print(f'n:{nn},u_infty:{u_infty},v_infty:{v_infty}')
            with open(name+'.txt', 'a') as file:
                file.write(str(nn) + ',' + str(u_infty) + ',' + str(v_infty) + '\n')

repeat_time = 1 ### set 300

### NURHM
n = [200,400,600,800,1000]
v = [1,-0.5,0]
N = lambda n: int(0.1*n*(np.log(n))**3)
m_lower = 2
m_upper = 8
simulation(n, N, v, m_lower, m_upper, repeat_time = repeat_time,name = 'NURHM')
basicfun.plot('NURHM',Figure_name,name=['a','b'])


### HSBM
N = lambda n: int(0.07*n**2)
m_lower = 5
m_upper = 6
simulation(n, N, v, m_lower, m_upper, repeat_time = repeat_time,name = 'HSBM')
basicfun.plot('HSBM',Figure_name,name=['c','d'])