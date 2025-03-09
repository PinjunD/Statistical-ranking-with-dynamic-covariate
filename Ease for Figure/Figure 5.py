import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import generator,algorithm
from matplotlib import rc
from joblib import Parallel,delayed
import time
rc('text', usetex=True)
rc('font', family='serif')
pd.set_option('future.no_silent_downcasting', True)
sz = 36
Figure_name = os.path.basename(__file__)[:-3]
cpu_cores = os.cpu_count()


np.random.seed(100)
def simulation(n, N, v,d, m_lower, m_upper, name = 'NURHM'):
        print('-'*10)
        H = generator.MultipleComparison(n,N,v,m_lower=m_lower,m_upper=m_upper,Type = name)
        u_true, v_true = H.u,H.v
        u_estimation, v_estimation = algorithm.AM(H.T,H.X,n,d,u_initial=u_true,v_initial=v_true)
        u_infty = max(abs(u_estimation-u_true))
        v_infty = max(abs(v_estimation-v_true))
        result = [H.n,u_infty,v_infty]
        return result
            
t1 = time.time()

repeat_time = 300 # set 300

# NURHM

## Settings
n = [200,400,600,800,1000]
v = [1,-0.5,0]
d = len(v)
N = lambda n: int(0.1*n*(np.log(n))**3)
m_lower = 2
m_upper = 8
tasks = [delayed(simulation)(nn, N, v, d, m_lower, m_upper,name = 'NURHM') 
         for nn in n for _ in range(repeat_time) ]
results = np.array(Parallel(n_jobs=cpu_cores)(tasks))
## Prepocessing data
S,T = [],[]
for nn in n:
    data = results[results[:,0]==nn,1:]
    S.append(np.mean(data,axis = 0))
    T.append(np.std(data,axis = 0))
S,T = np.array(S),np.array(T)
## plot
sz =36
y_ticks = [[0.2+0.1*i for i in range(8)],[0.01*i for i in range(5)]]
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
save = [Figure_name+'(a)',Figure_name+'(b)']
title_NURHM = [r'NURHM: $\widehat{\bm{u}}$ error', 
               r'NURHM: $\widehat{\bm{v}}$ error']
label_y = [r"$\|\widehat{\bm{u}} - \bm{{u}}^*\|_\infty$", 
           r"$\|\widehat{\bm{v}} - \bm{{v}}^*\|_\infty$"]
color = ['blue', 'orange']
for j in range(2):
    fig, ax = plt.subplots(figsize=(10.8, 10))
    ax.set_position([0.155, 0.125, 0.8, 0.8])
    ax.plot(n, S[:,j], marker='o', linestyle='-', color=color[j])
    ax.fill_between(n, -T[:, j] + S[:, j], T[:, j] + S[:, j], alpha=0.2, color=color[j])
    ax.set_title(title_NURHM[j], fontsize=sz)
    ax.set_xlabel('Numbers of items $n$', size=sz)
    ax.set_ylabel(label_y[j], size=sz)
    plt.xticks(range(200, 1200, 200), size=sz)
    plt.yticks(y_ticks[j], size=sz)
    plt.grid()
    plt.savefig(save[j]+'.png')


# HSBM

## Settings
N = lambda n: int(0.07*n**2)
m_lower = 5
m_upper = 6
tasks = [delayed(simulation)(nn, N, v, d, m_lower, m_upper,name = 'HSBM') 
         for nn in n for _ in range(repeat_time) ]
results = np.array(Parallel(n_jobs=cpu_cores)(tasks))
## Prepocessing data
S,T = [],[]
for nn in n:
    data = results[results[:,0]==nn,1:]
    S.append(np.mean(data,axis = 0))
    T.append(np.std(data,axis = 0))
S,T = np.array(S),np.array(T)
## plot
save = [Figure_name+'(c)',Figure_name+'(d)']
title_HSBM = [r'HSBM: $\widehat{\bm{u}}$ error', 
               r'HSBM: $\widehat{\bm{v}}$ error']
for j in range(2):
    fig, ax = plt.subplots(figsize=(10.8, 10))
    ax.set_position([0.155, 0.125, 0.8, 0.8])
    ax.plot(n, S[:,j], marker='o', linestyle='-', color=color[j])
    ax.fill_between(n, -T[:, j] + S[:, j], T[:, j] + S[:, j], alpha=0.2, color=color[j])
    ax.set_title(title_HSBM[j], fontsize=sz)
    ax.set_xlabel('Numbers of items $n$', size=sz)
    ax.set_ylabel(label_y[j], size=sz)
    plt.xticks(range(200, 1200, 200), size=sz)
    plt.yticks(y_ticks[j], size=sz)
    plt.grid()
    plt.savefig(save[j]+'.png')

t2 = time.time()
print(f'times:{t2-t1}')