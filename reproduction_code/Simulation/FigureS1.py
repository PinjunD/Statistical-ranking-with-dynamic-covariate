import os
from multiprocessing import Manager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import generator,algorithm
from matplotlib import rc
from joblib import Parallel,delayed
import time
rc('text', usetex=True)
rc('font', family='serif')
# pd.set_option('future.no_silent_downcasting', True)
sz = 36
Figure_name = os.path.basename(__file__)[:-3]
cpu_cores = os.cpu_count()
np.random.seed(100)

def printf(u_infty,v_infty,repeat,nn,name):
    if repeat == 1:
        repeat_str = str(repeat)+'st'
    elif repeat == 2:
        repeat_str = str(repeat)+'nd'
    elif repeat == 3:
        repeat_str = str(repeat)+'rd'
    else:
        repeat_str = str(repeat)+'th'
    lines = '-'*10+'\n'
    print1 = f'The {repeat_str} repetition of {name}: number of nodes:{nn}.\n'
    print2 = f'u_infty:{u_infty}, v_infty:{v_infty}.'
    screen = lines + print1 + print2
    print(screen)

def simulation(n, N, v,d, m_lower, m_upper, name = 'NURHM', shared_list = None):
        H = generator.MultipleComparison(n,N,v,m_lower=m_lower,m_upper=m_upper,Type = name)
        u_true, v_true = H.u,H.v
        u_estimation, v_estimation = algorithm.AM(H.T,H.X,n,d,u_initial=u_true,v_initial=v_true)
        u_infty = max(abs(u_estimation-u_true))
        v_infty = max(abs(v_estimation-v_true))
        result = [H.n,u_infty,v_infty]
        if shared_list is None:
            pass
        else:
            shared_list.append(0)
            printf(u_infty,v_infty,len(shared_list),n,name)
        return result

repeat_time = 300 # set 300

if __name__ == '__main__':

    print('In this program, we verify the uniform consistency '
        'of the MLE in the PlusDC model using simulated data.\n'
        'We set n={200,400,600,800,1000} and conduct'
        'the simulations for NURHM and HSBM with 300 repeations.\n'
        'Please note that the program takes nearly 3 hours to complete.')
    """for i in range(5):
        print(str(5-i)+'.'*6)
        time.sleep(1)"""
    time.sleep(3)
    print('Start!')


    t1 = time.time()

    # NURHM
    print('='*10+'NURHM'+'='*10)
    ## Settings
    n = [200,400,600,800,1000]
    v = [1,-0.5,0]
    d = len(v)
    N = lambda n: int(0.1*n*(np.log(n))**3)
    m_lower = 2
    m_upper = 8
    results = []
    for nn in n:
        manager = Manager()
        shared_list = manager.list() 
        tasks = [delayed(simulation)(nn, N, v, d, m_lower, m_upper,name = 'NURHM',shared_list=shared_list) 
                for i in range(repeat_time) ]
        results_temp = np.array(Parallel(n_jobs=cpu_cores)(tasks))
        results.append(results_temp)
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
    print(f'We have saved the NURHM results as {Figure_name}(a).png and {Figure_name}(b).png.\n')

    # HSBM
    print('\n='*10+'HSBM'+'='*10)
    ## Settings
    N = lambda n: int(0.07*n**2)
    m_lower = 5
    m_upper = 6
    tasks = [delayed(simulation)(nn, N, v, d, m_lower, m_upper,name = 'HSBM',repeat = repeat+1) 
            for nn in n for repeat in range(repeat_time) ]
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
    print(f'We have saved the HSBM results as {Figure_name}(c).png and {Figure_name}(d).png\n')

    t2 = time.time()
    print(f'Total Time:{int(t2-t1)}s')