import os
import sys
import time
from multiprocessing import Manager
import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import rc

import algorithm

rc('text', usetex=True)
rc('font', family='serif')
pd.set_option('future.no_silent_downcasting', True)
sz = 36
Figure_name = os.path.basename(__file__)[:-3]
cpu_cores = os.cpu_count()

def print_progress_bar(iteration, total, length=40):
    percent = (iteration / total)
    bar_length = int(length * percent)
    bar = '\u2588' * bar_length + '-' * (length - bar_length)
    sys.stdout.write(f'\r|{bar}| {percent:.1%} Complete')
    sys.stdout.flush()
def get_BIC(T,cov,s,N,n,shared_list):
    X = [x[:,s] for x in cov]
    d = len(s)
    u_plusDC,v_plusDC = algorithm.AM(T,X,n,d,E=1e-3,Eu=1e-8,Ev=1e-12,type = 'pair')
    likelihood = algorithm.multi_likelihood(T,X,u_plusDC,v_plusDC)
    BIC = (n-1+d) * np.log(N) - 2*likelihood*N
    shared_list.append(1)
    print_progress_bar(len(shared_list),64)
    return BIC
# loading data
if __name__ == "__main__":
    print('In this program, we investigate the aging effect among tennis players through PlusDC model.')
    time.sleep(2)
    print('\n'+"="*10+'Data Loading'+"="*10)
    print('Loading data from \'tennis(preprocessed).csv\'......')
    df = pd.read_csv('tennis(preprocessed).csv',low_memory=False)
    ## players ID
    players = df[['winner_name','loser_name']]
    counts = pd.Series(players.values.ravel()).value_counts()
    playerID = {value: index for index, value in enumerate(counts.index.tolist())}
    n = len(playerID)
    ## matches
    name_columns = ['winner_name','loser_name']
    Matches = df[name_columns].replace(playerID)
    T = np.array(Matches).tolist()
    N = len(T)
    ## age
    age_columns = ['winner_age','loser_age']
    Age = np.array(df[age_columns])
    print('Complete!')
    time.sleep(1)

    # Data analysis
    print('\n'+"="*10+'Data analysis'+"="*10)
    ## basis functions
    print('We construct covariates using Gaussian radial basis functions '
          '\nf(t;a,ρ) = exp{-ρ(t - a)^2}, where parameters (a,ρ) ∈ {25,30,35}×{0.01,0.03}')
    A = [25,30,35]
    Lamb = [0.01, 0.03]
    parameters = [(a,lamb) for a in A for lamb in Lamb]
    gauss_kernel = lambda x,a,lamb: np.exp(-lamb*(x-a)**2)
    gauss_kernel_set = lambda x: np.array([gauss_kernel(x,a,lamb) for (a,lamb) in parameters])
    cov = [gauss_kernel_set(age).T for age in Age]
    print('Then, we calculate the BIC corresponding to each covariates\' combination. (64 candidates)')
    print('running......')
    manager = Manager()
    shared_list = manager.list() 
    # get_subset
    get_subset = lambda n: [subset for i in range(n + 1)
                            for subset in itertools.combinations(list(range(n)), i)]
    subset = get_subset(6)
    res = Parallel(n_jobs=cpu_cores)(delayed(get_BIC)(T,cov,subset[i],N,n,shared_list) for i in range(len(subset)))
    s = subset[np.argmin(res)]
    print('\nWe have selected the best basis functions based on BIC.')
    print('(a,ρ) ∈ {(25,0.01),(25,0.03),(30,0.01),(35,0.01)}')
    time.sleep(1)

    print('Then, we compare the ranking from PlusDC and BT.')
    time.sleep(1)
    print('running......')
    ## fit PlusDC,BT
    d = len(s)
    X = [x[:,s] for x in cov]
    v=np.array([0]*d)
    KK = np.array([k[0] - k[1] for k in X])
    u_BT = algorithm.pair_fixv(T,KK,v,n,E = 1e-8,I=52)
    u_plusDC,v_plusDC = algorithm.AM(T,X,n,d,
                    E=1e-4/N,Eu=1e-8,Ev=1e-12,
                    I=52,type = 'pair')
    likelihood_BT = algorithm.pair_likelihood(T,X,u_BT,v)
    print(f"Complete! v={v_plusDC}")
    # plot table
    print('\n'+"="*10+'Ranking'+"="*10)
    np.set_printoptions(precision=3)
    plusDC_top10 = np.argsort(u_plusDC)[-10:][::-1]
    u_t10_plusDC = u_plusDC[plusDC_top10]
    sorted_indices = np.argsort(-u_BT)  
    ranks_BT = np.empty_like(sorted_indices)
    ranks_BT[sorted_indices] = np.arange(1, len(u_BT) + 1) 
    TopPlayers, RankBT, RankPlusDC = [],[],[] 
    for i,index in enumerate(plusDC_top10):
        time.sleep(1)
        player_name = [key for key, value in playerID.items() if value == index][0]
        TopPlayers.append(player_name)
        RankBT.append(ranks_BT[playerID[player_name]])
        RankPlusDC.append(i+1)
        print('-'*10+f'Top-{i+1}(PlusDC)'+'-'*10)
        print(f'Player: {player_name}. Score_PlusDC: {u_t10_plusDC[i]:.3f}.')
        print(f'Rank (BT):{ranks_BT[playerID[player_name]]}. Score_BT:{u_BT[playerID[player_name]]:.3f}.')
    df = pd.DataFrame({
        'Player': TopPlayers,
        r'$Rank_{BT}$': RankBT,
        r'$Rank_{PlusDC}$': RankPlusDC
    })
    time.sleep(2)
    print(df)
    print(f'The results have been saved in {Figure_name}.csv')
    df.to_csv(Figure_name+'(c).csv',index=False)


    time.sleep(3)
    print('At last, we plot the selected basis functions and aging effect.') 
    
    # plot basis function
    x = np.linspace(8,64,100)
    text = ['$(25, 0.01)$', '$(25, 0.03)$', '$(30, 0.01)$', '$(30, 0.03)$', '$(35, 0.01)$', '$(35, 0.03)$']
    color = ['slateblue','slateblue','coral','coral','deeppink','deeppink']
    linestyle = ['-',':','-',':','-',':']
    fig,ax = plt.subplots(figsize=(11.3,10))
    x = np.linspace(8,64,100)
    for ss in s:    
        ax.plot(x, gauss_kernel(x,parameters[ss][0],parameters[ss][1]), 
        color = color[ss], linewidth = 3,linestyle=linestyle[ss], label = text[ss])
    ax.set_xlabel('Age', size = sz)
    ax.set_ylabel('Value', size = sz)
    ax.set_ylim([-0.05, 1.18])
    plt.xticks(size = sz)
    plt.yticks(size = sz)
    rc('text', usetex=True)
    rc('font', family='serif')
    ax.legend(title='Selected $(a, \lambda)$', prop={'size': sz},\
            title_fontsize=sz, loc='upper right')
    plt.grid()
    plt.savefig(Figure_name +'(a).png')



    # plot aging effect
    time.sleep(3)
    age_effect = lambda t: sum([v_plusDC[i]*gauss_kernel(t,parameters[ss][0],parameters[ss][1]) for i,ss in enumerate(s)])

    x = np.linspace(8,64,100)
    y = age_effect(x)
    fig,ax = plt.subplots(figsize=(11.3,10))
    ax.plot(x, y, color = 'black', linewidth = 5)

    ax.axvline(x=17.4, color='red', linestyle=':', linewidth = 3)
    ax.axvline(x=36.6, color='red', linestyle=':', linewidth = 3)
    ax.fill_betweenx(y*100-20, 17.4, 36.6, color='gray', alpha=0.12)

    sz = 36
    ax.set_xlabel('Age', size = sz)
    ax.set_ylabel('Aging effect', size = sz)
    ax.set_ylim([-0.2, 4.4])
    plt.xticks(size = sz)
    plt.yticks(size = sz)

    rc('text', usetex=True)
    rc('font', family='serif')
    plt.grid()
    plt.savefig(Figure_name+'(b).png')
    print(f'These two figures have been saved in {Figure_name}(a)(b).png')
    