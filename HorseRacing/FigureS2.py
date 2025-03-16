import os
import sys
import time
from multiprocessing import Manager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
cpu_cores = os.cpu_count()

rc('text', usetex=True)
rc('font', family='serif')
# pd.set_option('future.no_silent_downcasting', True)
sz = 36
Figure_name = os.path.basename(__file__)[:-3]
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0,project_root)
from package import algorithm

data_file_path = lambda x: os.path.join(project_root, 'data', x)
save_path = lambda x: os.path.join(project_root, 'results', f'{Figure_name}{x}')

def print_progress_bar(iteration, total, length=40):
    percent = (iteration / total)
    bar_length = int(length * percent)
    bar = '\u2588' * bar_length + '-' * (length - bar_length)
    sys.stdout.write(f'\r|{bar}| {percent:.1%} Complete')
    sys.stdout.flush()


if __name__ == "__main__":
    print('Furthermore, we compare the results between PL and PlusDC.')
    time.sleep(3)
    # loading data
    print('\n'+"="*10+'Data Loading'+"="*10)
    print('Loading data from \'data\\runs(preprocessed).csv\'...')
    covariate_columns = ['Act. Wt.','Dr.','Win Odds']
    name_columns = 'Horse'
    df = pd.read_csv(data_file_path('runs(preprocessed).csv'),index_col='race_id',low_memory=False)

    horseID = {}
    T = []
    cov = []
    index_list = df.index.unique()
    index_list_num = len(index_list)
    for ii,index in enumerate(index_list):
        temp = df.loc[index].sort_values(by='Pla.')    
        ### covariates
        x = np.array(temp[covariate_columns])
        cov.append(x)
        place = np.array(temp['Pla.'])
        ###
        t = []
        horses_name = temp[name_columns].tolist()
        for i,horse_name in enumerate(horses_name):
            added_horse = horseID.keys()
            if horse_name in added_horse:
                horseID[horse_name]['times'] += 1
            else:
                n = len(added_horse)
                horseID[horse_name] = {'ID':n,'times':1,'place':[],'covariate':[]}
            #horseID[horse_name]['place'].append(i+1)
            horseID[horse_name]['place'].append(place[i])
            xx = x[i].copy()
            xx[-1] = np.exp(xx[-1])
            horseID[horse_name]['covariate'].append(xx)
            t.append(horseID[horse_name]['ID'])
        T.append(t)
        print_progress_bar(ii, index_list_num)
    n = len(horseID)
    N = len(T)
    d = len(covariate_columns)
    print('\n'*2+"="*10+'Data Analysis'+"="*10)
    print('We fit the PL model and the PlusDC model independently.')

    # Fit PlusDC,PL
    print('-'*10+'PL model'+'-'*10)
    print('Running...')
    u_pl,v_pl = algorithm.AM(T,cov,n,d,PL=True,Eu=1e-5)
    l_pl = algorithm.multi_likelihood(T,cov,u_pl,v_pl)

    
    print(f'The log-likelihood of PL model: {l_pl}')
    print('-'*10+'PlusDC model'+'-'*10)
    print('Running...')
    u_plusDC,v_plusDC = algorithm.AM(T,cov,n,d,
                                    E=1e-5,Eu=1e-5,Ev=1e-12,
                                    I=52)
    likelihood = algorithm.multi_likelihood(T,cov,u_plusDC,v_plusDC)
    print(f"The log-likelihood of PlusDC model: {likelihood}")
    print(f"The coefficient of covariates: {v_plusDC}")
    print('\n'+"="*10+'Ranking (PlusDC)'+"="*10)
    plusDC_top10 = np.argsort(u_plusDC)[-10:][::-1]
    u_t10_plusDC = u_plusDC[plusDC_top10]
    sorted_indices = np.argsort(-u_pl)  
    ranks_pl = np.empty_like(sorted_indices)
    ranks_pl[sorted_indices] = np.arange(1, len(u_pl) + 1) 
    Horseid,Race,Meanplace,Meancovariates,uPL,uPlusDC,RankPL,RankPlusDC = [],[],[],[],[],[],[],[]
    i = 0
    for index, uu in zip(plusDC_top10,u_t10_plusDC):
        time.sleep(0.5)
        i += 1
        horse_name = [key for key, value in horseID.items() if value['ID'] == index][0]
        id = horseID[horse_name]['ID']
        times = horseID[horse_name]['times']
        mean_place = np.mean(horseID[horse_name]['place'])
        mean_covariate = np.mean(horseID[horse_name]['covariate'],axis=0)
        uu_pl = u_pl[id]
        print('-'*10+f'Ranking:{i}'+'-'*10)
        print(f'Horse:{horse_name}')
        print(f'Race:{times}, Mean_Place:{mean_place}')
        print(f'Mean_covariate:{mean_covariate}')
        print(f'Score_PL:{uu_pl},Score_PlusDC:{uu}, Ranking_PL:{ranks_pl[id]}\n')
        Horseid.append(horse_name)
        Race.append(times)
        Meanplace.append("{:.3f}".format(mean_place))
        Meancovariates.append(np.round(mean_covariate, 3))
        uPL.append("{:.3f}".format(uu_pl))
        uPlusDC.append("{:.3f}".format(uu))
        RankPL.append(ranks_pl[id])
        RankPlusDC.append(i)




    ## Save right
    df = pd.DataFrame({
        'Horse id': Horseid,
        'Race': Race,
        'Mean place': Meanplace,
        'Mean covariates': Meancovariates,
        r'u_{PL}': uPL,
        r'u_{PlusDC}': uPlusDC,
        r'Rank_{PL}': RankPL,
        r'Rank_{PlusDC}': RankPlusDC
    })
    print('\n')
    print(df)
    df.to_csv(save_path('(right).csv'))
    print(f'The results have been saved in {Figure_name}(right).csv.')
    time.sleep(3)


    ## Plot left
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

    fig,ax = plt.subplots(figsize=(11.3,10))
    ax.scatter(u_pl,u_plusDC,s = 75,alpha=0.1,c='blue')
    sz = 36
    plt.xticks(size = sz)
    plt.yticks(size = sz)

    rc('text', usetex=True)
    rc('font', family='serif')
    plt.grid()
    ax.set_xlabel(r'$\hat{\bm{u}}$'+" (PL)",size = sz)
    ax.set_ylabel(r'$\hat{\bm{u}}$'+" (PlusDC)",size = sz)

    plt.savefig(save_path('(left).pdf'))
    print(f'The figure has also been saved as {Figure_name}(left).pdf.')
