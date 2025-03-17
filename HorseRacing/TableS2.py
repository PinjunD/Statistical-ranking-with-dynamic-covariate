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
def get_results(T,cov,s,N,n,shared_list):
    X = [x[:,s] for x in cov]
    d = len(s)
    u_plusDC,v_plusDC = algorithm.AM(T,X,n,d)
    likelihood = algorithm.multi_likelihood(T,X,u_plusDC,v_plusDC)
    AIC = (n-1+d) * 2 / N - 2 * likelihood
    BIC = (n-1+d) * np.log(N) / N - 2 * likelihood
    shared_list.append(1)
    print_progress_bar(len(shared_list),8)
    return [s,likelihood,AIC,BIC]

if __name__ == "__main__":
    print('In this program, we investigate the impact of covariates on horse racing results.')
    time.sleep(3)
    print('Specifically, we consider three factors to explain the internal scores of horses:')
    str_list1 = [
        'Actual weight: The weight that a horse carries during a competition.', 
        'Draw: The post position number assigned to a horse in a race.', 
        'Public belief: The winning probability derived from the final win odds shortly before the competition begins.'
        ]
    for i in range(3):
        time.sleep(1)
        print(f'({i+1}){str_list1[i]}')



    # loading data
    print('\n'+"="*10+'Data Loading'+"="*10)
    print('To begin with, we load the data from \'data\\runs(preprocessed).csv\'...')
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

    # models
    print('\n'*2+"="*10+'Data Analysis'+"="*10)
    print('We use PlusDC to fit the data using various combinations of the three covariates. ')
    time.sleep(1)
    print('Running...')
    manager = Manager()
    shared_list = manager.list() 
    # get_subset
    get_subset = lambda n: [subset for i in range(n + 1)
                            for subset in itertools.combinations(list(range(n)), i)]
    subset = get_subset(3)
    results = Parallel(n_jobs=cpu_cores)(delayed(get_results)(T,cov,ss,N,n,shared_list) for ss in subset)
    print('\n')
    print('Finally, we calculate the corresponding log-likelihood, AIC, and BIC.')
    time.sleep(3)
    print('Here, (i_1,i_2,i_3) represents the PlusDC model incorporating the $t$th covariate if i_t = 1.')
    time.sleep(3)
    print('The PL model, namely (0,0,0), is a special case of PlusDC.')
    Subset, likelihood, AIC_, BIC_ = [],[],[],[]
    print('='*20)
    for result in results:
        t = [0,0,0]
        for s in result[0]:
            t[s] = 1
        string = f'subset:{t}, log-likelihood:{result[1]}, AIC:{result[2]:.2f},BIC:{result[3]:.2f}'
        Subset.append(t)
        likelihood.append("{:.3f}".format(result[1]))
        AIC_.append("{:.3f}".format(result[2]))
        BIC_.append("{:.3f}".format(result[3]))
        print(string)
        time.sleep(1)

    df = pd.DataFrame({
        '': Subset,
        'log-likelihood': likelihood,
        'AIC': AIC_,
        'BIC': BIC_,
    })
    print(df)
    print(f'The results have been saved in {Figure_name}.csv.')
    df.to_csv(save_path('.csv'),index=False)
        