import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from joblib import Parallel,delayed
rc('text', usetex=True)
rc('font', family='serif')
pd.set_option('future.no_silent_downcasting', True)
sz = 36
Figure_name = os.path.basename(__file__)[:-3]

# loading data
covariate_columns = ['Act. Wt.','Dr.','Win Odds']
name_columns = 'Horse'

df = pd.read_csv('runs(preprocessed).csv',index_col='race_id',low_memory=False)

horseID = {}
T = []
cov = []
for index in df.index.unique():
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
n = len(horseID)
N = len(T)
d = len(covariate_columns)

import algorithm
folds = []
np.random.seed(42)
temp_fold = set(range(N))
I = int(N/100)
for i in range(I):
    temp_fold = list(temp_fold)
    index = np.random.choice(temp_fold, 100,replace=False)
    index = np.sort(index)[::-1]
    temp_fold = set(temp_fold) - set(index)
    folds.append(index.tolist())
folds[-1] = folds[-1] +list(temp_fold)

def horse_cross_validation(T,cov,n,d,subset,CV_filename):
    N = len(T)
    Ttrain = [T[i] for i in range(N) if i not in subset]
    Xtrain = [cov[i] for i in range(N) if i not in subset]
    Ttest = [T[i] for i in range(N) if i in subset]
    Xtest = [cov[i] for i in range(N) if i in subset]


    u_pl,v_pl = algorithm.AM(Ttrain,Xtrain,n,d,P=True,Eu=1e-5,detail=True)
    u_plusDC,v_plusDC = algorithm.AM(Ttrain,Xtrain,n,d,
                                 E=1e-5,Eu=1e-5,Ev=1e-12,
                                 I=52)
    
    full_likelihood_pl = algorithm.multi_likelihood(Ttest,Xtest,u_pl,v_pl)
    top_likelihood_pl = algorithm.multi_likelihood(Ttest,Xtest,u_pl,v_pl,l=1)
    three_likelihood_pl = algorithm.multi_likelihood(Ttest,Xtest,u_pl,v_pl,l=3)
    likelihood_pl = [full_likelihood_pl,top_likelihood_pl,three_likelihood_pl]

    full_likelihood_plusDC = algorithm.multi_likelihood(Ttest,Xtest,u_plusDC,v_plusDC)
    top_likelihood_plusDC = algorithm.multi_likelihood(Ttest,Xtest,u_plusDC,v_plusDC,l=1)
    three_likelihood_plusDC = algorithm.multi_likelihood(Ttest,Xtest,u_plusDC,v_plusDC,l=3)
    likelihood_plusDC = [full_likelihood_plusDC,top_likelihood_plusDC,three_likelihood_plusDC]

    ### belief
    NN = len(Xtest)
    full_belief,top_belief,three_belief = 0, 0, 0
    for xx in Xtest:
        b = np.exp(xx[:, -1])
        full_belief += sum([np.log(b[j] / sum(b[j:])) for j in range(len(b))])
        top_belief += np.log(b[0] / sum(b))
        three_belief += sum([np.log(b[j] / sum(b[j:])) for j in range(3)])
    full_belief /= NN
    top_belief /= NN
    three_belief /= NN
    belief = [full_belief,top_belief,three_belief]

    results = {'plusDC':likelihood_plusDC,
               'pl':likelihood_pl,
               'belief':belief}

    with open(CV_filename,'a') as f:
        for values in results.values():
            for value in values:
                f.write(str(value)+',')
            f.write(';')
        f.write('\n')

### Save data
CV_filename = 'horse_KFCV.txt'
results = Parallel(n_jobs=8)(delayed(horse_cross_validation)
                             (T,cov,n,d,fold,CV_filename) for fold in folds)

### Loading the results in above step.
Data = []
with open(CV_filename,'r') as file:
    for line in file:
        tem = np.zeros((3,3))
        models = line.split(';')[:-1]
        for i in range(3):
            tem[i,:] = list(map(float,models[i].split(',')[:-1]))
        Data.append(tem)
Data = np.array(Data)

save_name = ['(a)','(b)','(c)']
title = ['(Full)', '(Top-1)', '(Top-3)']
label = ['PlusDC', 'PL', 'Public belief']
colors = ['lightblue', 'orange', 'lightgreen']
yrange = [[-20,-19,-18,-17,-16],[-1.4,-1.7,-2.0,-2.3,-2.6],[-5.2,-5.7,-6.2,-6.7,-7.2]]


for j in range(3):
    fig, ax = plt.subplots(figsize=(10.8, 10))
    bx = ax.boxplot(Data[:,:,j], tick_labels=label, patch_artist=True,medianprops={'linewidth': 2})
    ax.set_position([0.165, 0.125, 0.8, 0.8])
    ax.set_ylabel('Normalized log-likelihood',size = sz)
    plt.title('$K$-fold cross-validation'+title[j],size =sz+4)
    plt.xticks(size=sz)
    plt.yticks(yrange[j],size=sz)
    ax.tick_params(axis='y', which='both')


    for median in bx['medians']:
        median.set_color('black')
    for patch, color in zip(bx['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    plt.savefig(Figure_name+save_name[j]+'.png')
plt.show()
