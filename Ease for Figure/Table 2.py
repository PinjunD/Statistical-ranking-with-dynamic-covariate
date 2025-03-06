import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
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
import itertools
from joblib import Parallel, delayed
# models
def get_results(T,cov,s,N,n):
    X = [x[:,s] for x in cov]
    d = len(s)
    u_plusDC,v_plusDC = algorithm.AM(T,X,n,d,
                                 E=1e-5,Eu=1e-5,Ev=1e-12,
                                 I=52)
    likelihood = algorithm.multi_likelihood(T,X,u_plusDC,v_plusDC)
    AIC = (n-1+d) * 2 / N - 2 * likelihood
    BIC = (n-1+d) * np.log(N) / N - 2 * likelihood
    print(f'{s}:{likelihood},{AIC},{BIC}')
    return [s,likelihood,AIC,BIC]
    

# get_subset
get_subset = lambda n: [subset for i in range(n + 1)
                         for subset in itertools.combinations(list(range(n)), i)]
subset = get_subset(3)
results = Parallel(n_jobs=9)(delayed(get_results)(T,cov,ss,N,n) for ss in subset)
Subset, likelihood, AIC_, BIC_ = [],[],[],[]
for result in results:
    t = [0,0,0]
    for s in result[0]:
        t[s] = 1
    string = f'subset:{t}, log-likelihood:{result[1]}, AIC:{result[2]:.2f},BIC:{result[3]:.2f}'
    print(string)
    Subset.append(t)
    likelihood.append("{:.3f}".format(result[1]))
    AIC_.append("{:.3f}".format(result[2]))
    BIC_.append("{:.3f}".format(result[3]))
df = pd.DataFrame({
    '': Subset,
    'log-likelihood': likelihood,
    'AIC': AIC_,
    'BIC': BIC_,
})
fig, ax = plt.subplots(figsize=(10.8,10))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')
plt.savefig(Figure_name+'.png')
    