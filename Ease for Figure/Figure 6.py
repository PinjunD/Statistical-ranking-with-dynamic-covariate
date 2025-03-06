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


# Fit PlusDC,PL
import algorithm
u_pl,v_pl = algorithm.AM(T,cov,n,d,P=True,Eu=1e-5,detail=True)
u_plusDC,v_plusDC = algorithm.AM(T,cov,n,d,
                                 E=1e-5,Eu=1e-5,Ev=1e-12,
                                 I=52,detail=True)
plusDC_top10 = np.argsort(u_plusDC)[-10:][::-1]
u_t10_plusDC = u_plusDC[plusDC_top10]
sorted_indices = np.argsort(-u_pl)  
ranks_pl = np.empty_like(sorted_indices)
ranks_pl[sorted_indices] = np.arange(1, len(u_pl) + 1) 
Horseid,Race,Meanplace,Meancovariates,uPL,uPlusDC,RankPL,RankPlusDC = [],[],[],[],[],[],[],[]
i = 0
for index, uu in zip(plusDC_top10,u_t10_plusDC):
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



## Plot right
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
fig, ax = plt.subplots(figsize=(10.8,10))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')
plt.savefig(Figure_name+'(right).png')
    


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

plt.savefig(Figure_name+'(left).png')
plt.show()