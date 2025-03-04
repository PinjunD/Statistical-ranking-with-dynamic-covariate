#  (a): Four Gaussian radial basis functions in the best model selected by BIC. 
#  (b): The estimated Aging effect(t) = 6.238 · exp{−0.01 · (t − 25)^2} − 0.865 · exp{−0.03 · (t − 25)^2} − 3.339 · exp{−0.01 · (t − 30)^2}+3.320·exp{−0.01 · (t − 35)^2}, where the gray region between the two vertical dashed lines captures the 99% of the age range of the participants’ careers in the dataset. 
#  (c): Top ten players ranked by the PlusDC model based on their estimated utilities; their ranks under the BT model are also reported

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

# Data analysis

## basis functions
A = [25,30,35]
Lamb = [0.01, 0.03]
parameters = [(a,lamb) for a in A for lamb in Lamb]
gauss_kernel = lambda x,a,lamb: np.exp(-lamb*(x-a)**2)
gauss_kernel_set = lambda x: np.array([gauss_kernel(x,a,lamb) for (a,lamb) in parameters])
cov = [gauss_kernel_set(age).T for age in Age]
## We have selected the best basis functions based on BIC.
s = (0, 1, 2, 4)
d = len(s)
print('selected subset:', s)
X = [x[:,s] for x in cov]
## fit PlusDC,BT
import algorithm
u_plusDC,v_plusDC = algorithm.AM(T,X,n,d,
                E=1e-4/N,Eu=1e-8,Ev=1e-12,
                I=52,type = 'pair',detail=True)
v=np.array([0]*d)
KK = np.array([k[0] - k[1] for k in X])
u_BT = algorithm.pair_fixv(T,KK,v,n,E = 1e-8,I=52)
likelihood_BT = algorithm.pair_likelihood(T,X,u_BT,v)

# plot table
np.set_printoptions(precision=3)
plusDC_top10 = np.argsort(u_plusDC)[-10:][::-1]
u_t10_plusDC = u_plusDC[plusDC_top10]
sorted_indices = np.argsort(-u_BT)  
ranks_BT = np.empty_like(sorted_indices)
ranks_BT[sorted_indices] = np.arange(1, len(u_BT) + 1) 
TopPlayers, RankBT, RankPlusDC = [],[],[] 
for i,index in enumerate(plusDC_top10):
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
fig, ax = plt.subplots(figsize=(6,3))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')
plt.savefig(Figure_name+'(c).png')

# plot basis function
basis_functions_set = [gauss_kernel(x,parameters[ss][0],parameters[ss][1]) for ss in s]
text = ['$(25, 0.01)$', '$(25, 0.03)$', '$(30, 0.01)$', '$(30, 0.03)$', '$(35, 0.01)$', '$(35, 0.03)$']
color = ['slateblue','slateblue','coral','coral','deeppink','deeppink']
linestyle = ['-',':','-',':','-',':']
fig,ax = plt.subplots(figsize=(11.3,10))
x = np.linspace(8,64,100)
for ss in s:    
    ax.plot(x, gauss_kernel(x,parameters[ss][0],parameters[ss][1]), 
    color = color[ss], linewidth = 3,linestyle=linestyle[ss], label = text[ss])
sz = 36
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
plt.show()