# Log score dynamics of the top ten players ranked by model (1.2). 
# For each player, we plot their log scores as a function 
# of time starting from his career and ending at his retirement.
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
plusDC_top10 = np.argsort(u_plusDC)[-20:][::-1]
u_t10_plusDC = u_plusDC[plusDC_top10]
top_player = []
for i,index in enumerate(plusDC_top10):
    player_name = [key for key, value in playerID.items() if value == index][0]
    top_player.append(player_name)
    print(f'top-{i+1}: player: {player_name}, score: {u_t10_plusDC[i]}')

## Fit aging effect
age_effect = lambda t: sum([v_plusDC[i]*gauss_kernel(t,parameters[ss][0],parameters[ss][1]) for i,ss in enumerate(s)])
total_num = 10
players = top_player[:total_num]
players_information = {}

## load birthday information
with open('birthday.txt','r') as f:
    lines = f.readlines()[:total_num]
for line,player in zip(lines,players):
    win_age = df[df['winner_name'] == player]['winner_age']
    lose_age = df[df['loser_name'] == player]['loser_age']

    birthday = float(line[:-1].split(':')[1])
    start_year = birthday+min(min(win_age),min(lose_age),19)
    end_year = birthday+max([max(win_age),max(lose_age)])

    temp = {'birthday':birthday,
            'start_year':start_year,
            'end_year':end_year,
            }

    players_information[player] = temp
fig,ax = plt.subplots(figsize=(90,30))
sz = 105
ax.set_xlabel('Year', size = sz)
ax.set_ylabel('Log score', size = sz)
ax.set_ylim([3.3, 8.1])
ax.set_xlim([1969, 2025])
plt.xticks(size = sz)
plt.yticks(size = sz)
rc('text', usetex=True)
rc('font', family='serif')
year_tag = [('goldenrod', '--'),('red', '--'), ('salmon', '--'),\
            ('royalblue', '--'), ('limegreen', '--'), ('cyan', '--'),\
            ('magenta', '--'), ('springgreen', '--'), ('indianred', '--'), \
            ('darkcyan', '--')]
year_tag = [('goldenrod', '--'),('red', '--'), ('salmon', '--'),\
            ('royalblue', '--'), ('limegreen', '--'), ('cyan', '--'),\
            ('magenta', '--'), ('springgreen', '--'), ('indianred', '--'), \
            ('darkcyan', '--')]


for i,player in enumerate(players):

    start_year = players_information[player]['start_year']
    end_year = players_information[player]['end_year']
    birthday = players_information[player]['birthday']

    period = np.linspace(start_year, end_year, 200)
    
    u = u_plusDC[playerID[player]]
    score = age_effect(period-birthday)+u

    
    ax.plot(period,score,color = year_tag[i][0], 
            linestyle = year_tag[i][1], linewidth = 6, label = top_player[i])
ax.legend(prop={'size': 60}, title_fontsize=sz, loc = 'lower left', bbox_to_anchor=(0, 0))


plt.grid()
plt.savefig(Figure_name+'.png')
plt.show()
    