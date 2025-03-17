# Log score dynamics of the top ten players ranked by model (1.2). 
# For each player, we plot their log scores as a function 
# of time starting from his career and ending at his retirement.
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

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

# loading data
if __name__ == "__main__":
    print('In this program, we plot the log-scores of tennis players over time.')
    time.sleep(2)
    print('\n'+"="*10+'Data Loading'+"="*10)
    print('Loading data from \'data\\tennis(preprocessed).csv\'...')
    df = pd.read_csv(data_file_path('tennis(preprocessed).csv'),low_memory=False)
    ## players ID
    players = df[['winner_name','loser_name']]
    counts = pd.Series(players.values.ravel()).value_counts()
    playerID = {value: index for index, value in enumerate(counts.index.tolist())}
    playerID_str = {value: str(index) for index, value in enumerate(counts.index.tolist())}
    n = len(playerID)
    ## matches
    name_columns = ['winner_name','loser_name']
    Matches = df[name_columns].replace(playerID_str)
    T = np.array(Matches).astype(int).tolist()
    N = len(T)
    ## age
    age_columns = ['winner_age','loser_age']
    Age = np.array(df[age_columns])
    print('Complete!')
    time.sleep(1)
    
    # Data analysis
    print('\n'+"="*10+'Data analysis'+"="*10)
    print('We have selected the optimal basis functions based on the BIC.')
    print('Here, (a,ρ) ∈ {(25,0.01),(25,0.03),(30,0.01),(35,0.01)}.')
    A = [25,30,35]
    Lamb = [0.01, 0.03]
    parameters = [(a,lamb) for a in A for lamb in Lamb]
    gauss_kernel = lambda x,a,lamb: np.exp(-lamb*(x-a)**2)
    gauss_kernel_set = lambda x: np.array([gauss_kernel(x,a,lamb) for (a,lamb) in parameters])
    cov = [gauss_kernel_set(age).T for age in Age]
    s = (0, 1, 2, 4)
    d = len(s)
    X = [x[:,s] for x in cov]
    ## fit PlusDC
    print('We use PlusDC to fit the data using various combinations of the four bases.')
    print('Running...')
    u_plusDC,v_plusDC = algorithm.AM(T,X,n,d,TYPE = 'pair')
    plusDC_top10 = np.argsort(u_plusDC)[-10:][::-1]
    u_t10_plusDC = u_plusDC[plusDC_top10]
    top_player = []
    print(f'Complete! v = {v_plusDC}')
    print('Next, we fit the PlusDC model and identify the top 10 players.')
    print("="*20)
    for i,index in enumerate(plusDC_top10):
        player_name = [key for key, value in playerID.items() if value == index][0]
        top_player.append(player_name)
        time.sleep(1)
        print(f'top-{i+1}: player: {player_name}, score: {u_t10_plusDC[i]}')

    ## Fit age effect
    age_effect = lambda t: sum([v_plusDC[i]*gauss_kernel(t,parameters[ss][0],parameters[ss][1]) for i,ss in enumerate(s)])
    total_num = 10
    players = top_player[:total_num]
    players_information = {}

    ## load birthday information
    print('Loading birthday information...')
    with open(data_file_path('birthday.txt'),'r') as f:
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
    print('Complete!')
    print('Plot...')
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
    plt.savefig(save_path('.pdf'))
    print("Complete!")
    time.sleep(1)
    print(f'We have saved the results as {Figure_name}.pdf.')
        