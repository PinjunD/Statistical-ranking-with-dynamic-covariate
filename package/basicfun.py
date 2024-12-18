import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
def order(R):
    m = len(R)
    o = [m-1 for _ in range(m)]
    for i in range(m-1):
        R = R/sum(R)
        win = np.random.multinomial(1,R)
        ind = np.nonzero(win == 1)[0][0]
        o[ind] = i
        R[ind] = 0
    return o
def u_uniform(n):
    u = np.random.uniform(-0.5, 0.5, n)
    u += - np.mean(u)
    return u
def x_center(n,d):
    x = np.random.uniform(-0.5, 0.5, (n,d))
    x += - np.mean(x)
    return x
def x_generator(x):
    m, d = x.shape
    variables = np.random.normal(0,1,size=(m,d))
    return variables

###save data
def create_file():
    folder_name = "new_folder"
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path,current_dir


###plot

rc('text', usetex=True)
rc('font', family='serif')
def tex_infty(u,b=None):
    if b is None:
        a = rf"$\Vert\widehat{{\mathbf{{{u}}}}} - \mathbf{{{u}}}^*\Vert_\infty$"
    else:
        a = rf"$\Vert\widehat{{{u}}}_{b} - {u}^*_{b}\Vert_\infty$"
    return a
def find_all_notepad(path = None,feature = ''):
    F = []
    if path == None:
        path = os.getcwd()
    else:
        pass
    try:
        for file in os.listdir(path):
            if '.txt' in file and feature in file:
                file_path = os.path.join(path, file)
                F.append(file_path)
            else:
                pass
    except:
        print("please enter a proper path! ")
    return F


def read_notepad(path_list):
    result = []
    for path in path_list:
        lines = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                lines.append(line.strip())
        result += lines
    return result
def dealwith(T):
    D = []
    for t in T:
        d = []
        A = t.split(',')
        for a in A[:3]:
            d.append(float(a))
        D.append(d)
    try:
        Data = np.array(D)
    except:
        Data = D
    return Data

def pro_data(Data):
    n_list = list(set(Data[:, 0]))
    n_list.sort()
    N = []
    Mean = []
    Std = []
    for n in n_list:
        D = Data[Data[:, 0] == n]
        D = D[:, [1,2]]
        M = np.mean(D, axis=0)
        S = np.std(D, axis=0)
        N.append(int(n))
        Mean.append(M)
        Std.append(S)
    Mean = np.array(Mean)
    Std = np.array(Std)
    return N, Mean, Std
def plot(LLabel):
    current_dir = os.getcwd()
    data_dir = current_dir + '\\' + 'synthesis' + '\\'
    save_dir = current_dir + '\\' + 'image' + '\\'
    Data = []
    N = []
    M = []
    S = []
    Label = [f'{LLabel}']
    for l in Label:
        L = find_all_notepad(path = data_dir,feature=l)
        Text = read_notepad(L)
        D = dealwith(Text)
        Data.append(D)
        n, m, s = pro_data(D)
        N.append(n)
        M.append(m)
        S.append(s)
    N = N[0]

    #label = ['$N_2 = 0.15n^2$','$N_1 = 2n^\\frac{3}{2}$',]
    label = [f'{LLabel} $\\mathbf{{\widehat{{u}}}}$ error', 
             f'{LLabel} $\\mathbf{{\widehat{{v}}}}$ error']
    y_ticks = [[0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8],[0, 0.01, 0.02, 0.03,0.04]]
    color = ['blue', 'orange']
    name = ['u','v']
    for j in range(2):
        Nm = name[j]
        fig, ax = plt.subplots(figsize=(10.8, 10))

        ax.set_position([0.155, 0.125, 0.8, 0.8])
        for i in range(len(Label)):
            ax.plot(N[:], M[i][:, j], marker='o', linestyle='-', label=label[i],color=color[j])
            ax.fill_between(N, M[i][:, j] - S[i][:, j], M[i][:, j] + S[i][:, j], alpha=0.2, color=color[j])
        sz = 36
        ax.set_title(label[j], fontsize=sz+4)
        ax.set_xlabel('Numbers of items $n$', size=sz)
        ax.set_ylabel(tex_infty(Nm), size=sz)
        plt.xticks(range(200, 1200, 200), size=sz)
        plt.yticks(y_ticks[j], size=sz)

        plt.grid()
        plt.savefig(save_dir+f'{LLabel}-{Nm}.pdf', format='pdf', dpi=300, transparent=True)

        plt.show()
if __name__ == '__main__':
    plot('NURHM')



