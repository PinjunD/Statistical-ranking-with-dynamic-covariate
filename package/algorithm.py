import numpy as np
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc

''' 
Notations
    T is the edge list (len=N), whose element is a list of index (with m_i)
    X is the covariates list (len=N), whose element is np.array (in R^{m_i*d})
    u is a np.array (shape = n) corresponding to intrinsic score (in R^n)
    v is a np.array (shape = d) corresponding to covariates coefficient (in R^n)
'''

#Multiple Case

## Algorithm

### Calculate log-likelihood 
'''
l = k : Top-k log-likelihood
l = None : Full log-likelihood
'''
def multi_likelihood(T,X,u,v = None,l = None):
    if len(v) == 0 or v is None:
        d = len(X[0].T)
        v = np.zeros(d)
    else:
        pass
    N = len(T)
    result = 0
    for i, t in enumerate(T):
        R = np.exp(u[t] + X[i] @ v)
        if l is None:
            l = len(t)-1
        else:
            pass
        for j in range(l):
            tem = R[j] / sum(R[j:])
            result += np.log(tem)
    return result/N

### W is a np.array (in R^n) representing the winning times of each items
def multi_Win(T,n):
    W = np.zeros((n))
    for i, t in enumerate(T):
        W[t[:-1]] += 1
    return W
### D_r is a list (len=N) containing the dynamic score in each competition
def multi_DynamicScore_Win(X,v):
    D_r = []
    for k in X:
        tem = np.exp(k@v)
        tem = tem/sum(tem)
        D_r.append(tem)
    return D_r
### Each step of optimizing u
def multi_update_R(R,W,D,T,n):
    M = np.zeros(n)
    for i, t in enumerate(T):
        d = D[i]
        tem = d*R[t]
        s1 = [1/sum(tem[i:]) for i in range(len(t)-1)]
        pair_denominator = [d[i]*sum(s1[:i+1]) for i in range(len(t))]
        M[t] += pair_denominator
    R_new = W/M
    R_new /= np.sum(R_new)
    return R_new

### The whole algorithm to optimize u
'''
'W' should be calculated by 'multiple_Win'
'E' is the accuracy of the iterations
'I' is the maxinum iteration times
'u_initial' is the initial value of u
You can set 'detail = True' to print the iteration times of u in algorithm
'''
def multi_fixv(T,X,v,n,W, E = 1e-6 , I = 50, u_initial = None, detail = False):
    D = multi_DynamicScore_Win(X, v)
    if u_initial is None:
        R = np.ones(n)/n
    else:
        R = np.exp(u_initial)/sum(np.exp(u_initial))
    i, error = 1, 1
    while error > E and i < I:
        R_new = multi_update_R(R,W,D,T,n)
        update = R_new - R
        error = sum(abs((update)))
        R = R_new
        i += 1
    if detail:
        print(f'u iteration times: {i}')
    else:
        pass
    u = np.log(R_new)
    u = u - np.mean(u)
    return u

### Each step of optimizing v
def multi_update_v(v,u,T,X,d):
    tem = 0
    H = np.zeros((d,d))
    for i, xx in enumerate(X):
        coe = np.exp(u[T[i]]+xx@v)
        for j in range(len(xx)-1):
            x = xx[j:]
            c = coe[j:]/sum(coe[j:])
            XX = x.T@c
            tem += xx[j] - XX
            D = x.T@np.diag(c)@x
            O = np.outer(XX,XX)
            H += D - O
    H = np.linalg.inv(H)
    update = H@tem
    return update

### The whole algorithm to optimize u
'''
'E' is the accuracy of the iterations
'I' is the maxinum iteration times
'v_initial' is the initial value of u
You can set 'detail = True' to print the iteration times of u in algorithm
'''
def multi_fixu(T,X,u,d, E=1e-6, I = 1000,v_initial = None,detail = False):
    if v_initial is None:
        v = np.zeros(d)
    else:
        v = v_initial[:]
        pass
    i, error = 1, 1
    while error > E*1000 and i < I:
        v_update = multi_update_v(v,u,T,X,d)
        v = v + v_update
        if d == 1:
            error = abs(v_update)
        else:
            error = sum(abs(v_update))
        i += 1
    if detail:
        print(f'v iteration times: {i} and v = {v}')
    else:
        pass
    return v
### Alternating maximization algorithm
'''
'E' is the accuracy of the iterations
'I' is the maxinum iteration times
'v_initial' is the initial value of u
You can set 'detail = True' to print the iteration times of u in algorithm
'''
def multi_alternative(T,X,n,d, 
                      u_initial = None ,v_initial = None,
                      E = 1e-6,Eu=1e-4,Ev=1e-8,I=50,
                      P = False, detail = False,save_mle = True):
    W = multi_Win(T,n)
    # Is PL?
    if d == 0 or P:
        PL = True
    else:
        PL = False
    # Initials
    if v_initial is None or P:
        v = np.zeros(d)
    else:
        v = v_initial
    if u_initial is None:
        u = np.zeros(n)
    else:
        u = u_initial
    l1 = multi_likelihood(T,X,u,v)
    L = [l1]
    i, error = 1, 1
    if PL:
        # PL
        u = multi_fixv(T, X, v, n, W,E=Eu,I=1000, u_initial = u)
    else:
        # PlusDC
        while error > E and i < 100:
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {L[-1]}')
            else:
                pass
            v1 = multi_fixu(T, X, u, d, v_initial=v.copy(),E=Ev,I=I,detail=detail)
            u1 = multi_fixv(T, X, v1, n, W, u_initial = u,I=I,E=Eu,detail=detail)
            u = u1
            v = v1
            l2 = multi_likelihood(T, X, u, v)
            L.append(l2)
            error = l2 - l1
            l1 = l2
            i += 1
    if save_mle:
        return L
    else:
        return u, v


## Pairwise case (Faster than the general algorithm)

### sigmoid function s
s = lambda x: np.exp(x)/(1+np.exp(x))

### likelihood
def pair_likelihood(T,X,u,v = None):
    if len(v) == 0 or v is None:
        d = len(X[0].T)
        v = np.zeros(d)
    else:
        pass
    N = len(T)
    result = 0
    different_score = u[T]+X@v
    p = s(different_score[:,0]-different_score[:,1])
    result = np.mean(np.log(p))
    return result
### Calculate the information of winning
def get_win(T,n):
    win = {i: 0 for i in range(n)}
    lose = {i: 0 for i in range(n)}
    win_count = np.zeros(n)
    lose_count = np.zeros(n)
    for i in range(n):
        win[i] = np.where(T[:,0]==i)[0]
        lose[i] = np.where(T[:,1]==i)[0]
        win_count[i] = len(win[i])
        lose_count[i] = len(lose[i])
    return win,lose,win_count

### Each step of updating u
def pair_update_R(R,T,K,v,win,lose,win_count,n):
    R1 = R[T[:,0]]
    R2 = R[T[:,1]]
    c = np.exp(-K@v)
    res = np.array([np.sum(1/(R2[win[i]]*c[win[i]]+R[i])) + \
                        np.sum(1/(R[i]+R1[lose[i]]/c[lose[i]])) for i in range(n)])
    #u_est = np.log(1/res*win_count)
    R = 1/res*win_count
    return R
### The whole algorithm of optimizing u
def pair_fixv(T,K,v,n,win,lose,win_count, E = 1e-6 , I = 1000, u_initial = None,detail = False):
    if u_initial is None:
        R = np.ones(n)/n
    else:
        R = np.exp(u_initial)/sum(np.exp(u_initial))
    i, error = 1, 1
    while error > E and i < I:
        R_new = pair_update_R(R, T, K, v,win,lose,win_count,n)
        #update = np.log(R_new / R)
        update = R_new - R
        error = sum(abs((update)))
        R = R_new
        i += 1
    u = np.log(R_new)
    u = u - np.mean(u)
    if detail:
        print(f'u iteration times: {i}')
    else:
        pass
    return u

### Each step of updating v
def pair_update_v(v,u,T,K):
    K = np.copy(K[:,0]-K[:,1])
    tem = u[T]
    s = tem[:,0]-tem[:,1] + K@v
    l1 = 1/(1+np.exp(s))
    l2 = K.T@ ((np.exp(s)/(1+np.exp(s))**2)[:, np.newaxis] * K)
    update = np.linalg.inv(l2)@K.T@l1
    return update

### The whole algorithm of optimizing u
def pair_fixu(T,K,u,d, E=1e-6, I = 1000,v_initial = None,detail = False):
    if v_initial is None:
        v = np.zeros(d)
    else:
        v = v_initial[:]
        pass
    i, error = 1, 1
    while error > E and i < I:
        v_update = pair_update_v(v, u, T, K)
        v = v + v_update
        if d == 1:
            error = abs(v_update)
        else:
            error = sum(abs(v_update))
        i += 1
    if detail:
        print(f'v iteration times: {i} and v = {v}')
    else:
        pass
    return v

def pair_alternative(T,X,n,d, u_initial = None ,v_initial = None,
                     E = 1e-6,Eu=1e-4,Ev=1e-8,I = 50,
                     P = False,detail = False,save_likelihood = False):
    
    K = X[:,0]-X[:,1]
    win,lose,win_count = get_win(T,n)
    # Is PL?
    if d == 0 or P:
        PL = True
    else:
        PL = False
    # Initials
    if v_initial is None or P:
        v = np.zeros(d)
    else:
        v = v_initial
    if u_initial is None:
        u = np.zeros(n)
    else:
        u = u_initial
    l1 = pair_likelihood(T,K,u,v)
    L = [l1]
    i, error = 1, 1

    if PL:
        u = pair_fixv(T,K,v,n,win,lose,win_count, 
                      u_initial = u,E=Eu,detail=detail)
    else:
        while error > E and i < 100 and not PL:
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {L[-1]}')
            else:
                pass
            v = pair_fixu(T, K, u, d, v_initial = v.copy(),E=Ev,I=I,detail=detail)
            u = pair_fixv(T, K, v, n, u_initial = u,E=Eu,I=I,detail=detail)
            l2 = pair_likelihood(T, K, u, v)
            L.append(l2)

            error = l2 - l1
            l1 = l2
            i += 1
    if save_likelihood:
        return L
    else:
        return u, v

def AM(T,X,n, d, u = None ,v = None,P = False,E = 1e-3,Eu=1e-4,Ev=1e-8,detail = False,type = 'multi',I=50,save = False):
    if type == 'multi':
        u, v= multi_alternative(T,X,n,d, u = u ,v = v,P = P,E = E,Eu=Eu,Ev=Ev,I=I,detail = detail)
    elif type == 'pair':
        u, v= pair_alternative(T,X,n,d, u = u ,v = v,P = P,E = E,Eu=Eu,Ev=Ev,I=I,detail = detail)
    else:
        print('please choose \'multi\' or \'pair\'')
    return u,v







###
def pair_hingeloss(T,K,u,v = None):
    if len(v) == 0 or v is None:
        d = len(K[0].T)
        v = np.zeros(d)
    else:
        pass
    N = len(T)
    result = 0
    for i, t in enumerate(T):
        R = np.exp(u[t] + K[i] @ v)
        tem = R[0] / sum(R)
        result += 1 - tem
    return result/N
def compute_auc(T,K,u,v = None):
    fpr,tpr = pair_compute_p(T,K,u,v)
    Auc = auc(fpr, tpr)
    return Auc
def pair_compute_p(T,X,u,v):
    p = []
    for i, t in enumerate(T):
        R = np.exp(u[t] + X[i] @ v)
        tem = R[0] / sum(R)
        p.append(tem)
    p = np.array(p)
    y = np.ones(len(p))
    p = np.concatenate((p,1-p))
    ytrue = np.concatenate((y,1-y))
    fpr, tpr, thresholds = roc_curve(ytrue, p)
    return fpr, tpr


def pair_covariate_coeffecient(T,K,u,d, E=1e-6, I = 1000,v_initial = None,detail = False):
    KK = np.array([k[0] - k[1] for k in K])
    if v_initial is None:
        v = np.zeros(d)
    else:
        v = v_initial[:]
        pass
    i, error = 1, 1
    while error > E and i < I:
        v_update = pair_update_v(v, u, T, KK)
        v = v + v_update
        if d == 1:
            error = abs(v_update)
        else:
            error = sum(abs(v_update))
        i += 1
    if detail:
        print(f'v iteration times: {i} and v = {v}')
    else:
        pass
    return v



def Newton_pair(T,K,u,d, I = 100,v_initial = None,detail = False):
    K = np.array([k[0] - k[1] for k in K])
    if v_initial is None:
        v = np.zeros(d)
    else:
        v = v_initial[:]
        pass
    i= 1
    V = [v]
    while  i < I:
        v_update = pair_update_v(v, u, T, K)
        v = v + v_update
        V.append(v)
        i += 1
    if detail:
        print(f'v iteration times: {i} and v = {v}')
    else:
        pass
    return V

def MM_pair(T,K,v,n, I = 5, u_initial = None,detail = False):
    K = np.array([k[0] - k[1] for k in K])
    if u_initial is None:
        R = np.ones(n)/n
    else:
        R = np.exp(u_initial)/sum(np.exp(u_initial))
    i, error = 1, 1
    u = np.log(R)
    u = u - np.mean(u)
    U = [u]
    while i < I:
        R_new = pair_update_R(R, T, K, v)
        #update = np.log(R_new / R)
        R = R_new
        i += 1
        u = np.log(R)
        u = u - np.mean(u)
        U.append(u)
    if detail:
        print(f'u iteration times: {i}')
    else:
        pass
    return U


def estimate_v(X,d, E=1e-6, I = 1000,v_initial = None,detail = False):
    if v_initial is None:
        v = np.zeros(d)
    else:
        v = v_initial[:]
        pass
    i, error = 1, 1
    while error > E and i < I:
        l1 = 1/(1+np.exp(X@v))
        l2 = X.T@ ((l1*(1-l1))[:, np.newaxis] * X)
        v_update = np.linalg.inv(l2)@X.T@l1
        v = v + v_update
        if d == 1:
            error = abs(v_update)
        else:
            error = sum(abs(v_update))
        i += 1
    if detail:
        print(f'v iteration times: {i} and v = {v}')
    else:
        pass
    return v

###CV
def tennis_cross_validation(T,cov,n,subset,data_path,CV_name):
    N = len(T)
    Ttrain = [T[i] for i in range(N) if i not in subset]
    Xtrain = [cov[i] for i in range(N) if i not in subset]
    Ttest = [T[i] for i in range(N) if i in subset]
    Xtest = [cov[i] for i in range(N) if i in subset]
    u_pl,v_pl = AM(Ttrain,Xtrain,n,P=True,Eu=1e-5,type = 'pair')
    if np.isnan(u_pl).any():
        pass
    else:
        u_plusDC,v_plusDC = AM(Ttrain,Xtrain,n,
                                    E=1e-5,Eu=1e-5,Ev=1e-12,
                                    I=52,type = 'pair')
        cross_entropy_pl = pair_likelihood(Ttest,Xtest,u_pl,v_pl)
        cross_entropy_plusDC = pair_likelihood(Ttest,Xtest,u_plusDC,v_plusDC)
        hinge_loss_pl = pair_hingeloss(Ttest,Xtest,u_pl,v_pl)
        hinge_loss_plusDC = pair_hingeloss(Ttest,Xtest,u_plusDC,v_plusDC)
        logcosh_loss_pl = pair_logcoshloss(Ttest,Xtest,u_pl,v_pl)
        logcosh_loss_plusDC = pair_logcoshloss(Ttest,Xtest,u_plusDC,v_plusDC)
        auc_pl = compute_auc(Ttest,Xtest,u_pl,v_pl)
        auc_plusDC = compute_auc(Ttest,Xtest,u_plusDC,v_plusDC)

        #with open(data_path+"CV_filename",'a') as f:
        with open(data_path+CV_name['Cross entropy'],'a') as f:
            f.write(str(cross_entropy_pl))
            f.write(';')
            f.write(str(cross_entropy_plusDC))
            f.write('\n')
        with open(data_path+CV_name['Hinge loss'],'a') as f:
            f.write(str(hinge_loss_pl))
            f.write(';')
            f.write(str(hinge_loss_plusDC))
            f.write('\n')
        with open(data_path+CV_name['Log-cosh loss'],'a') as f:
            f.write(str(logcosh_loss_pl))
            f.write(';')
            f.write(str(logcosh_loss_plusDC))
            f.write('\n')
        with open(data_path+CV_name['AUC'],'a') as f:
            f.write(str(auc_pl))
            f.write(';')
            f.write(str(auc_plusDC))
            f.write('\n')
def horse_cross_validation(T,cov,n,subset,data_path,CV_filename):
    N = len(T)
    Ttrain = [T[i] for i in range(N) if i not in subset]
    Xtrain = [cov[i] for i in range(N) if i not in subset]
    Ttest = [T[i] for i in range(N) if i in subset]
    Xtest = [cov[i] for i in range(N) if i in subset]


    u_pl,v_pl = AM(Ttrain,Xtrain,n,P=True,Eu=1e-5,detail=True)
    u_plusDC,v_plusDC = AM(Ttrain,Xtrain,n,
                                 E=1e-5,Eu=1e-5,Ev=1e-12,
                                 I=52)
    
    full_likelihood_pl = multi_likelihood(Ttest,Xtest,u_pl,v_pl)
    top_likelihood_pl = multi_likelihood_one(Ttest,Xtest,u_pl,v_pl)
    three_likelihood_pl = multi_likelihood_three(Ttest,Xtest,u_pl,v_pl)
    likelihood_pl = [full_likelihood_pl,top_likelihood_pl,three_likelihood_pl]

    full_likelihood_plusDC = multi_likelihood(Ttest,Xtest,u_plusDC,v_plusDC)
    top_likelihood_plusDC = multi_likelihood_one(Ttest,Xtest,u_plusDC,v_plusDC)
    three_likelihood_plusDC = multi_likelihood_three(Ttest,Xtest,u_plusDC,v_plusDC)
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

    with open(data_path+CV_filename,'a') as f:
        for values in results.values():
            for value in values:
                f.write(str(value)+',')
            f.write(';')
        f.write('\n')



def binomial_test(p,num):
    pl = p[p!=0.5]
    pp = np.where(pl > 0.5, 1 - pl, pl)
    
    kmeans = KMeans(n_clusters=num, random_state=42)
    kmeans.fit(pp.reshape(-1, 1))
    labels = kmeans.labels_
    R = []
    for i in range(num):
        p_temp = pl[labels == i]
        pp_temp = pp[labels == i]
        positive = sum(p_temp<0.5) 
        r = (positive-sum(pp_temp))/np.sqrt(sum(pp_temp*(1-pp_temp))) 
        R.append(r)
    chi = sum(np.array(R)**2)
    return chi,len(R)
def test_statistics(X,d,num,v_true = None):
    XX = X.copy()
    if v_true is None:
        pass
    else:
        p = s(XX@v_true)
        Y = np.random.binomial(1,p)
        XX[Y==0,:] = -XX[Y==0,:]
    v_hat = estimate_v(XX,d)
    p_hat = s(XX@v_hat)
    chi,dof = binomial_test(p_hat,num)
    p_value = 1-chi2.cdf(chi,dof-1)
    return (chi,p_value)
