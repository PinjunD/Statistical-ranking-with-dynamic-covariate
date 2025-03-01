import numpy as np
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc

def multi_likelihood_three(T,K,u,v = None):
    if len(v) == 0 or v is None:
        d = len(K[0].T)
        v = np.zeros(d)
    else:
        pass
    N = len(T)
    result = 0
    for i, t in enumerate(T):
        R = np.exp(u[t] + K[i] @ v)
        for j in range(3):
            tem = R[j] / sum(R[j:])
            result += np.log(tem)
    return result/N
def multi_likelihood_one(T,K,u,v = None):
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
        result += np.log(tem)
    return result/N

##

def multi_likelihood(T,K,u,v = None):
    if len(v) == 0 or v is None:
        d = len(K[0].T)
        v = np.zeros(d)
    else:
        pass
    N = len(T)
    result = 0
    for i, t in enumerate(T):
        R = np.exp(u[t] + K[i] @ v)
        for j in range(len(t)-1):
            tem = R[j] / sum(R[j:])
            result += np.log(tem)
    return result/N
def multi_Win(T,n):
    W = np.zeros((n))
    for i, t in enumerate(T):
        W[t[:-1]] += 1
    return W
def multi_DynamicScore_Win(K,v):
    D_r = []
    for k in K:
        tem = np.exp(k@v)
        tem = tem/sum(tem)
        D_r.append(tem)
    return D_r
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
def multi_fixv(T,K,v,n,W, E = 1e-6 , I = 50, utr = None, detail = False):
    D = multi_DynamicScore_Win(K, v)
    if utr is None:
        R = np.ones(n)/n
    else:
        R = np.exp(utr)/sum(np.exp(utr))
    i, error = 1, 1
    while error > E and i < I:
        R_new = multi_update_R(R,W,D,T,n)
        update = R_new - R
        error = sum(abs((update)))
        R = R_new
        i += 1
    if detail:
        print(f'u iterative times: {i}')
    else:
        pass
    u = np.log(R_new)
    u = u - np.mean(u)
    return u
###############
def multi_update_v(v,u,T,K,d):
    tem = 0
    H = np.zeros((d,d))
    for i, X in enumerate(K):
        coe = np.exp(u[T[i]]+X@v)
        for j in range(len(X)-1):
            x = X[j:]
            c = coe[j:]/sum(coe[j:])
            XX = x.T@c
            tem += X[j] - XX
            D = x.T@np.diag(c)@x
            O = np.outer(XX,XX)
            H += D - O
    H = np.linalg.inv(H)
    update = H@tem
    return update
def multi_fixu(T,K,u,d, E=1e-6, I = 1000,vtr = None,detail = False):
    if vtr is None:
        v = np.zeros(d)
    else:
        v = vtr[:]
        pass
    i, error = 1, 1
    while error > E*1000 and i < I:
        v_update = multi_update_v(v,u,T,K,d)
        v = v + v_update
        if d == 1:
            error = abs(v_update)
        else:
            error = sum(abs(v_update))
        i += 1
    if detail:
        print(f'v iterative times: {i} and v = {v}')
    else:
        pass

    return v



##

def pair_likelihood(T,K,u,v = None):
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
        result += np.log(tem)
    return result/N

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
def pair_logcoshloss(T,K,u,v = None):
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
        tem = np.log(np.cosh(1-tem))
        result += tem
    return result/N
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
def compute_auc(T,K,u,v = None):
    fpr,tpr = pair_compute_p(T,K,u,v)
    Auc = auc(fpr, tpr)
    return Auc
def pair_false_prediction_rate(T,K,u,v = None):
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
        tem = tem<=0.5
        result += tem
    return result/N

def pair_update_R(R,T,K,v):
    M = np.zeros_like(R)
    W = np.zeros_like(R)
    coe = np.exp(K@v)
    for i, t in enumerate(T):
        W[t[0]] += 1
        M[t[0]] += coe[i]/(coe[i]*R[t[0]]+R[t[1]])
        M[t[1]] += 1/(coe[i]*R[t[0]]+R[t[1]])
    R_new = W/M
    R_new /= np.sum(R_new)
    return R_new

def pair_fixv(T,K,v,n, E = 1e-6 , I = 1000, utr = None,detail = False):
    if utr is None:
        R = np.ones(n)/n
    else:
        R = np.exp(utr)/sum(np.exp(utr))
    i, error = 1, 1
    while error > E and i < I:
        R_new = pair_update_R(R, T, K, v)
        #update = np.log(R_new / R)
        update = R_new - R
        error = sum(abs((update)))
        R = R_new
        i += 1
    u = np.log(R_new)
    u = u - np.mean(u)
    if detail:
        print(f'u iterative times: {i}')
    else:
        pass
    return u

def pair_update_v(v,u,T,K):
    tem = u[T]
    s = tem[:,0]-tem[:,1] + K@v
    l1 = 1/(1+np.exp(s))
    l2 = K.T@ ((np.exp(s)/(1+np.exp(s))**2)[:, np.newaxis] * K)
    update = np.linalg.inv(l2)@K.T@l1
    return update
def pair_fixu(T,K,u,d, E=1e-6, I = 1000,vtr = None,detail = False):
    if vtr is None:
        v = np.zeros(d)
    else:
        v = vtr[:]
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
        print(f'v iterative times: {i} and v = {v}')
    else:
        pass
    return v



def AM(T,K,n, d =None, u = None ,v = None,P = False,E = 1e-3,Eu=1e-4,Ev=1e-8,detail = False,type = 'multi',I=50,save = False):
    if type == 'multi':
        u, v= multi_alternative(T,K,n,d, u = u ,v = v,P = P,E = E,Eu=Eu,Ev=Ev,I=I,detail = detail,save = save)
    elif type == 'pair':
        u, v= pair_alternative(T,K,n,d, u = u ,v = v,P = P,E = E,Eu=Eu,Ev=Ev,I=I,detail = detail,save = save)
    else:
        print('please choose \'multi\' or \'pair\'')
    return u,v

def pair_alternative(T,K,n,d=None, u = None ,v = None,P = False,
                     E = 1e-6,Eu=1e-4,Ev=1e-8,I = 50,detail = False,save = False):
    KK = np.array([k[0] - k[1] for k in K])
    if d is None:
        d = KK.shape[-1]
    else:
        pass
    if d == 0 or P:
        PL = True
    else:
        PL = False
    if v is None or P:
        v = np.zeros(d)
    else:
        pass
    if u is None:
        u = np.zeros(n)
    else:
        pass
    l1 = pair_likelihood(T,K,u,v)
    i, error = 1, 1
    if save:
        #U = [u]
        #V = [v]
        L = [l1]
        while error > E and i < 100 and not PL:
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {l1}')
            else:
                pass
            v1 = pair_fixu(T, KK, u, d, vtr = v.copy(),E=Ev,I=I,detail=detail)
            u1 = pair_fixv(T, KK, v1, n, utr = u,E=Eu,I=I,detail=detail)
            
            u = u1
            v = v1
            l2 = pair_likelihood(T, K, u, v)
            L.append(l2)
            error = l2 - l1
            l1 = l2
            i += 1
        u = pair_fixv(T, KK, v, n, utr = u,E=Eu,detail=detail)
        return L
    else:
        while error > E and i < 100 and not PL:
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {l1}')
            else:
                pass
            v1 = pair_fixu(T, KK, u, d, vtr = v.copy(),E=Ev,I=I,detail=detail)
            u1 = pair_fixv(T, KK, v1, n, utr = u,E=Eu,I=I,detail=detail)
            
            u = u1
            v = v1
            l2 = pair_likelihood(T, K, u, v)
            error = l2 - l1
            l1 = l2
            i += 1
        u = pair_fixv(T, KK, v, n, utr = u,E=Eu,detail=detail)
        return u, v

def multi_alternative(T,K,n, d = None, u = None ,v = None,P = False,
                      E = 1e-6,Eu=1e-4,Ev=1e-8,I=50,detail = False,save = False):
    if n is None:
        node = list(set(sum(T, [])))
        n = len(node)
        T = [[node.index(e) for e in t] for t in T]
    else:
        pass
    if d is None:
        d = len(K[0].T)
    else:
        pass
    W = multi_Win(T,n)
    if d == 0 or P:
        PL = True
    else:
        PL = False
    if v is None or P:
        v = np.zeros(d)
    else:
        pass
    if u is None:
        u = np.zeros(n)
    else:
        pass
    l1 = multi_likelihood(T,K,u,v)
    i, error = 1, 1
    if save:
        U = [u]
        V = [v]
        while error > E and i < 100 and not PL:
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {l1}')
            else:
                pass
            v1 = multi_fixu(T, K, u, d, vtr=v.copy(),E=Ev,I=I,detail=detail)
            u1 = multi_fixv(T, K, v1, n, W, utr = u,I=I,E=Eu,detail=detail)
            u = u1
            v = v1
            U.append(u)
            V.append(v)
            l2 = multi_likelihood(T, K, u, v)
            error = l2 - l1
            l1 = l2
            i += 1
        u = multi_fixv(T, K, v, n, W,E=Eu,I=1000, utr = u)
        if P:
            pass
        else:
            v = multi_fixu(T, K, u, d, E=Ev,vtr=v.copy())
        return U, V
    else:
        while error > E and i < 100 and not PL:
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {l1}')
            else:
                pass
            v1 = multi_fixu(T, K, u, d, vtr=v.copy(),E=Ev,I=I,detail=detail)
            u1 = multi_fixv(T, K, v1, n, W, utr = u,I=I,E=Eu,detail=detail)
            u = u1
            v = v1
            l2 = multi_likelihood(T, K, u, v)
            error = l2 - l1
            l1 = l2
            i += 1
        u = multi_fixv(T, K, v, n, W,E=Eu,I=1000, utr = u)
        if P:
            pass
        else:
            v = multi_fixu(T, K, u, d, E=Ev,vtr=v.copy())
        return u, v



###
def pair_covariate_coeffecient(T,K,u,d, E=1e-6, I = 1000,vtr = None,detail = False):
    KK = np.array([k[0] - k[1] for k in K])
    if vtr is None:
        v = np.zeros(d)
    else:
        v = vtr[:]
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
        print(f'v iterative times: {i} and v = {v}')
    else:
        pass
    return v



def Newton_pair(T,K,u,d, I = 100,vtr = None,detail = False):
    K = np.array([k[0] - k[1] for k in K])
    if vtr is None:
        v = np.zeros(d)
    else:
        v = vtr[:]
        pass
    i= 1
    V = [v]
    while  i < I:
        v_update = pair_update_v(v, u, T, K)
        v = v + v_update
        V.append(v)
        i += 1
    if detail:
        print(f'v iterative times: {i} and v = {v}')
    else:
        pass
    return V

def MM_pair(T,K,v,n, I = 5, utr = None,detail = False):
    K = np.array([k[0] - k[1] for k in K])
    if utr is None:
        R = np.ones(n)/n
    else:
        R = np.exp(utr)/sum(np.exp(utr))
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
        print(f'u iterative times: {i}')
    else:
        pass
    return U


def estimate_v(X,d, E=1e-6, I = 1000,vtr = None,detail = False):
    if vtr is None:
        v = np.zeros(d)
    else:
        v = vtr[:]
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
        print(f'v iterative times: {i} and v = {v}')
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
s = lambda x: np.exp(x)/(1+np.exp(x))



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
