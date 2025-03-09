import numpy as np
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
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
            k = len(t)-1
        else:
            k = l
        for j in range(k):
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
                      P = False, detail = False,save_likelihood = False):
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
    if np.isnan(u).any() or np.isnan(v).any():
        print('The optimal solution does not exist')
    else:
        pass
    if save_likelihood:
        return L
    else:
        return u, v


## Pairwise case (Faster than the general algorithm)

### sigmoid function s
sig = lambda x: np.exp(x)/(1+np.exp(x))

### likelihood
def pair_likelihood(T,K,u,v = None):
    if len(v) == 0 or v is None:
        d = len(K[0].T)
        v = np.zeros(d)
    else:
        pass
    if type(K) is list:
        K = np.array([x[0,:]-x[1,:] for x in K])
    else:
        pass
    result = 0
    different_score = u[T][:,0]-u[T][:,1]+K@v
    p = sig(different_score)
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
def pair_fixv(T,K,v,n,win=None,lose=None,win_count=None, E = 1e-6 , I = 1000, u_initial = None,detail = False):
    T = np.copy(T)
    if win is None:
        win,lose,win_count = get_win(T,n)
    else:
        pass
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
    tem = u[T]
    s = tem[:,0]-tem[:,1] + K@v
    l1 = 1 - sig(s)
    l2 = K.T@ ((l1*(1-l1))[:, np.newaxis] * K)
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
### Similar to the first
def pair_alternative(T,X,n,d, u_initial = None ,v_initial = None,
                     E = 1e-6,Eu=1e-4,Ev=1e-8,I = 50,
                     P = False,detail = False,save_likelihood = False):
    
    K = np.array([x[0,:]-x[1,:] for x in X])
    T = np.array(T)
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
        u = pair_fixv(T, K, v, n, win, lose,win_count, 
                      u_initial = u,E=Eu,detail=detail)
    else:
        while error > E and i < 100 and not PL:
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {L[-1]}')
            else:
                pass
            v = pair_fixu(T, K, u, d, v_initial = v.copy(),E=Ev,I=I,detail=detail)
            u = pair_fixv(T, K, v, n, win, lose, win_count,
                           u_initial = u,E=Eu,I=I,detail=detail)
            l2 = pair_likelihood(T, K, u, v)
            L.append(l2)

            error = l2 - l1
            l1 = l2
            i += 1
    if np.isnan(u).any() or np.isnan(v).any():
        print('The optimal solution does not exist')
    else:
        pass
    if save_likelihood:
        return L
    else:
        return u, v

## Comprehensive method
def AM(T,X,n,d,u_initial = None, v_initial = None,P = False,E = 1e-3,Eu=1e-4,Ev=1e-8,detail = False,type = 'multi',I=50):
    if type == 'multi':
        u, v= multi_alternative(T,X,n,d, u_initial = u_initial, v_initial = v_initial, P = P,E = E,Eu=Eu,Ev=Ev,I=I,detail = detail)
    elif type == 'pair':
        u, v= pair_alternative(T,X,n,d, u_initial = u_initial , v_initial = v_initial, P = P,E = E,Eu=Eu,Ev=Ev,I=I,detail = detail)
    else:
        print('please choose \'multi\' or \'pair\'')
    return u,v