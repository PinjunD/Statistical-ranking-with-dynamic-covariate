import numpy as np
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
    print(N)
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
        #print("s:",t[:-1])
        W[t[:-1]] += 1
    return W
def multi_DynamicScore_Win(K,v):
    D_r = []
    for k in K:
        #print("s:",t[:-1])
        tem = np.exp(k@v)
        tem = tem/sum(tem)
        D_r.append(tem)
    return D_r
def multi_updata_R(R,W,D,T,n):
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
        R_new = multi_updata_R(R,W,D,T,n)
        #updata = np.log(R_new / R)
        updata = R_new - R
        error = max(abs((updata)))
        #print(error)
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
def multi_updata_v(v,u,T,K,d):
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
    updata = H@tem
    return updata
def multi_fixu(T,K,u,d, E=1e-6, I = 1000,vtr = None,detail = False):
    if vtr is None:
        v = np.zeros(d)
    else:
        v = vtr[:]
        pass
    i, error = 1, 1
    l0 = multi_likelihood(T, K, u, v)
    while error > E*1000 and i < I:
        v_updata = multi_updata_v(v,u,T,K,d)
        #print(v_updata)
        v = v + v_updata
        if d == 1:
            error = abs(v_updata)
        else:
            error = max(abs(v_updata))
        #print(error)
        l1 = multi_likelihood(T, K, u, v)
        """if l1>=l0:
            print(f'True:{l1}')
        else:
            print(f'False:{l1-l0}')"""
        l0 = l1
        i += 1
    if detail:
        print(f'v iterative times: {i} and v = {v}')
    else:
        pass

    return v

def multi_alternative(T,K,n, d = None, u = None ,v = None,P = False,
                      E = 1e-6,Eu=1e-4,Ev=1e-8,I=50,detail = False):
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
    while error > E and i < 100 and not PL:
        if detail:
            print('-'*5+f'{i}'+'-'*5)
            print(f'log-likelihood: {l1}')
        else:
            pass
        u1 = multi_fixv(T, K, v, n, W, utr = u,I=I,E=Eu,detail=detail)
        v1 = multi_fixu(T, K, u, d, vtr=v.copy(),E=Ev,detail=detail)
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
def pair_updata_R(R,T,K,v):
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

def pair_fixv(T,K,v,n, E = 1e-6 , I = 50, utr = None,detail = False):
    if utr is None:
        R = np.ones(n)/n
    else:
        R = np.exp(utr)/sum(np.exp(utr))
    i, error = 1, 1
    while error > E and i < I:
        R_new = pair_updata_R(R, T, K, v)
        #updata = np.log(R_new / R)
        updata = R_new - R
        error = max(abs((updata)))
        R = R_new
        i += 1
    u = np.log(R_new)
    u = u - np.mean(u)
    if detail:
        print(f'u iterative times: {i}')
    else:
        pass
    return u

def pair_updata_v(v,u,T,K):
    tem = u[T]
    s = tem[:,0]-tem[:,1] + K@v
    l1 = 1/(1+np.exp(s))
    l2 = K.T@ ((np.exp(s)/(1+np.exp(s))**2)[:, np.newaxis] * K)
    updata = np.linalg.inv(l2)@K.T@l1
    return updata
def pair_fixu(T,K,u,d, E=1e-6, I = 1000,vtr = None,detail = False):
    if vtr is None:
        v = np.zeros(d)
    else:
        v = vtr[:]
        pass
    i, error = 1, 1
    while error > E and i < I:
        v_updata = pair_updata_v(v, u, T, K)
        v = v + v_updata
        if d == 1:
            error = abs(v_updata)
        else:
            error = max(abs(v_updata))
        i += 1
    if detail:
        print(f'v iterative times: {i} and v = {v}')
    else:
        pass
    return v

def pair_alternative(T,K,n,d=None, u = None ,v = None,P = False,
                     E = 1e-6,Eu=1e-4,Ev=1e-8,I = 50,detail = False):
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
    while error > E and i < 100 and not PL:
        
        if detail:
            print('-'*5+f'{i}'+'-'*5)
            print(f'log-likelihood: {l1}')
        else:
            pass
        u1 = pair_fixv(T, KK, v, n, utr = u,E=Eu,I=I,detail=detail)
        v1 = pair_fixu(T, KK, u1, d, vtr = v.copy(),E=Ev,detail=detail)
        
        u = u1
        v = v1
        l2 = pair_likelihood(T, K, u, v)
        error = l2 - l1
        l1 = l2
        i += 1
        I += 10*i
    u = pair_fixv(T, KK, v, n, utr = u,E=Eu,I=1000,detail=detail)
    return u, v

def AM(T,K,n, d =None, u = None ,v = None,P = False,E = 1e-3,Eu=1e-4,Ev=1e-8,detail = False,type = 'multi',I=50):
    if type == 'multi':
        u, v= multi_alternative(T,K,n,d, u = u ,v = v,P = P,E = E,Eu=Eu,Ev=Ev,I=I,detail = detail)
    elif type == 'pair':
        u, v= pair_alternative(T,K,n,d, u = u ,v = v,P = P,E = E,Eu=Eu,Ev=Ev,I=I,detail = detail)
    else:
        print('please choose \'multi\' or \'pair\'')
    return u,v

