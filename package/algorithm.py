import numpy as np
from typing import Callable, Dict, Any, List


# Alternating maximization
def AM(
    hyperedges_list: List[List[int]],
    covariates_list: List[np.ndarray],
    n: int,
    d: int, 
    u_initial: np.ndarray = None,
    v_initial: np.ndarray = None,
    E: float = 1e-10,
    Eu: float = 1e-10,
    Ev: float = 1e-10,
    I: int = 50,
    detail: bool = False,
    PL: bool = False,
    TYPE: str ='multi'
):
    '''Alternating maximization:

    Args:
        T (list of len(N)): The list of edges (list of len(m_i)).
        X (list of len(n)): The list of covariates (d dimensional np.array)
        n (int): The number of items.
        d (int): The dimension of covariates.
        u_initial (np.array of shape(n)): The initial value of u.
        v_initial (np.array of shape(v)): The initial value of v.
        E (float): The accuracy of the iterations in alternating algorithm.
        Eu (float): The accuracy of the iterations in updating u.
        Ev (float): The accuracy of the iterations in updating v.
        I (int): The maxinum iteration times
        detail (bool): Print the iteration details?
        P (bool): Fit PL?
        save_likelihood (bool): Save log-likelihood?
        type (str): 'multi' or 'pair'?

    Return:
        u,v (np.array of shape(n),np.array of shape(d)): The optimizer.
    '''
    if TYPE == 'multi':
        u, v= multi_alternative(hyperedges_list, covariates_list, n, d, 
                            u_initial = u_initial, v_initial = v_initial, 
                            PL = PL,E = E,Eu = Eu,Ev = Ev,I = I,detail = detail)
    elif TYPE == 'pair':
        u, v= pair_alternative(hyperedges_list, covariates_list, n, d, 
                            u_initial = u_initial, v_initial = v_initial, 
                            PL = PL,E = E,Eu = Eu,Ev = Ev,I = I,detail = detail)
    else:
        print('please choose \'multi\' or \'pair\'')
    return u, v

def multi_likelihood(
        hyperedges_list: List[List[int]],
        covariates_list: List[np.ndarray],
        u: np.ndarray,
        v: np.ndarray = None,
        l: int = None
    ):
    """ Compute the log-likelihood in multiple case:
    Args:
        hyperedges_set (List[List[int]]): A list as a set containing each comparison ranking. (Dimension: N(n))
        covariates_set (List[np.ndarray]): A list as all the comparison rankings. (Dimension: N(n))
        u (np.ndarray): An np.ndarray as the intrinsic score. (Dimension: n)
        v (np.ndarray): An np.ndarray as the coefficient of covariates.
        l (int or None): The Top-k log-likelihood or Full log-likelihood.

    Returns:
        L (float): A float as log-likelihood.
    
    """
    if v is None:
        d = len(covariates_list[0].T)
        v = np.zeros(d)
    else:
        pass
    N = len(hyperedges_list)
    result = 0
    for i, t in enumerate(hyperedges_list):
        R = np.exp(u[t] + covariates_list[i] @ v)
        if l is None:
            k = len(t)-1
        else:
            k = l
        for j in range(k):
            tem = R[j] / sum(R[j:])
            result += np.log(tem)
    L = result/N
    return L

def multi_alternative(
        hyperedges_list: List[List[int]],
        covariates_list: List[np.ndarray],
        n: int,
        d: int, 
        u_initial: np.ndarray = None ,
        v_initial: np.ndarray = None,
        E: float = 1e-6,
        Eu: float = 1e-4,
        Ev: float = 1e-8,
        I: int = 50,
        detail: bool = False,
        PL: bool = False,
        save_likelihood: bool = False):
    '''Alternating maximization in multiple case:

    Args:
        hyperedges_set (List[List[int]]): A list as a set containing each comparison ranking. (Dimension: N(n))
        covariates_set (List[np.ndarray]): A list as all the comparison rankings. (Dimension: N(n))
        n (int): An int as the number of items.
        d (int): An int as the dimension of covariates.
        u_initial (np.ndarray): An np.ndarray as the intrinsic score. (Dimension: n)
        v_initial (np.ndarray): An np.ndarray as the coefficient of covariates.
        E (float): The accuracy of the iterations in alternating algorithm.
        Eu (float): The accuracy of the iterations in updating u.
        Ev (float): The accuracy of the iterations in updating v.
        I (int): The maxinum iteration times
        detail (bool): Print the iteration details?
        PL (bool): Fit PL?
        save_likelihood (bool): Save log-likelihood?

    Return:
        u,v (np.ndarray,np.ndarray): The optimizer.
    '''
    
    W = multi_Win(hyperedges_list,n)
    # Is PL?
    if d == 0 or PL:
        PL = True
    else:
        PL = False
    # Initials
    if v_initial is None or PL:
        v = np.zeros(d)
    else:
        v = v_initial
    if u_initial is None:
        u = np.zeros(n)
    else:
        u = u_initial
    # Calculate log-likelihood
    l1 = multi_likelihood(hyperedges_list,covariates_list,u,v)
    L = [l1]
    i, error = 1, 1
    if PL:
        # PL
        u = multi_fixv(hyperedges_list, covariates_list, v, n, W,E=Eu,I=1000, u_initial = u,detail = detail)
    else:
        # PlusDC
        while error > E and i < 100:
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {L[-1]}')
            else:
                pass
            v1 = multi_fixu(hyperedges_list, covariates_list, u, d, v_initial=v.copy(),E=Ev,I=I,detail=detail)
            u1 = multi_fixv(hyperedges_list, covariates_list, v1, n, W, u_initial = u,I=I,E=Eu,detail=detail)
            u = u1
            v = v1
            l2 = multi_likelihood(hyperedges_list, covariates_list, u, v)
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

def pair_likelihood(
        hyperedges_list: List[List[int]],
        covariates_list: List[np.ndarray],
        u: np.ndarray,
        v: np.ndarray = None
        ):
    """ Compute the log-likelihood in pairwise case:

    Args:
        hyperedges_set (List[List[int]]): A list as a set containing each comparison ranking. (Dimension: N(n))
        covariates_set (List[np.ndarray]): A list as all the comparison rankings. (Dimension: N(n))
        u (np.ndarray): An np.ndarray as the intrinsic score. (Dimension: n)
        v (np.ndarray): An np.ndarray as the coefficient of covariates.

    Returns:
        L (float): The corresponding log-likelihood.
    
    """
    if v is None:
        d = len(covariates_list[0].T)
        v = np.zeros(d)
    else:
        pass
    
    K = np.array([x[0,:]-x[1,:] for x in covariates_list])
    T = np.array(hyperedges_list)
    result = 0
    different_score = u[T][:,0]-u[T][:,1]+K@v
    p = sig(different_score)
    result = np.mean(np.log(p))
    return result

def pair_alternative(
        hyperedges_list: List[List[int]],
        covariates_list: List[np.ndarray],
        n: int,
        d: int, 
        u_initial: np.ndarray = None ,
        v_initial: np.ndarray = None,
        E: float = 1e-6,
        Eu: float = 1e-4,
        Ev: float = 1e-8,
        I: int = 50,
        detail: bool = False,
        PL: bool = False,
        save_likelihood: bool = False):
    '''Alternating maximization in pairwise case:

    Args:
        T (list of len(N)): The list of edges (list of len(m_i)).
        X (list of len(n)): The list of covariates (d dimensional np.array)
        n (int): The number of items.
        d (int): The dimension of covariates.
        u_initial (np.array of shape(n)): The initial value of u.
        v_initial (np.array of shape(v)): The initial value of v.
        E (float): The accuracy of the iterations in alternating algorithm.
        Eu (float): The accuracy of the iterations in updating u.
        Ev (float): The accuracy of the iterations in updating v.
        I (int): The maxinum iteration times
        detail (bool): Print the iteration details?
        P (bool): Fit PL?
        save_likelihood (bool): Save log-likelihood?

    Return:
        u,v (np.array of shape(n),np.array of shape(d)): The optimizer.
    '''
    K = np.array([x[0,:]-x[1,:] for x in covariates_list])
    T = np.array(hyperedges_list)
    win,lose,win_count = get_win(T,n)
    # Is PL?
    if d == 0 or PL:
        PL = True
    else:
        PL = False
    # Initials
    if v_initial is None or PL:
        v = np.zeros(d)
    else:
        v = v_initial
    if u_initial is None:
        u = np.zeros(n)
    else:
        u = u_initial
    l1 = pair_likelihood(hyperedges_list,covariates_list,u,v)
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
            l2 = pair_likelihood(hyperedges_list,covariates_list, u, v)
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




"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""

"""Some intermediate functions. """

"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""




sig = lambda x: np.exp(x)/(1+np.exp(x))
 
### Multiple
def multi_Win(T,n):
    """ Compute the winning times of each items.

    Args:
        T (list of len(N)): The list of edges (list of items).
        n (int): The number of items.

    Return:
        W (np.array of shape(n)): The winning times of each items.
    """
    W = np.zeros((n))
    for i, t in enumerate(T):
        W[t[:-1]] += 1
    return W

def multi_DynamicScore_Win(X,v):
    """ An intermediate step in the next function.

    Args:
        X (list of len(N)): The list of edges (list of items).
        v (np.array of shape(d)): The coefficient of covariates.

    Return:
        D_r (list of len(N)): Each element (np.array of shape(m_i)) 
    """
    D_r = []
    for k in X:
        tem = np.exp(k@v)
        tem = tem/sum(tem)
        D_r.append(tem)
    return D_r

def multi_update_R(R,W,D,T,n):
    '''The algorithm to optimize u:

    Args:
        R (np.array of shape(n)): Exponential of the intrinsic score.
        W (list of len(N)): (calculated by 'multiple_Win')
        D (list of len(N)): Each element (np.array of shape(m_i)) 
        T (list of len(N)): The list of edges (list of len(m_i)).
        n (int): The number of items.

    Return:
        R_new (np.array of shape(n)): The exponential of updated intrinsic score.
    '''
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

def multi_fixv(T,X,v,n,W, E = 1e-6 , I = 50, u_initial = None, detail = False):
    '''The algorithm to optimize u:

    Args:
        T (list of len(N)): The list of edges (list of len(m_i)).
        X (list of len(n)): The list of covariates (d dimensional np.array)
        v (np.array of shape(d)): The coefficient of covariates.
        n (int): The number of items.
        W (list of len(N)): (calculated by 'multiple_Win')
        E (float): The accuracy of the iterations
        I (int): The maxinum iteration times
        u_initial (np.array of shape(n)): The initial value of u
        detail (bool): Print the iteration details.

    Return:
        u (np.array of shape(n)): The updated intrinsic score.
    '''

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

def multi_update_v(T,X,u,d,v):
    '''Each step of optimizing v:

    Args:
        T (list of len(N)): The list of edges (list of len(m_i)).
        X (list of len(n)): The list of covariates (d dimensional np.array)
        u (np.array of shape(n)): The intrinsic score.
        d (int): The dimension of covariates.
        v (np.array of shape(d)): The coefficient of covariates.

    Return:
        update (np.array of shape(d)): The increment of v.
    '''
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

def multi_fixu(T,X,u,d, E=1e-6, I = 1000,v_initial = None,detail = False):
    '''The algorithm to optimize v:

    Args:
        T (list of len(N)): The list of edges (list of len(m_i)).
        X (list of len(n)): The list of covariates (d dimensional np.array)
        u (np.array of shape(n)): The intrinsic score.
        d (int): The dimension of covariates.
        E (float): The accuracy of the iterations
        I (int): The maxinum iteration times
        v_initial (np.array of shape(n)): The initial value of oefficient v.
        detail (bool): Print the iteration details.

    Return:
        v (np.array of shape(d)): The updated covariates' coefficient.
    '''
    if v_initial is None:
        v = np.zeros(d)
    else:
        v = v_initial[:]
        pass
    i, error = 1, 1
    while error > E*1000 and i < I:
        v_update = multi_update_v(T,X,u,d,v)
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

### Pairwise
def get_win(T,n):
    """ Compute the winning times of each items.

    Args:
        T (list of len(N)): The list of edges (list of items).
        n (int): The number of items.

    Return:
        win (dict of len(n)): The winning competitions of each player.
        lose (dict of len(n)): The losing competitions of each player.
        win_count (dict of len(n)): The winning times of each player.
    """
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

def pair_update_R(R,T,K,v,win,lose,win_count,n):
    '''The algorithm to optimize u:

    Args:
        R (np.array of shape(n)): Exponential of the intrinsic score.
        T (list of len(N)): The list of edges (list of len(m_i)).
        K (np.array of shape(N,d)): The difference of covariates.
        v (np.array of shape(d)): The coefficient of covariates.
        win (dict of len(n)): The winning competitions of each player.
        lose (dict of len(n)): The losing competitions of each player.
        win_count (dict of len(n)): The winning times of each player.
        n (int): The number of items.


    Return:
        R (np.array of shape(n)): The exponential of updated intrinsic score.
    '''
    R1 = R[T[:,0]]
    R2 = R[T[:,1]]
    c = np.exp(-K@v)
    res = np.array([np.sum(1/(R2[win[i]]*c[win[i]]+R[i])) + \
                        np.sum(1/(R[i]+R1[lose[i]]/c[lose[i]])) for i in range(n)])
    #u_est = np.log(1/res*win_count)
    R = 1/res*win_count
    return R

def pair_fixv(T,K,v,n,win,lose,win_count, E = 1e-6 , I = 1000, u_initial = None,detail = False):
    '''The algorithm to optimize u:

    Args:
        T (list of len(N)): The list of edges (list of len(m_i)).
        K (np.array of shape(N,d)): The difference of covariates.
        v (np.array of shape(d)): The coefficient of covariates.
        n (int): The number of items.
        win (dict of len(n)): The winning competitions of each player.
        lose (dict of len(n)): The losing competitions of each player.
        win_count (dict of len(n)): The winning times of each player.
        E (float): The accuracy of the iterations
        I (int): The maxinum iteration times
        u_initial (np.array of shape(n)): The initial value of u
        detail (bool): Print the iteration details.

    Return:
        u (np.array of shape(n)): The updated intrinsic score.
    '''
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

def pair_update_v(T,K,u,v):
    '''The algorithm to optimize u:

    Args:
        T (list of len(N)): The list of edges (list of len(m_i)).
        K (np.array of shape(N,d)): The difference of covariates.
        u (np.array of shape(n)): The intrinsic score.
        v (np.array of shape(d)): The coefficient of covariates.

    Return:
        R (np.array of shape(n)): The exponential of updated intrinsic score.
    '''
    tem = u[T]
    s = tem[:,0]-tem[:,1] + K@v
    l1 = 1 - sig(s)
    l2 = K.T@ ((l1*(1-l1))[:, np.newaxis] * K)
    update = np.linalg.inv(l2)@K.T@l1
    return update

def pair_fixu(T,K,u,d, E=1e-6, I = 1000,v_initial = None,detail = False):
    '''The algorithm to optimize v:

    Args:
        T (list of len(N)): The list of edges (list of len(m_i)).
        X (list of len(n)): The list of covariates (d dimensional np.array)
        u (np.array of shape(n)): The intrinsic score.
        d (int): The dimension of covariates.
        E (float): The accuracy of the iterations
        I (int): The maxinum iteration times
        v_initial (np.array of shape(n)): The initial value of oefficient v.
        detail (bool): Print the iteration details.

    Return:
        v (np.array of shape(d)): The updated covariates' coefficient.
    '''
    if v_initial is None:
        v = np.zeros(d)
    else:
        v = v_initial[:]
        pass
    i, error = 1, 1
    while error > E and i < I:
        v_update = pair_update_v(T, K, u, v)
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
