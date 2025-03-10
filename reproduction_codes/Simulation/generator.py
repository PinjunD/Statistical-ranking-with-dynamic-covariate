import numpy as np
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
normalize = lambda n: 25*n+4*np.log(n)**3
p_in1 =lambda n: 5*n/normalize(n)
p_in2 =lambda n: 20*n/normalize(n)
p_cross =lambda n: 4*np.log(n)**3/normalize(n)
p_SBM = {'number':lambda x: int(x/3),
         'p':lambda x: [p_in1(x),p_in2(x),p_cross(x)]
}

class MultipleComparison:
    def __init__(self, n, N, v, 
                 u_generator = u_uniform,
                 x_center = x_center,
                 x_generator = x_generator, 
                 m_lower = 2, m_upper = 3,Type = 'NURHM',p_SBM = p_SBM):
        self.n = n
        self.N = N(n)
        self.v = np.array(v)
        self.d = len(v)
        self.type = Type
        self.m = np.random.randint(m_lower,m_upper,self.N)
        self.u = u_generator(n)
        self.x_center = x_center(n,self.d)
        self.x_generator = x_generator
        self.T = []
        self.X = []
        self.get_community(p_SBM)
        self.get_edges()
        """print(f'complete get edges: Num_Edges={len(self.T)}, '
              f'Num_Nodes = {n},m_lower = {m_lower},m_upper = {m_upper}')"""
    def get_community(self,p_SBM):
        if self.type == 'HSBM':
            self.n1 = p_SBM['number'](self.n)
            self.p = p_SBM['p'](self.n)
            """self.n1 = int(self.n/3)
            n = self.n
            N = 25*n+4*np.log(n)**3
            p1 = 5*n/N
            p2 = 20*n/N
            p3 = 4*np.log(n)**3/N
            self.p = [p1, p2, p3]#community1 community2 cross"""

        else:
            pass
    
    def get_edges(self):
        for m in self.m:
            edge = self.choose_node(m)
            latent_score = self.u[edge]
            dynamic_score = self.x_generator(self.x_center[edge])
            R = np.exp(latent_score + dynamic_score@self.v)
            o = order(R)
            new_edge = [x for _, x in sorted(zip(o, edge))]
            new_X = np.array([x for _, x in sorted(zip(o, dynamic_score))])
            self.T.append(new_edge)
            self.X.append(new_X)

    def choose_node(self,m):
        if self.type == 'NURHM':
            e = np.random.choice(self.n, size=m, replace=False)
        elif self.type == 'HSBM':
            edge_position = np.random.choice([0, 1, 2], p=self.p)
            if edge_position == 0:
                e = np.random.choice(self.n1, size=m, replace=False)
            elif edge_position == 1:
                e = np.random.choice(range(self.n1,self.n), size=m, replace=False)
            else:
                e0 = np.random.choice(self.n, size=m-2, replace=False).tolist()
                n1 = [i for i in range(self.n1) if i not in e0]
                n2 = [i for i in range(self.n1, self.n) if i not in e0]
                e1 = np.random.choice(n1,1).tolist()
                e2 = np.random.choice(n2,1).tolist()
                e = e0 + e1 + e2
        else:
            e = None
        return e
    def Recomparison(self):
        T_new = []
        X_new = []
        for T,X in zip(self.T,self.X):
            latent_score = self.u[T]
            R = np.exp(latent_score + X@self.v)
            o = order(R)
            new_edge = [x for _, x in sorted(zip(o, T))]
            new_X = np.array([x for _, x in sorted(zip(o, X))])
            T_new.append(new_edge)
            X_new.append(new_X)
        self.T = T_new
        self.X = X_new

    
if __name__ == '__main__':
    n = 200
    N = lambda n: int(0.1*n*(np.log(n))**3)
    v = [1,-0.5,0]
    m_lower = 2
    m_upper = 4
    H = MultipleComparison(n,N,v,m_lower=m_lower,m_upper=m_upper,Type='HSBM')
    print(H.T)