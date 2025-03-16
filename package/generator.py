import numpy as np
from typing import Callable,Dict,Any,List
# Some initial configurations
def u_uniform(n):
    """Generate the intrinsic score from uniform distribution.
    
    Args:
        n (int): The number of items.
    
    Returns :
        u (np.array): The intrinsic score of items. (u.shape = n)
    """
    u = np.random.uniform(-0.5, 0.5, n)
    u += - np.mean(u)
    return u

def x_center(n,d):    
    """Generate the covariate center from uniform distribution.
    
    Args:
        n (int): The number of items.
        d (int): The dimension of covariates.
    
    Returns :
        x (np.array): The covariate center of items. (x.shape = n,d)
    """
    x = np.random.uniform(-0.5, 0.5, (n,d))
    x += - np.mean(x)
    return x

def x_generator(x):
    """Generate the dynamic covariate from covariate center.

    Args:
        x (np.array): The covariate center of items. 

    Returns :
        variables (np.array): The intrinsic score of items. (variables.shape = x.shape)
    """
    m, d = x.shape
    variables = np.random.normal(0,1,size=(m,d))
    return variables

def HSBM_setting(n):
    """Define the p_SBM configurations in HSBM (function of n)
        Args:
            n (int): The number of items.
            
        Returns:
            p_SBM (dict): 'number' is the number of items in the first community (int)
                        'p' is the list of probability (float) for 'within' and 'cross'.
    """
    normalize = lambda n: 25*n+4*np.log(n)**3
    p_in1 =lambda n: 5*n/normalize(n)
    p_in2 =lambda n: 20*n/normalize(n)
    p_cross =lambda n: 4*np.log(n)**3/normalize(n)
    p_SBM = {'number':lambda x: int(x/3),
            'p':lambda x: [p_in1(x),p_in2(x),p_cross(x)]
    }
    return p_SBM


# Comparison Generator
class MultipleComparison:
    """ Generate a multiple comparison graph
    
    Attributes:
        n (int): An int as the number of items.
        N (Callable[[int], int]): A function for generating hyperedges' number.
        m (List[int]): A list for representing the sizes of hyperedges.
        type (str): A string for selecting either 'NURHM' or 'HSBM'.
        x_center (np.ndarray): An np.ndarray as the center of the covariates. (Dimension: n*d)
        x_generator (Callable[[np.ndarray], np.ndarray]): 
            A function for generating dynamic covariates based on x_center.
        u (np.ndarray): An np.ndarray as the intrinsic score. (Dimension: n)
        v (np.ndarray): An np.ndarray as the coefficient of covariates.
        d (int): An int as the dimension of covariates.
        hyperedges_set (List[List[int]]): A list as a set containing each comparison ranking. (Dimension: N(n))
        covariates_set (List[np.ndarray]): A list as all the comparison rankings. (Dimension: N(n))

    Methods:
        get_community(Dict) -> None
        choose_node(m) -> None
        get_edges() -> None
        get_order(np.ndarray) -> np.ndarray:
    """


    def __init__(
            self, 
            n: int, 
            N: Callable[[int], int], 
            v: np.ndarray,      
            u_generator: Callable[[int], np.ndarray] = u_uniform,
            x_center: Callable[[int], np.ndarray] = x_center,
            x_generator: Callable[[np.ndarray], np.ndarray] = x_generator, 
            m_lower: int = 2, 
            m_upper: int = 3,
            Type: str = 'NURHM',
            p_SBM: Callable[[int],Dict[str,Any]] = HSBM_setting
        ):
        """ Initiate configurations:

            Args:
                n (int): The number of items.
                N (Callable[[int], int]): The number of edges. 
                m_lower (int): The lower bound of edge size.
                m_upper (int): The upper bound of edge size.
                u_generator (Callable[[int], ]): Generate the intrinsic score from uniform distribution.
                x_center (function): Generate the covariate center from uniform distribution.
                x_generator (function): Generate the dynamic covariate from covariate center.
                Type (str): {'NURHM','HSBM'}.
                p_SBM (dict): If Type=='HSBM', you can change this setting.
        """
        self.n = n
        self.N = N(n)
        self.m = np.random.randint(m_lower,m_upper,self.N)
        self.type = Type

        self.u = u_generator(n)
        self.v = np.array(v)
        self.d = len(v)
        self.x_center = x_center(n,self.d)
        self.x_generator = x_generator
        
        
        self.hyperedges_set = []
        self.covariates_set = []
        self.get_community(p_SBM(n))
        self.get_edges()

    def get_community(self, config: Dict[str,Any]) -> None :
        """ Setting community configuration (pass if NURHM).
        """
        if self.type == 'HSBM':
            self.n1 = config['number'](self.n)
            self.p = config['p'](self.n)
        else:
            pass
    def choose_node(self,m:int) -> None :
        """ Randomly select nodes

        Args:
            m (int): The size of edge

        Return:
            e (list): The list of items that are involved in the comparison.
        """
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
    def get_edges(self) -> None :
        """ Generate edges and edge-dependent covariates.
        """
        for m in self.m:
            edge = self.choose_node(m)
            latent_score = self.u[edge]
            dynamic_score = self.x_generator(self.x_center[edge])
            R = np.exp(latent_score + dynamic_score@self.v)
            o = self.get_order(R)
            new_edge = [x for _, x in sorted(zip(o, edge))]
            new_X = np.array([x for _, x in sorted(zip(o, dynamic_score))])
            self.hyperedges_set.append(new_edge)
            self.covariates_set.append(new_X)
    def get_order(self,R:np.ndarray):
        """ Generate the comparison results

        Args:
            R (list): A list of items that are involved in the comparison.
        
        Returns :
            o (list): A list of items rearranged in order of rank.
        """
        m = len(R)
        o = [m-1 for _ in range(m)]
        for i in range(m-1):
            R = R/sum(R)
            win = np.random.multinomial(1,R)
            ind = np.nonzero(win == 1)[0][0]
            o[ind] = i
            R[ind] = 0
        return o

if __name__ == '__main__':
    # Toy example

    n = 200
    N = lambda n: int(0.1*n*(np.log(n))**3)
    v = [1,-0.5,0]
    m_lower = 2
    m_upper = 4
    H = MultipleComparison(n,N,v,m_lower=m_lower,m_upper=m_upper,Type='HSBM')
    print(H.hyperedges_set)