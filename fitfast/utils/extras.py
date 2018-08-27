from ..imports import *

def gp_sum(a, r, n):
    r""" 
    Returns the sum of Geometric Progression. This is required here for 
    calculating the number of epochs given the cycle_mult (r) parameter.
    """
    return a * n if r == 1 else math.ceil(a * (1 - r ** n) / (1 - r))

def delistify(x): return x[0] if isinstance(x, (list, tuple)) else x

def listify(x, y):
    if not isinstance(x, collections.Iterable): x = [x]
    n = y if type(y) == int else len(y)
    if len(x) == 1: x = x * n
    return x

def datafy(x):
    if isinstance(x, (list, tuple)): return [o.data for o in x]
    else: return x.data

def one_hot(a,c): return np.eye(c)[a]

def save(fn, a): 
    r""" Utility function that savess model, function, etc as pickle.
    """
    pickle.dump(a, open(fn, 'wb'))
def load(fn): 
    r""" Utility function that loads a pickled model, function, etc.
    """
    return pickle.load(open(fn, 'rb'))

def load_array(fname): 
    r"""
    Load array using bcolz, which is based on numpy, for fast array saving and 
    loading operations. 
    """
    return bcolz.open(fname)[:]