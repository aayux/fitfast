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

def draw_line(ax,x):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.plot([x, x], [ymin, ymax], color='red', linestyle='dashed')

def draw_text(ax,x, text):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(x, (ymin + ymax) / 2, text, horizontalalignment='center', 
            verticalalignment='center', fontsize=14, alpha=0.5)

def curve_smoothing(vals, beta):
    avg_val = 0
    smoothed = []
    for (i, v) in enumerate(vals):
        avg_val = beta * avg_val + (1 - beta) * v
        smoothed.append(avg_val / (1 - beta ** (i + 1)))
    return smoothed

def chunk_iter(iterable, chunk_size):
    r""" A generator that yields chunks of iterable, chunk_size at a time.
    """
    while True:
        chunk = []
        try:
            for _ in range(chunk_size): chunk.append(next(iterable))
            yield chunk
        except StopIteration:
            if chunk: yield chunk
            break