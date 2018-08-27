from ..imports import *

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')
USE_GPU = torch.cuda.is_available()

def no_op(*args, **kwargs): return

def map_over(x, f):
    return [f(o) for o in x] if isinstance(x, (list, tuple)) else f(x)

def map_none(x, f): return None if x is None else f(x)

def A(*a):
    r""" Convert iterable object into numpy array
    """
    return np.array(a[0]) if len(a) == 1 else [np.array(o) for o in a]

def T(a, half=False, cuda=True):
    r"""
    Convert numpy array into a pytorch tensor. If CUDA is available and 
    USE_GPU=True, store resulting tensor in GPU.
    """
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = torch.cuda.HalfTensor(a) if half else torch.FloatTensor(a)
        else: raise NotImplementedError(a.dtype)
    if cuda: a = to_gpu(a, async=True)
    return a

def V_(x, requires_grad=False, volatile=False):
    r"""equivalent to pytorch_variable, which creates a pytorch tensor.
    """
    return pytorch_variable(x, volatile=volatile, requires_grad=requires_grad)

def V(x, requires_grad=False, volatile=False):
    r"""creates a single or a list of pytorch tensors, depending on input x.
    """
    return map_over(x, lambda o: V_(o, requires_grad, volatile))

def VV_(x): 
    r"""creates a volatile tensor, which does not require gradients.
    """
    return pytorch_variable(x, volatile=True)

def VV(x):
    r"""creates a single or a list of pytorch tensors, depending on input x.
    """
    return map_over(x, VV_)

def pytorch_variable(x, volatile, requires_grad=False):
    if type (x) != Variable:
        if IS_TORCH_04: x = Variable(T(x), requires_grad=requires_grad)
        else: x = Variable(T(x), requires_grad=requires_grad, volatile=volatile)
    return x

def to_np(v):
    r"""
    returns an np.array object given an input of np.array, list, tuple, 
    torch variable or tensor.
    """
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v = v.data
    if isinstance(v, torch.cuda.HalfTensor): v = v.float()
    return v.cpu().numpy()

def to_gpu(x, *args, **kwargs):
    r""" Puts Pytorch variable to gpu, if CUDA is available and USE_GPU=True.
    """
    return x.cuda(*args, **kwargs) if USE_GPU else x

def _trainable_params(m):
    r"""
    Returns a list of trainable parameters in the model, i.e., those that 
    require gradients.
    """
    return [p for p in m.parameters() if p.requires_grad]

def chain_params(p):
    r"""
    Chains the list of trainable parameters in the model returned by 
    _trainable_params.
    """
    if isinstance(p, (list, tuple)):
        return list(chain(*[_trainable_params(o) for o in p]))
    return _trainable_params(p)

def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b

def apply_leaf(m, f):
    r""" Apply function repeatedly to each child/leaf node.
    """
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    if len(c) > 0:
        for l in c: apply_leaf(l, f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))

def sgd_with_momentum(momentum=0.9):
    r""" The default optimizer.
    """
    return lambda *args, **kwargs: optim.SGD(*args, momentum=momentum, **kwargs)

def one_hot(a, c): return np.eye(c)[a]

def partition(a, sz): 
    r""" Splits iterables a in equal parts of given size.
    """
    return [a[i : i + sz] for i in range(0, len(a), sz)]

def partition_by_cores(a):
    return partition(a, len(a) // num_cpus() + 1)

def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()

def set_grad_enabled(mode): 
    return torch.set_grad_enabled(mode) if IS_TORCH_04 else contextlib.suppress()

def no_grad_context(): 
    return torch.no_grad() if IS_TORCH_04 else contextlib.suppress()

def children(m): 
    return m if isinstance(m, (list, tuple)) else list(m.children())

def save_model(m, p): torch.save(m.state_dict(), p)

def load_model(m, p):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    
    # list "detatches" the iterator
    for n in list(sd.keys()):
        if n not in names and f'{n}_raw' in names:
            if f'{n}_raw' not in sd: sd[f'{n}_raw'] = sd[n]
            del sd[n]
    m.load_state_dict(sd)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return F.log_softmax(l_x, dim=-1)