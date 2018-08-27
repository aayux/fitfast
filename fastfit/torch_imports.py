import os
from distutils.version import LooseVersion
import torch, torchvision, torchtext
from torch import nn, cuda, backends, FloatTensor, LongTensor, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset
from torch.nn.init import kaiming_uniform, kaiming_normal
from torchvision.transforms import Compose

import warnings
warnings.filterwarnings('ignore', message='Implicit dimension choice', 
                         category=UserWarning)

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')
if IS_TORCH_04:
    from torch.nn.init import kaiming_uniform_ as kaiming_uniform
    from torch.nn.init import kaiming_normal_ as kaiming_normal

def children(m): 
  return m if isinstance(m, (list, tuple)) else list(m.children())

def save_model(m, p): torch.save(m.state_dict(), p)

def load_model(m, p):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    for n in list(sd.keys()): # list "detatches" the iterator
        if n not in names and f'{n}_raw' in names:
            if f'{n}_raw' not in sd: sd[f'{n}_raw'] = sd[n]
            del sd[n]
    m.load_state_dict(sd)

def load_pre(pre, f, fn):
    m = f()
    path = os.path.dirname(__file__)
    if pre: load_model(m, f'{path}/weights/{fn}.pth')
    return m