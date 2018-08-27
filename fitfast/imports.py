# general utilities
import os, numpy as np, math, collections, threading, json, bcolz, random, scipy
import pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time, \
                 shutil, copy
import seaborn as sns, matplotlib
import graphviz, sklearn_pandas, sklearn, warnings, pdb
import contextlib
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import chain
from functools import partial
from collections import Iterable, Counter, OrderedDict
from pandas_summary import DataFrameSummary
from sklearn import metrics, ensemble, preprocessing
from operator import itemgetter, attrgetter
from pathlib import Path
from distutils.version import LooseVersion
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

np.set_printoptions(precision=5, linewidth=110, suppress=True)

# pytorch imports
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

# TO DO: - Remove unused imports
#        - Make everything Torch 0.4.1