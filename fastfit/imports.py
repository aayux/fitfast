import PIL, os, numpy as np, math, collections, threading, json, bcolz, \
                         random, scipy, cv2
import pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time, \
                 shutil, copy
import seaborn as sns, matplotlib
import IPython, graphviz, sklearn_pandas, sklearn, warnings, pdb
import contextlib
from abc import abstractmethod
from glob import glob, iglob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import chain
from functools import partial
from collections import Iterable, Counter, OrderedDict
from isoweek import Week
from pandas_summary import DataFrameSummary
from sklearn import metrics, ensemble, preprocessing
from operator import itemgetter, attrgetter
from pathlib import Path
from distutils.version import LooseVersion

from matplotlib import pyplot as plt, rcParams, animation
matplotlib.rc('animation', html='html5')
np.set_printoptions(precision=5, linewidth=110, suppress=True)

import tqdm as tq
from tqdm import tqdm_notebook, tnrange

from tqdm import tqdm, trange
tnrange = trange
tqdm_notebook = tqdm

# TO DO: 
# Replace with Tensorboard
# Remove unused imports


