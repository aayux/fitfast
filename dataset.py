import csv

from .imports import *
from .torch_imports import *
from .core import *
from .layer_optimizer import *
from .dataloader import DataLoader

class BaseDataset(Dataset):
    r"""
    An abstract class representing a dataset. Extends torch.utils.data.Dataset.
    """
    def __init__(self, transform=None):
        self.transform = transform
        self.n = self.get_n()
        self.c = self.get_c()
        self.sz = self.get_sz()

    def _get(self, transform, x, y):
        return (x, y) if transform is None else transform(x, y)

    def _getitem(self, idx):
        x = self.get_x(idx)
        y = self.get_y(idx)
        return self._get(self.transform, x, y)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            xs, ys = zip(*[self._getitem(i) for i in range(*idx.indices(self.n))])
            return np.stack(xs), ys
        return self._getitem(idx)

    def __len__(self): return self.n

    @abstractmethod
    def get_n(self):
        r"""Return number of elements in the dataset, ie., len(self).
        """
        raise NotImplementedError

    @abstractmethod
    def get_c(self):
        r"""Return number of classes in a dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sz(self):
        r"""Return maximum size of a sample in a dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def get_x(self, i):
        r"""Return sample at index i.
        """
        raise NotImplementedError

    @abstractmethod
    def get_y(self, i):
        r"""Return label at index i.
        """
        raise NotImplementedError

    @property
    def is_multi(self):
        r"""
        Returns True if this data set contains multiple labels per sample.
        """
        return False

    @property
    def is_reg(self):
        r"""Returns True if training a regression model.
        """
        return False

class ArraysDataset(BaseDataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        assert(len(x) == len(y))
        super().__init__(transform)
    
    def get_x(self, i): return self.x[i]
    
    def get_y(self, i): return self.y[i]
    
    def get_n(self): return len(self.y)
    
    def get_sz(self): return self.x.shape[1]


class ArraysIndexDataset(ArraysDataset):
    def get_c(self): return int(self.y.max()) + 1
    
    def get_y(self, i): return self.y[i]


class ArraysIndexRegressionDataset(ArraysIndexDataset):
    def is_reg(self): return True
    
    
class ArraysNhotDataset(ArraysDataset):
    def get_c(self): return self.y.shape[1]
    
    @property
    def is_multi(self): return True

class TextDataset(Dataset):
    r""" Primary Dataset class for text data. Extends torch.utils.data.Dataset.
    """
    def __init__(self, x, y, backwards=False, sos=None, eos=None):
        self.x = x
        self.y = y
        self.backwards = backwards
        self.sos = sos
        self.eos = eos

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.backwards: x = list(reversed(x))
        if self.eos is not None: x = x + [self.eos]
        if self.sos is not None: x = [self.sos]+x
        return np.array(x), self.y[idx]

    def __len__(self): return len(self.x)

class ModelData(object):
    r"""
    Encapsulates DataLoaders and Datasets for training, validation, test. 
    (Base class for and from fastai *Data classes)
    """
    def __init__(self, path, trn_dl, val_dl, test_dl=None):
        self.path = path
        self.trn_dl = trn_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

    @classmethod
    def from_dls(cls, path, trn_dl, val_dl, test_dl=None): 
        return cls(path, trn_dl, val_dl, test_dl)

    @property
    def is_reg(self): return self.trn_ds.is_reg
    
    @property
    def is_multi(self): return self.trn_ds.is_multi
    
    @property
    def trn_ds(self): return self.trn_dl.dataset
    
    @property
    def val_ds(self): return self.val_dl.dataset
    
    @property
    def test_ds(self): return self.test_dl.dataset
    
    @property
    def trn_y(self): return self.trn_ds.y
    
    @property
    def val_y(self): return self.val_ds.y