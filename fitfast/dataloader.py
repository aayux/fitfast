import torch
import queue
from torch.utils.data.sampler import SequentialSampler, RandomSampler, \
                                     BatchSampler
from .imports import *
from .utils.core import *
import collections
import sys
import traceback
import threading

STRING_CLASSES = (str, bytes)

def _get_tensor(batch, pin, half=False):
    if isinstance(batch, (np.ndarray, np.generic)):
        batch = T(batch, half=half, cuda=False).contiguous()
        if pin: batch = batch.pin_memory()
        return to_gpu(batch)
    elif isinstance(batch, STRING_CLASSES):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: _get_tensor(sample, pin, half) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [_get_tensor(sample, pin, half) for sample in batch]
    raise TypeError(f'batch must contain numbers, dicts or lists; '
                    f'found {type(batch)}')


class DataLoader(object):
    r"""
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        data (Dataset): dataset from which to load the data.
        bs (int, optional): how many samples per batch to load
                (default: 1).
        shuffle (bool, optional): set to True to have the data reshuffled at 
                every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
                the dataset. If specified, shuffle must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
                indices at a time. Mutually exclusive with bs, shuffle,
                sampler, and drop_last.
        pad_token (int, optional): ...
        num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main 
                process. (default: 0)
        pin_memory (bool, optional): If True, the data loader will copy  tensors
                into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to True to drop the last incomplete 
                batch, if the dataset size is not divisible by the batch size. 
                If False and the size of dataset is not divisible by the batch 
                size, then the last batch will be smaller. (default: False)
        pred_pad (bool, optional): ...
        half (bool, optional): ...
        transpose (bool, optional): ...
        transpose_y (bool,  optional): ...

    """
    def __init__(self, data, bs=1, shuffle=False, sampler=None, 
                 batch_sampler=None, pad_token=0, num_workers=None, 
                 pin_memory=False, drop_last=False, pre_pad=True, half=False,
                 transpose=False, transpose_y=False):
        
        self.data = data
        self.bs = bs
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.pre_pad = pre_pad
        self.transpose = transpose
        self.transpose_y = transpose_y
        self.pad_token = pad_token
        self.half = half

        if batch_sampler is not None:
            if bs > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(f'batch_sampler is mutually exclusive of bs, '
                                 f'shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                sampler = RandomSampler(data) \
                          if shuffle else SequentialSampler(data)
            batch_sampler = BatchSampler(sampler, bs, drop_last)

        if num_workers is None:
            self.num_workers = num_cpus()

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def _jag_stack(self, batches):
        r"""
        ??? DOCUMENTATION REQUIRED ???
        """
        if len(batches[0].shape) not in (1, 2): return np.stack(batches)
        ml = max(len(o) for o in batches)
        if min(len(o) for o in batches) == ml: return np.stack(batches)
        batch = np.zeros((len(batches), ml), 
                          dtype=batches[0].dtype) + self.pad_token
        for i, o in enumerate(batches):
            if self.pre_pad: batch[i, -len(o):] = o
            else: batch[i, :len(o)] = o
        return batch

    def _np_collate(self, batches):
        r"""
        Merges a list of samples to form a mini-batch.
        """
        batch = batches[0]
        if isinstance(batch, (np.ndarray, np.generic)):
            return self._jag_stack(batches)
        elif isinstance(batch, (int, float)): return np.array(batches)
        elif isinstance(batch, STRING_CLASSES): return batches
        elif isinstance(batch, collections.Mapping):
            return {key: self._np_collate([b[key] for b in batches])\
                                                  for key in batch}
        elif isinstance(batch, collections.Sequence):
            return [self._np_collate(samples) for samples in zip(*batches)]
        raise TypeError((f'batch must contain numbers, dicts or lists; '
                         f'found {type(batch)}'))

    def _get_batch(self, idxs):
        bacth = self._np_collate([self.data[idx] for idx in idxs])
        if self.transpose: res[0] = res[0].T
        if self.transpose_y: res[1] = res[1].T
        return batch

    def __iter__(self):
        r"""
        Iterates once over the DataLoader's dataset, as specified by the sampler
        """
        if self.num_workers == 0:
            for batch in map(self._get_batch, iter(self.batch_sampler)):
                yield _get_tensor(batch, self.pin_memory, self.half)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
                # Prevents error in 3.6 wherein queue is infinite 
                # and can result in memory exhaustion
                for c in chunk_iter(iter(self.batch_sampler), 
                                         self.num_workers * 10):
                    for batch in e.map(self._get_batch, c):
                        yield _get_tensor(batch, self.pin_memory, self.half)

    def __len__(self): return len(self.batch_sampler)

