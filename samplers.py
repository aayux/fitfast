from .core import *
from .learner import *
from torch.utils.data.sampler import Sampler

class SortSampler(Sampler):
    def __init__(self, source, key): 
    	self.source = data_source
    	self.key = key
    def __len__(self): return len(self.source)
    def __iter__(self):
        return iter(sorted(range(len(self.source)), key=self.key, reverse=True))


# Possibly implement a better random sampler, following:
# forums.fast.ai/t/sortishsampler-pitfalls/17817
class SortishSampler(Sampler):
    r"""
    Returns an iterator that traverses the the data in randomly ordered batches 
    that are approximately the same size. The max key size batch is always 
    returned in the first call because of pytorch cuda memory allocation 
    sequencing. Without that max key returned first multiple buffers may be 
    allocated when the first created isn't large enough to hold the next in the 
    sequence.
    """
    def __init__(self, source, key, bs):
        self.source = source
        self.key = key
        self.bs = bs

    def __len__(self): return len(self.source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.source))
        sz = self.bs * 50
        ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) \
        												for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
        
        # find the chunk with the largest key
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])
        
        # then make sure it goes first
        ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]
        
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)

# class WeightedRandomSampler(Sampler):
#     r"""
#     Samples elements from [0,..,len(weights)-1] with given probabilities/weights.

#     Arguments:
#         weights (sequence): A sequence of weights, not necessary summing up to 
#                 one.
#         num_samples (int): Number of samples to draw.
#         replacement (bool): If True, samples are drawn with replacement. If not,
#                 they are drawn without replacement, which means that when a 
#                 sample index is drawn for a row, it cannot be drawn again for 
#                 that row.
#     """

#     def __init__(self, weights, num_samples, replacement=True):
#         if not isinstance(num_samples, _int_classes) \
#             or isinstance(num_samples, bool) or num_samples <= 0:
#             raise ValueError(f'num_samples should be a positive integeral '
#                              f'value, but got num_samples={num_samples}')
#         if not isinstance(replacement, bool):
#             raise ValueError(f'replacement should be a boolean value, but got '
#                              f'replacement={replacement}')
#         self.weights = torch.tensor(weights, dtype=torch.double)
#         self.num_samples = num_samples
#         self.replacement = replacement

#     def __iter__(self):
#         return iter(torch.multinomial(self.weights, self.num_samples, 
#                                       self.replacement))

#     def __len__(self):
#         return self.num_samples