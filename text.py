from .core import *
from .learner import *
from .models.ulmfit import *

class LanguageModelLoader():
    r""" 
    Returns a language model iterator that iterates through batches that are
    of length N(bptt, 5) The first batch returned is always bptt + 25; the max 
    possible width.  This is done because of they way that pytorch allocates 
    cuda memory in order to prevent multiple buffers from being created as the 
    batch width grows.
    """
    def __init__(self, nums, bs, bptt, backwards=False):
        self.bs = bs
        self.bptt = bptt
        self.backwards = backwards
        self.data = self.batchify(nums)
        self.i, self.iter = 0, 0
        self.n = len(self.data)

    def __iter__(self):
        self.i, self.iter = 0, 0
        while self.i < self.n - 1 and self.iter < len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self): return self.n // self.bptt - 1

    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[: nb * self.bs])
        data = data.reshape(self.bs, -1).T
        if self.backwards: data = data[::-1]
        return T(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i : i + seq_len], source[i + 1 : i + 1 + seq_len].view(-1)


class LanguageModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [*zip(m.rnns, m.dropouths), (self.model[1], m.dropouti)]


class LanguageModelData():
    def __init__(self, path, pad_idx, n_tok, trn_dl, val_dl, test_dl=None, 
                 **kwargs):
        self.path = path
        self.pad_idx = pad_idx
        self.n_tok = n_tok
        self.trn_dl = trn_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

    def get_model(self, opt_fn, emb_sz, n_hid, n_layers, **kwargs):
        m = rnn_language_model(self.n_tok, emb_sz, n_hid, n_layers, self.pad_idx,
                               **kwargs)
        model = LanguageModel(to_gpu(m))
        return RNNLearner(self, model, opt_fn=opt_fn)


class RNNLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return F.cross_entropy
    
    def fit(self, *args, **kwargs): 
        return super().fit(*args, **kwargs, seq_first=True)

    def save_encoder(self, name): 
        save_model(self.model[0], self.get_model_path(name))
    
    def load_encoder(self, name): 
        load_model(self.model[0], self.get_model_path(name))


class TextModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [(m.encoder, m.dropouti), *zip(m.rnns, m.dropouths), 
                (self.model[1])]