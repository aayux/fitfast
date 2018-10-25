from .imports import *
from .utils.core import *
from .base import *
from .learner import *

class SequentialRNN(nn.Sequential):
    r"""
    The SequentialRNN layer is the native torch's Sequential wrapper that puts 
    the Encoder and Decoder modules sequentially in the model.
    """
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

class LanguageModelIterator(object):
    r""" 
    Returns a language model iterator that iterates through batches that are
    of length N(bptt, 5) The first batch returned is always bptt + 25, the max 
    possible width. This is done because of they way that pytorch allocates 
    CUDA memory in order to prevent multiple buffers from being created as the 
    batch width grows.
    """
    def __init__(self, data, bs, bptt, backwards=False):
        self.bs = bs
        self.bptt = bptt
        self.backwards = backwards
        self.data = self.batchify(data)
        self.seq_iter = 0
        self.iter = 0
        self.n = len(self.data)

    def __iter__(self):
        self.seq_iter = 0
        self.iter = 0
        while self.seq_iter < self.n - 1 and self.iter < len(self):
            if self.seq_iter == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            batch = self.get_batch(self.seq_iter, seq_len)
            self.seq_iter += seq_len
            self.iter += 1
            yield batch

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


class LanguageModel(BaseModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [*zip(m.rnns, m.drop_hs), (self.model[1], m.drop_i)]
        

class RNNLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return F.cross_entropy
    
    def fit(self, *args, **kwargs): 
        return super().fit(*args, **kwargs, seq_first=True)

    def get_model_path(self, name): 
        return os.path.join(self.models_path, name) + '.h5'
    
    def save_encoder(self, name): 
        self.models_path = os.path.join(self.work_dir, MODELS_DIR)
        os.makedirs(self.models_path, exist_ok=True)
        save_model(self.model[0], self.get_model_path(name))
    
    def load_encoder(self, name):
        self.models_path = os.path.join(self.work_dir, MODELS_DIR)
        load_model(self.model[0], self.get_model_path(f'{name}_encoder'))

    def thaw(self, name, gradual, chain_thaw=False, thaw_all=True, last=False, 
             clr=None, alt_clr=None):
        self.load_encoder(name)
        
        if gradual:
            # gradual unfreezing as specified in arxiv.org/abs/1801.06146
            self.freeze_to(-1)
            self.fit(n_cycles=1, cycle_len=1, clr=clr, alt_clr=alt_clr, 
                     callbacks=[])
            self.freeze_to(-2)
            self.fit(n_cycles=1, cycle_len=1, clr=clr, alt_clr=alt_clr, 
                     callbacks=[])

        if chain_thaw:
            lrs = [0.0001 for _ in range(5)]
            nl = len(self.get_layer_groups())
            
            # fine-tune last layer
            self.freeze_to(-1)
            self.fit(n_cycles=1, cycle_len=1, clr=clr, alt_clr=alt_clr, 
                     callbacks=[])
            
            # fine-tune all layers up to the second-last one
            n = 0
            while n < nl - 1:
                self.freeze_all_but(self, n)
                self.fit(n_cycles=1, cycle_len=1, clr=clr, 
                         alt_clr=alt_clr, callbacks=[])
                n += 1

        if thaw_all:
            self.unfreeze()
        else:
            self.freeze_to(-3)

        if last:
            self.freeze_to(-1)


class TextModel(BaseModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [(m.encoder, m.drop_i), *zip(m.rnns, m.drop_hs), 
                (self.model[1])]
