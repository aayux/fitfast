from .imports import *
from .utils.core import *
from .base import *
from .lm import *
from .learner import *

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
    
    def save_encoder(self, wd, name): 
        self.models_path = os.path.join(wd, MODELS_DIR)
        os.makedirs(self.models_path, exist_ok=True)
        save_model(self.model[0], self.get_model_path(name))
    
    def load_encoder(self, name):
        self.models_path = os.path.join(wd, MODELS_DIR)
        load_model(self.model[0], self.get_model_path(name))


class TextModel(BaseModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [(m.encoder, m.drop_i), *zip(m.rnns, m.drop_hs), 
                (self.model[1])]