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


class LanguageModelLoader(object):
    r"""
    This class provides the entry point for dealing with supported NLP tasks.
    
    Usage:
    - Use one of the factory constructors to obtain an instance of the class.
    - Use the get_model method to return a RNNLearner instance (a network suited
        for NLP tasks), then proceed with training.
    """
    def __init__(self, train, val, n_tokens, pad_token, test=None, 
                 **kwargs):
        r""" 
        Constructor for the class. Three instances of the LanguageModel are 
        constructed, one each for training, validation data and the test 
        datasets.
            
            Arguments:
                train (LanguageModelLoader): Training data.
                val (LanguageModelLoader): Validation data.
                n_tokens (int): number of unique vocabulary words (or tokens) 
                        in the source dataset.
                pad_token (int): The int value used for padding text.
                n_token (int): The int value used for padding text.
                test (LanguageModelLoader): Testing dataset.
                kwargs: Other arguments:
        """
        self.train = train
        self.val = val
        self.test = test
        self.pad_token = pad_token
        self.n_tokens = n_tokens

    def get_model(self, lm, optimizer, em, nh, nl, **kwargs):
        r"""
        Method returns a RNNLearner object, that wraps an instance of the 
        RNNEncoder module.
        Arguments:
            optimizer (Optimizer): the torch optimizer function to use
            em (int): embedding size
            nh (int): number of hidden inputs
            nl (int): number of hidden layers
            kwargs: other arguments
        Returns:
            An instance of the RNNLearner class.
        """
        m = lm.get_language_model(self.n_tokens, em, nh, nl, 
                                      self.pad_token, **kwargs)
        model = LanguageModel(to_gpu(m))
        return RNNLearner(self, model, optimizer=optimizer)


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
        return [(m.encoder, m.dropouti), *zip(m.rnns, m.dropouths), 
                (self.model[1])]