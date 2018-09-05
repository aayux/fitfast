from .imports import *
from.encoders import *
from .metrics import *
from .logging import *
from .text import *
from .lm import *
from .classifiers.linear import Linear
from .utils.core import *

class LearningParameters(object):
    r"""
    n_cycle (int): Number of cycles.
    cycle_len (int): Number of cycles before lr is reset to the initial
            value, eg. if cycle_len = 3, then the lr is varied between a 
            maximum and minimum value over 3 epochs.
    cycle_mult (int): Additional parameter for influencing how the lr 
            resets over the cycles. For an intuitive explanation.
    use_wd_sched (bool, optional): set to True to enable weight 
            regularization using the technique mentioned in 
            arxiv.org/abs/1711.05101. When this is True by itself the 
            regularization is detached from gradient update and applied 
            directly to the weights.
    norm_wds (bool, optional): when this is set to True along with 
            use_wd_sched, the regularization factor is normalized with 
            each training cycle.
    wds_sched_mult (function, optional): when this is provided along 
            with use_wd_sched as True, the value computed by this 
            function is multiplied with the regularization strength. 
            This function is passed the WeightDecaySchedule object. And 
            example function that can be passed is:
            f = lambda x: np.array(x.layer_opt.lrs) / x.init_lrs
    use_swa (bool, optional): when this is set to True, it will enable 
            the use of Stochastic Weight Averaging 
            arxiv.org/abs/1803.05407. The learner will include an 
            additional model (in the swa_model attribute) for keeping 
            track of the average weights as described in the paper. All 
            testing of this technique so far has been in image 
            classification, so use in other contexts is not guaranteed 
            to work.
    swa_start (int, optional): if use_swa is set to True, then this 
            determines the epoch to start keeping track of the average 
            weights. It is 1-indexed per the paper's conventions.
    swa_eval_freq (int, optional): if use_swa is set to True, this 
            determines the frequency at which to evaluate the 
            performance of the swa_model. This evaluation can be costly 
            for models using BatchNorm (requiring a full pass through 
            the data), which is why the default is not to evaluate after
            each epoch.
    """
    def __init__(self, fintune=True, discriminative=True):
        lr = self.lr = 1.2e-2
        self.lrm = 2.6
        self.lrs = lr
        if discriminative:
            self.lrs = [lr / 6, lr / 3, lr, lr / 2] if finetune \
            else [lr / (lrm ** 4), lr / (lrm ** 3), lr/(lrm ** 2), lr / lrm, lr]
        self.wds = 3e-6
        self.n_cycles = 2
        self.clip = None

        self.cycle_len = 1 
        self.cycle_mult = 1 

        self.use_clr = None
        self.use_alt_clr = None

        self.use_wd_sched = False
        self.norm_wds = False
        self.wds_sched_mult = None 

        self.use_swa = False
        self.swa_start = 1
        self.swa_eval_freq = 5


class LanguageModelLoader(object):
    r"""
    This class provides the entry point for dealing with supported NLP tasks.
    
    Usage:
    - Use one of the factory constructors to obtain an instance of the class.
    - Use the get_model method to return a RNNLearner instance (a network suited
        for NLP tasks), then proceed with training.
    """
    def __init__(self, train_ids, val_ids, lang, bs, bptt, n_tokens=30000, 
                 pad_token=1, em=300, nh=1000, nl=3, **kwargs):
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
        self.train = LanguageModelIterator(train_ids, bs, bptt)
        self.val = LanguageModelIterator(val_ids, bs, bptt)
        self.pad_token = pad_token
        self.n_tokens = n_tokens
        self.lang = lang
        self.em = em
        self.nh = nh
        self.nl = nl

    def get_model(self, lm, itos=None, finetune=True, **kwargs):
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
        m = lm.get_language_model(self.n_tokens, self.em, self.nh, self.nl, 
                                  self.pad_token, **kwargs)
        model = LanguageModel(to_gpu(m))
        learner = RNNLearner(self, model)
        if finetune:
            weights = self.load_pretrained_weights(learner, itos)
            learner.model.load_state_dict(weights)
        return learner

    def load_pretrained_weights(self, l, itos):
        data_dir = Path('./data/wiki')
        pre_dir = data_dir / self.lang
        
        weights = torch.load(pre_dir / 'models' / 'wikitext.h5', 
                       map_location=lambda storage, loc: storage)
        ew = to_np(weights['0.encoder.weight'])
        row_m = ew.mean(0)

        itos_ = pickle.load(open(pre_dir / 'tmp' / f'itos.pkl', 'rb'))
        stoi_ = collections.defaultdict(lambda: -1, 
                                        {v: k for k, v in enumerate(itos_)})
        nw = np.zeros((self.n_tokens, self.em), dtype=np.float32)
        nb = np.zeros((self.n_tokens,), dtype=np.float32)
        
        for i, w in enumerate(itos):
            r = stoi_[w]
            if r >= 0:
                nw[i] = ew[r]
            else:
                nw[i] = row_m

        weights['0.encoder.weight'] = T(nw)
        weights['0.encoder_with_dropout.embed.weight'] = T(np.copy(nw))
        weights['1.decoder.weight'] = T(np.copy(nw))
        return weights

class ClassifierLoader(object):
    def __init__(self, work_dir, bs, bptt, dims=None, sampler=None, max_seq=1000, 
                 n_tokens=30000, pad_token=1, em=300, nh=1000, nl=3, **kwargs):
        self.pad_token = pad_token
        self.n_tokens = n_tokens
        self.dims = dims
        self.em = em
        self.nh = nh
        self.nl = nl

        x_train, x_val, y_train, y_val = self._get_data(work_dir)
        n_classes = len(np.unique(y_train))

        if dims is None:
            self.dims = (em * 3, 50, n_classes)

        if sampler is None:
            train_samp = SortishSampler(x_train, key=lambda x: len(x_train[x]), 
                                        bs=bs // 2)
        else:
            train_samp = _get_sampler(sampler, x_train y_train, bs // 2)
        val_samp = SortedSampler(x_val, key=lambda x: len(x_val[x]))

        train = TextDataset(x_train, y_train)
        val = TextDataset(x_val, y_val)
        del x_train, x_val, y_train, y_val
        
        train = DataLoader(train, bs // 2, transpose=True, num_workers=1, 
                           pad_idx=1, sampler=train_samp)
        val = DataLoader(val, bs, transpose=True, num_workers=1, pad_idx=1, 
                         sampler=val_samp)
        self.md = ModelData(work_dir, train, val)
    
    def get_clf(self, lm, clf, **kwargs):
        # select the encoder module
        # TO DO: Remove hardcode, setup
        # some kind of autodetect.
        if isinstance(lm, AWDLSTMModeler): en = AWDLSTMEncoder
        elif isinstance(lm, QRNNModeler): en = QRNNEncoder
        # elif isinstance(lm, TDModeler): en = TDEncoder
        
        m = clf.get_classifier(en, self.bptt, self.dims, self.max_seq, 
                               self.n_tokens, self.pad_token, self.em, self.nh, 
                               self.nl, **kwargs)
        model = TextModel(to_gpu(m))
        learner = RNNLearner(self.md, model)
        return learner

    def _get_data(self, work_dir):
        tmp_dir = Path(work_dir) / 'tmp'
        x_train = np.load(tmp_dir / 'train_ids.npy')
        x_val = np.load(tmp_dir / 'val_ids.npy')
        y_train = np.load(tmp_dir / 'lbl_train.npy')
        y_val = np.load(tmp_dir / 'lbl_val.npy')
        return x_train, x_val, y_train, y_val

def _get_sampler(sampler, x, y, bs):
    if sampler == 'random':
        return RandomSampler(x)
    elif sampler == 'sortish':
        return SortishSampler(x, key=lambda x_: len(x[x_]), bs=bs)
    elif sampler == 'weighted':
        ratio = np.unique(y, return_counts=True)[1]
        weight = np.array([1 / count[0], 1 / count[1]])
        weight = weight[y]
        return WeightedRandomSampler(weight, len(weight))
    else: raise ValueError(f'Unknown sampler {sampler}.')

