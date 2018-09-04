from .imports import *
from .metrics import *
from .logging import *
from .text import *
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
    def __init__(self, discriminative=False):
        lr = self.lr = 1.2e-2
        self.lrs = [lr / 6, lr / 3, lr, lr / 2] if discriminative else lr
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

class Loader(LanguageModelLoader):
    def __init__(self, train_ids, val_ids, lang, bs, bptt, n_tokens=3000, 
                 pad_token=1, em=300, nh=1000, nl=3, **kwargs):
        self.train = LanguageModelIterator(train_ids, bs, bptt)
        self.val = LanguageModelIterator(val_ids, bs, bptt)
        self.pad_token = pad_token
        self.n_tokens = n_tokens
        self.lang = lang
        self.em = em
        self.nh = nh
        self.nl = nl

    def get_model(self, lm, itos=None, finetune=True, **kwargs):
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