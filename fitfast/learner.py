from .imports import *
from .utils.core import *
from .utils.extras import *
from .stepper import *
from .dataset import *
from .schedules import *
from .layer_optimizer import *
from .logging import summarize, SaveBestModel
from .metrics import *
from .tricks.swa import *
# from .fp16 import *
import time


class Learner(object):
    def __init__(self, data, models, **kwargs):
        r"""
        Combines a LanguageModelLoader object with a nn.Module object, such that
        you can train that module.
        
        Args:
            data_ (LanguageModelLoader): An instance of LanguageModelLoader.
                    This is different from data property of the class.
            models(module): chosen neural architecture for solving a supported 
                    problem.
            optimizer(function): Optimizer function, uses SGD with momentum of 
                                .9 if none is provided.
            metrics(list): Array of functions for evaluating a desired metric, 
                    for eg. accuracy.
            clip(float): Gradient clip chosen to limit the change in the 
                    gradient to prevent exploding gradients.
            crit (function): The loss function used. Defaults to MSE Loss.
        """
        self.data_ = data
        self.models = models
        self.sched = None
        self.wd_sched = None
        self.fp16 = False

    def compile(self, work_dir, lparams, optimizer=None, regularizer=None, 
                metrics=None, crit=None, callbacks=None):
        self.lparams = lparams
        self.optimizer = optimizer or sgd_with_momentum()
        self.regularizer = regularizer
        self.metrics = metrics
        self.crit = crit if crit else self._get_crit(self.data_)
        self.callbacks = callbacks if callbacks else []
        self.work_dir = work_dir

    @classmethod
    def from_model_data(cls, m, data, **kwargs):
        self = cls(data, BaseModel(to_gpu(m)), **kwargs)
        self.unfreeze()
        return self

    def __getitem__(self, idx): return self.children[idx]

    @property
    def children(self): return children(self.model)

    @property
    def model(self): return self.models.model

    @property
    def data(self): return self.data_

    def summary(self): 
        return summarize(self.model, 
                            [torch.rand(3, 3, self.data.sz, self.data.sz)])

    def __repr__(self): return self.model.__repr__()

    def set_bn_freeze(self, m, do_freeze):
        if hasattr(m, 'running_mean'): m.bn_freeze = do_freeze

    def bn_freeze(self, do_freeze):
        apply_leaf(self.model, lambda m: self.set_bn_freeze(m, do_freeze))

    def freeze_to(self, n):
        c = self.get_layer_groups()
        for l in c: set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)

    def freeze_all_but(self, n):
        c = self.get_layer_groups()
        for l in c: set_trainable(l, False)
        set_trainable(c[n], True)
        
    def freeze_groups(self, groups):
        c = self.get_layer_groups()
        self.unfreeze()
        for g in groups:
            set_trainable(c[g], False)
            
    def unfreeze_groups(self, groups):
        c = self.get_layer_groups()
        for g in groups:
            set_trainable(c[g], True)

    def unfreeze(self): self.freeze_to(0)

    def get_model_path(self, name): 
        self.models_path = os.path.join(self.work_dir, MODELS_DIR)
        return os.path.join(self.models_path, name) + '.h5'
    
    def save(self, name):
        self.models_path = os.path.join(self.work_dir, MODELS_DIR)
        os.makedirs(self.models_path, exist_ok=True)
        save_model(self.model, self.get_model_path(name))
        if hasattr(self, 'swa_model'): 
            save_model(self.swa_model, 
                       self.get_model_path(name)[:-3] + '_swa.h5')
                       
    def load(self, path): 
        load_model(self.model, path)

    def set_data(self, data): self.data_ = data

    def half(self):
        if self.fp16: return
        self.fp16 = True
        if type(self.model) != FP16: self.models.model = FP16(self.model)
    
    def float(self):
        if not self.fp16: return
        self.fp16 = False
        if type(self.model) == FP16: self.models.model = self.model.module
        self.model.float()
    
    def _fit(self, model, data, layer_opt, save_best_model=False, n_cycles=None, 
            cycle_len=None, use_clr=None, use_alt_clr=None, callbacks=None, 
            **kwargs):

        r"""
        Method does some preparation before finally delegating to the 'fit' 
        method for fitting the model. Namely, if cycle_len is defined, it adds a
        'Cosine Annealing' scheduler for varying the learning rate across 
        iterations.

        Method also computes the total number of epochs to fit based on provided
        'cycle_len', 'cycle_mult', and 'n_cycles' parameters.

        Arguments:
            model (Learner): Any neural architecture for solving a supported 
                    problem, eg. RNNLearner etc.
            data (ModelData): An instance of ModelData.
            layer_opt (LayerOptimizer): An instance of the LayerOptimizer class.
            save_best_model (bool): Use to save weights of best model during 
                    training.
            metrics (function): Some function for evaluating a desired metric, 
                    eg. accuracy.
            callbacks (list(Callback)): Callbacks to apply during the training.

        Returns: None
        """
        if n_cycles is None:
            n_cycles = self.lparams.n_cycles
        if cycle_len is None:
            cycle_len = self.lparams.cycle_len
        if use_clr is None:
            use_clr = self.lparams.use_clr
        if use_alt_clr is None:
            use_alt_clr = self.lparams.use_alt_clr
        if callbacks is None:
            callbacks = self.callbacks

        cl = cycle_len

        if self.lparams.use_wd_sched:
            # This needs to come before CosineAnnealing() because we need to 
            # read the initial learning rate from layer_opt.lrs, 
            # but CosineAnnealing() alters the layer_opt.lrs value initially 
            # (divides by 100)
            if np.sum(layer_opt.wds) == 0:
                print(f'fit() warning: use_wd_sched is set to True, but weight '
                f'decay(s) passed are 0. Use wds to pass weight decay values.')
            n_batches = len(data.train)
            self.wd_sched = WeightDecaySchedule(layer_opt, n_batches, cl, 
                                                self.lparams.cycle_mult, 
                                                n_cycles, self.lparams.norm_wds, 
                                                self.lparams.wds_sched_mult)
            callbacks += [self.wd_sched]

        if use_clr is not None:
            clr_div, cut_div = use_clr[:2]
            momenta = use_clr[2:] if len(use_clr) > 2 else None
            assert cl, 'use_clr requires cycle_len argument.'
            cycle_batches = len(data.train) * cl
            self.sched = CircularLearningRate(layer_opt, 
                                              cycle_batches, 
                                              on_cycle_end=None, 
                                              div=clr_div, cut_div=cut_div, 
                                              momenta=momenta)
        elif use_alt_clr is not None:
            div, ratio = use_alt_clr[:2]
            momenta = use_alt_clr[2:] \
                      if len(use_alt_clr) > 3 else None
            assert cl, 'use_alt_clr requires the parameter cycle_len'
            cycle_batches = len(data.train) * cl
            self.sched = CircularLearningRateAlt(layer_opt, 
                                                 cycle_batches,
                                                 on_cycle_end=None, div=div, 
                                                 ratio=ratio, 
                                                 momenta=momenta)
        elif cl:
            cycle_batches = len(data.train) * cl
            self.sched = CosineAnnealing(layer_opt, cycle_batches, 
                                         on_cycle_end=None, 
                                         cycle_mult=self.lparams.cycle_mult)
        elif not self.sched: self.sched =  Recorder(layer_opt)
        callbacks += [self.sched]

        if save_best_model:
            assert self.work_dir, \
            'fit function requires argument work_dir with save_best_model'
            callbacks += [SaveBestModel(self, layer_opt, self.metrics)]

        if self.lparams.use_swa:
            # make a copy of the model to track average weights
            self.swa_model = copy.deepcopy(model)
            callbacks += [SWA(model, self.swa_model, swa_start)]

        n_epochs = int(gp_sum(cl if cl else 1, 
                             self.lparams.cycle_mult, n_cycles))
        
        return fit(model, data, n_epochs, layer_opt.opt, self.crit, 
                   metrics=self.metrics, callbacks=callbacks, 
                   regularizer=self.regularizer, clip=self.lparams.clip, 
                   fp16=self.fp16, 
                   swa_model=self.swa_model if self.lparams.use_swa else None, 
                   swa_start=self.lparams.swa_start, 
                   swa_eval_freq=self.lparams.swa_eval_freq, **kwargs)

    def get_layer_groups(self): return self.models.get_layer_groups()

    def get_layer_opt(self, lrs, wds):

        r"""
        Method returns an instance of the LayerOptimizer class, which allows for 
        setting differential learning rates for different parts of the model.

        Args:
            lrs (float or list(float)): Learning rate(s) for the model
            wds (float or list(float)): Weight decay parameter(s).

        Returns: An instance of a LayerOptimizer
        """
        return LayerOptimizer(self.optimizer, self.get_layer_groups(), lrs, wds)

    def fit(self, **kwargs):

        r"""
        Method gets an instance of LayerOptimizer and delegates to self.fit_gen.

        Note that one can specify a list of learning rates which, when 
        appropriately defined, will be applied to different segments of an 
        architecture. This seems mostly relevant to ImageNet-trained models, 
        where we want to alter the layers closest to the images by much smaller 
        amounts.

        Likewise, a single or list of weight decay parameters can be specified, 
        which if appropriate for a model, will apply variable weight decay 
        parameters to different segments of the model.

        Args:
            lrs (float or list(float)): Learning rate for the model.
            n_cycles (int): Number of cycles to fit the model for.
            wds (float or list(float)): Weight decay parameter(s).
            kwargs: Other arguments

        Returns: None
        """
        self.sched = None
        layer_opt = self.get_layer_opt(self.lparams.lrs, self.lparams.wds)
        return self._fit(self.model, self.data, layer_opt, **kwargs)

    def thaw(self, gradual, chain_thaw=False, thaw_all=True, last=False, 
             clr=None, alt_clr=None):
        if gradual:
            # gradual unfreezing as specified in arxiv.org/abs/1801.06146
            self.freeze_to(-1)
            self.fit(n_cycles=1, cycle_len=1, use_clr=clr, use_alt_clr=alt_clr, 
                     callbacks=[])
            self.freeze_to(-2)
            self.fit(n_cycles=1, cycle_len=1, use_clr=clr, use_alt_clr=alt_clr, 
                     callbacks=[])

        if chain_thaw:
            lrs = [0.0001 for _ in range(5)]
            nl = len(self.get_layer_groups())
            
            # fine-tune last layer
            self.freeze_to(-1)
            self.fit(n_cycles=1, cycle_len=1, use_clr=clr, use_alt_clr=alt_clr, 
                     callbacks=[])
            
            # fine-tune all layers up to the second-last one
            n = 0
            while n < nl - 1:
                freeze_all_but(self, n)
                self.fit(n_cycles=1, cycle_len=1, use_clr=clr, 
                         use_alt_clr=alt_clr, callbacks=[])
                n += 1

        if thaw_all:
            self.unfreeze()
        else:
            self.freeze_to(-3)

        if last:
            self.freeze_to(-1)

    def warm_up(self, lr, wds=None):
        layer_opt = self.get_layer_opt(lr / 4, self.lparams.wds)
        self.sched = LearningRateFinder(layer_opt, len(self.data.train), lr, 
                                        linear=True)
        return self.fit_gen(self.model, self.data, layer_opt, 1)

    def lr_find(self, start_lr=1e-5, end_lr=10, wds=None, linear=False, **kwargs):
        r"""
        Helps you find an optimal learning rate for a model.

         It uses the technique developed in the 2015 paper Cyclical Learning 
         Rates for Training Neural Networks, arxiv.org/abs/1506.01186 where we 
         simply keep increasing the learning rate from a very small value, until
         the loss starts decreasing.

        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                    to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            wds (iterable/float)

        Examples:
            As training moves us closer to the optimal weights for a model, the 
            optimal learning rate will be smaller. We can take advantage of that 
            knowledge and provide lr_find() with a starting learning rate 1000x 
            smaller than the model's current learning rate as such:

            >> learn.lr_find(lr / 1000)

            >> lrs = np.array([1e-4, 1e-3, 1e-2])
            >> learn.lr_find(lrs / 1000)

        Notes:
            lr_find() may finish before going through each batch of examples if
            the loss decreases enough.
        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LearningRateFinder(layer_opt, len(self.data.train), 
                                        end_lr, linear=linear)
        self.fit_gen(self.model, self.data, layer_opt, 1, **kwargs)
        self.load('tmp')

    def lr_find_alt(self, start_lr=1e-5, end_lr=10, num_it=100, wds=None, 
                    linear=False, stop=True, **kwargs):
        r"""
        A variant of lr_find() that helps find the best learning rate. It 
        doesn't do an epoch but a fixed num of iterations (which may be more or 
        less than an epoch depending on your data).
        
        At each step, it computes the validation loss and the metrics on the next
        batch of the validation data, so it's slower than lr_find().

        Args:
            start_lr (float/numpy array): Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float): The maximum learning rate to try.
            num_it: The number of iterations you want it to run
            wds (iterable/float)
            stop : Stops (or not) when the losses starts to explode.
        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LearningRateFinderAlt(layer_opt, num_it, end_lr, 
                                           linear=linear, metrics=self.metrics, 
                                           stop=stop)
        self.fit_gen(self.model, self.data, layer_opt, 
                     num_it // len(self.data.train) + 1, all_val=True, **kwargs)
        self.load('tmp')

    def predict(self, is_test=False, use_swa=False):
        dl = self.data.test if is_test else self.data.val
        m = self.swa_model if use_swa else self.model
        return predict(m, dl)

    def predict_with_targs(self, is_test=False, use_swa=False):
        dl = self.data.test if is_test else self.data.val
        m = self.swa_model if use_swa else self.model
        return predict_with_targs(m, dl)

    def fit_opt_sched(self, phases, cycle_save_name=None, best_save_name=None, 
                      stop=False, data_list=None, callbacks=None, cut=None,
                      use_swa=False, swa_start=1, swa_eval_freq=5, **kwargs):
        r"""
        Wraps the content of phases to send them to model.fit

        This will split the training in several parts, each with their own 
        learning rates/wds/momenta/optimizer detailed in phases.

        Additionaly we can add a list of different data objets in data_list to 
        train on different datasets (to change the size for instance) for each 
        of these groups.

        Arguments:
            phases: A list of TrainingPhase objects
            stop: When True, stops the training if the loss goes too high
            data_list: A list of different Data objects.
            kwargs: Other arguments
            use_swa (bool, optional): When this is set to True, it will enable 
                    the use of Stochastic Weight Averaging,
                    arxiv.org/abs/1803.05407. The learner will include an 
                    additional model (in the swa_model attribute) for keeping 
                    track of the average weights as described in the paper. All 
                    testing of this technique so far has been in image 
                    classification, so use in other contexts is not guaranteed 
                    to work.
            swa_start (int, optional): If use_swa is set to True, then this 
                    determines the epoch to start keeping track of the average 
                    weights. It is 1-indexed per the paper's conventions.
            swa_eval_freq (int, optional): If use_swa is set to True, this 
                    determines the frequency at which to evaluate the 
                    performance of the swa_model. This evaluation can be costly
                    for models using BatchNorm (requiring a full pass through 
                    the data), which is why the default is not to evaluate after
                    each epoch.
        Returns: None
        """
        if data_list is None: data_list = []
        if callbacks is None: callbacks = []
        layer_opt = LayerOptimizer(phases[0].optimizer, self.get_layer_groups(), 
                                   1e-2, phases[0].wds)
        if len(data_list) == 0: 
            nb_batches = [len(self.data.train)] * len(phases)
        else: nb_batches = [len(data.trn_dl) for data in data_list] 
        
        self.sched = OptimScheduler(layer_opt, phases, nb_batches, stop)
        callbacks.append(self.sched)
        metrics = self.metrics
        
        if best_save_name is not None:
            callbacks += [SaveBestModel(self, layer_opt, metrics, best_save_name)]
        if use_swa:
            # make a copy of the model to track average weights
            self.swa_model = copy.deepcopy(self.model)
            callbacks += [SWA(self.model, self.swa_model, swa_start)]
        n_epochs = [phase.epochs for phase in phases] if cut is None else cut
        if len(data_list) == 0: data_list = [self.data]
        return fit(self.model, data_list, n_epochs,layer_opt, self.crit, 
                   metrics=metrics, callbacks=callbacks, 
                   regularizer=self.regularizer, clip=self.clip, fp16=self.fp16, 
                   swa_model=self.swa_model if use_swa else None, 
                   swa_start=swa_start, swa_eval_freq=swa_eval_freq, **kwargs)

    def _get_crit(self, data): return F.mse_loss

