from .imports import *
from .layer_optimizer import *
from .utils.extras import draw_line, draw_text, curve_smoothing
from enum import IntEnum

class Recorder(Callback):
    r"""
    Saves and displays loss functions and other metrics. Also the default 
    learning rate schedule when none is specified in a learner. 
    """
    def __init__(self, layer_opt, save_path='', record_momentum=False, 
                 metrics=[]):
        super().__init__()
        self.layer_opt = layer_opt
        self.init_lrs = np.array(layer_opt.lrs)
        self.save_path = save_path 
        self.record_momentum = record_momentum 
        self.metrics = metrics

    def on_train_begin(self):
        self.losses, self.lrs, self.iters, self.epochs, self.times =  \
                                                        [[] for _ in range(5)]
        self.start_at = timer()
        self.val_losses, self.rec_metrics = [], []
        if self.record_momentum:
            self.momenta = []
        self.iter_ = 0
        self.epoch = 0

    def on_epoch_end(self, metrics):
        self.epoch += 1
        self.epochs.append(self.iter_)
        self.times.append(timer() - self.start_at)
        self.save_metrics(metrics)

    def on_batch_end(self, loss):
        self.iter_ += 1
        self.lrs.append(self.layer_opt.lr)
        self.iters.append(self.iter_)
        if isinstance(loss, list):
            self.losses.append(loss[0])
            self.save_metrics(loss[1:])
        else: self.losses.append(loss)
        if self.record_momentum: self.momenta.append(self.layer_opt.momentum)

    def save_metrics(self, metrics):
        self.val_losses.append(delistify(metrics[0]))
        if len(metrics) > 2: self.rec_metrics.append(metrics[1:])
        elif len(metrics) == 2: self.rec_metrics.append(metrics[1])

    def plot_loss(self, n_skip=10, n_skip_end=5):
        r"""
        Plots loss function. Plot will be displayed in console and both plot and 
        loss values are saved in save_path. 
        """
        plt.switch_backend('agg')
        plt.plot(self.iters[n_skip : -n_skip_end], 
                 self.losses[n_skip : -n_skip_end])
        plt.savefig(os.path.join(self.save_path, 'loss.png'))
        np.save(os.path.join(self.save_path, 'losses.npy'), self.losses[10:])

    def plot_learningrate(self):
        r"""
        Plots learning rate in console, depending on the enviroment of the 
        learner.
        """
        plt.switch_backend('agg')
        if self.record_momentum:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            for i in range(0, 2): axs[i].set_xlabel('iterations')
            axs[0].set_ylabel('learning rate')
            axs[1].set_ylabel('momentum')
            axs[0].plot(self.iters, self.lrs)
            axs[1].plot(self.iters, self.momenta)   
        else:
            plt.xlabel('iterations')
            plt.ylabel('learning rate')
            plt.plot(self.iters, self.lrs)
            plt.savefig(os.path.join(self.save_path, 'learning_rate_sched.png'))

class LearningRateUpdater(Recorder):
    r"""
    Abstract class from which all learning rate updaters inherit. Calculates and 
    updates new learning rate and momentum at the end of each batch. Implemented
    as a callback.
    """
    def on_train_begin(self):
        super().on_train_begin()
        self.update_lr()
        if self.record_momentum:
            self.update_momentum()

    def on_batch_end(self, loss):
        res = super().on_batch_end(loss)
        self.update_lr()
        if self.record_momentum:
            self.update_momentum()
        return res

    def update_lr(self):
        new_lrs = self.calc_lr(self.init_lrs)
        self.layer_opt.set_lrs(new_lrs)
    
    def update_momentum(self):
        new_momentum = self.calc_momentum()
        self.layer_opt.set_momentum(new_momentum)

    @abstractmethod
    def calc_lr(self, init_lrs): raise NotImplementedError
    
    @abstractmethod
    def calc_momentum(self): raise NotImplementedError


class LearningRateFinder(LearningRateUpdater):
    r"""
    Helps you find an optimal learning rate for your model, as per Cyclical 
    Learning Rates for Training Neural Networks, arxiv.org/abs/1506.01186. 
    Learning rate is increased in linear or log scale, depending on user input, 
    and the result of the loss funciton is retained and can be plotted later. 
    """
    def __init__(self, layer_opt, nb, end_lr=10, linear=False, metrics = []):
        self.linear = linear
        self.stop = True
        ratio = end_lr / layer_opt.lr
        self.lr_mult = (ratio / nb) if linear else ratio ** (1 / nb)
        super().__init__(layer_opt, metrics=metrics)

    def on_train_begin(self):
        super().on_train_begin()
        self.best = 1e9

    def calc_lr(self, init_lrs):
        mult = self.lr_mult * self.iter_ if self.linear \
                                             else self.lr_mult ** self.iter_
        return init_lrs * mult

    def on_batch_end(self, metrics):
        loss = metrics[0] if isinstance(metrics, list) else metrics
        if self.stop and (math.isnan(loss) or loss > self.best * 4):
            return True
        if (loss < self.best and self.iter_ > 10): self.best = loss
        return super().on_batch_end(metrics)

    def plot(self, n_skip=10, n_skip_end=5):
        r""" Plots the loss function with respect to learning rate, in log scale. 
        """
        plt.ylabel('validation loss')
        plt.xlabel('learning rate (log scale)')
        plt.plot(self.lrs[n_skip : -(n_skip_end + 1)], 
                 self.losses[n_skip : -(n_skip_end + 1)])
        plt.xscale('log')

class LearningRateFinderAlt(LearningRateFinder):
    r"""
    A variant of lr_find() that helps find the best learning rate. It doesn't do
    an epoch but a fixed num of iterations (which may be more or less than an 
    epoch depending on your data).
    """
    def __init__(self, layer_opt, nb, end_lr=10, linear=False, metrics=[], 
                 stop=True):
        self.nb = nb
        self.metrics = metrics
        super().__init__(layer_opt, nb, end_lr, linear, metrics)
        self.stop = stop

    def on_batch_end(self, loss):
        if self.iter_ == self.nb:
            return True
        return super().on_batch_end(loss)

    def plot(self, n_skip=10, n_skip_end=5, smoothed=True):
        if self.metrics is None: self.metrics = []
        n_plots = len(self.metrics) + 2
        fig, axs = plt.subplots(n_plots, figsize=(6, 4 * n_plots))
        
        for i in range(0, n_plots): axs[i].set_xlabel('learning rate')
        
        axs[0].set_ylabel('training loss')
        axs[1].set_ylabel('validation loss')
        
        for i, m in enumerate(self.metrics):
            axs[i + 2].set_ylabel(m.__name__)
            if len(self.metrics) == 1:
                values = self.rec_metrics
            else:
                values = [rec[i] for rec in self.rec_metrics]
            if smoothed: values = curve_smoothing(values, 0.98)
            axs[i + 2].plot(self.lrs[n_skip : -n_skip_end], 
                            values[n_skip : -n_skip_end])
        
        plot_loss = curve_smoothing(self.val_losses, 0.98) \
                                    if smoothed else self.val_losses
        axs[0].plot(self.lrs[n_skip : -n_skip_end], 
                    self.losses[n_skip : -n_skip_end])
        axs[1].plot(self.lrs[n_skip : -n_skip_end], 
                    plot_loss[n_skip : -n_skip_end])

class CosineAnnealing(LearningRateUpdater):
    r""" Learning rate scheduler that implements a cosine annealing schedule.
    """
    def __init__(self, layer_opt, nb, on_cycle_end=None, cycle_mult=1):
        self.nb = nb
        self.on_cycle_end = on_cycle_end,
        self.cycle_mult = cycle_mult
        super().__init__(layer_opt)

    def on_train_begin(self):
        self.cycle_iter = 0
        self.cycle_count = 0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        if self.iter_ < self.nb / 20:
            self.cycle_iter += 1
            return init_lrs / 100.

        rate = np.cos(np.pi * (self.cycle_iter) / self.nb) + 1
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return init_lrs / 2 * rate


class CircularLearningRate(LearningRateUpdater):
    r"""
    A learning rate updater that implements the Circular Learning Rate scheme
    (arxiv.org/abs/1506.01186). Learning rate is increased and then decreased 
    linearly. 
    """
    def __init__(self, layer_opt, nb, div=4, cut_div=8, on_cycle_end=None, 
                 momenta=None):
        self.nb = nb
        self.div = div
        self.cut_div = cut_div
        self.on_cycle_end = on_cycle_end
        if momenta is not None:
            self.momenta = momenta
        super().__init__(layer_opt, record_momentum=(momenta is not None))

    def on_train_begin(self):
        self.cycle_iter = 0
        self.cycle_count = 0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        cut_point = self.nb // self.cut_div
        if self.cycle_iter > cut_point:
            ratio = 1 - (self.cycle_iter - cut_point) / (self.nb - cut_point)
        else: ratio = self.cycle_iter / cut_point
        res = init_lrs * (1 + ratio * (self.div - 1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res
    
    def calc_momentum(self):
        cut_point = self.nb // self.cut_div
        if self.cycle_iter > cut_point:
            ratio = (self.cycle_iter - cut_point) / (self.nb - cut_point)
        else: ratio = 1 - self.cycle_iter / cut_point
        res = self.momenta[1] + ratio * (self.momenta[0] - self.momenta[1])
        return res

class CircularLearningRateAlt(LearningRateUpdater):
    r""" 
    A variant of the Circular Learning Rate proposed in A disciplined approach 
    to neural network hyper-parameters: Part 1 -- learning rate, batch size, 
    momentum, and weight decay (arxiv.org/abs/1803.09820).
    """
    def __init__(self, layer_opt, nb, div=10, ratio=10, on_cycle_end=None, 
                 momenta=None):
        self.nb = nb
        self.div = div
        self.ratio = ratio
        self.on_cycle_end = on_cycle_end
        self.cycle_nb = int(nb * (1 - ratio / 100) / 2)
        if momenta is not None:
            self.momenta = momenta
        super().__init__(layer_opt, record_momentum=(momenta is not None))

    def on_train_begin(self):
        self.cycle_iter = 0
        self.cycle_count = 0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        if self.cycle_iter > 2 * self.cycle_nb:
            ratio = (self.cycle_iter - 2 * self.cycle_nb) \
                / (self.nb - 2 * self.cycle_nb)
            res = init_lrs * (1 + (ratio * (1 - 100) / 100)) / self.div
        elif self.cycle_iter > self.cycle_nb:
            ratio = 1 - (self.cycle_iter - self.cycle_nb)/self.cycle_nb
            res = init_lrs * (1 + ratio * (self.div - 1)) / self.div
        else:
            ratio = self.cycle_iter / self.cycle_nb
            res = init_lrs * (1 + ratio * (self.div - 1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res

    def calc_momentum(self):
        if self.cycle_iter > 2 * self.cycle_nb:
            res = self.momenta[0]
        elif self.cycle_iter > self.cycle_nb:
            ratio = 1 - (self.cycle_iter - self.cycle_nb) / self.cycle_nb
            res = self.momenta[0] + ratio * \
                 (self.momenta[1] - self.momenta[0])
        else:
            ratio = self.cycle_iter / self.cycle_nb
            res = self.momenta[0] + ratio * \
                 (self.momenta[1] - self.momenta[0])
        return res

class WeightDecaySchedule(Callback):
    def __init__(self, layer_opt, n_batches, cycle_len, cycle_mult, 
                 n_cycles, norm_wds=False, wds_sched_mult=None):
        r"""
        Implements the weight decay schedule as proposed in Fixing Weight Decay 
        Regularization in Adam (arxiv.org/abs/1711.05101).

        Arguments:
            layer_opt: Object if class LayerOptimizer.
            n_batches: Number of batches in an epoch.
            cycle_len: Num epochs in initial cycle. 
                    Subsequent cycle_len = previous cycle_len * cycle_mult
            cycle_mult: The cycle multiplier.
            n_cycles: Number of cycles to be executed.
        """
        super().__init__()
        self.layer_opt = layer_opt
        self.n_batches = n_batches
        
        # weight decay parameter value as set by user
        self.init_wds = np.array(layer_opt.wds)
        
        # learning rate as initlaised by user
        self.init_lrs = np.array(layer_opt.lrs)
        
        # holds the new weight decay factors, calculated at on_batch_begin()
        self._wds = None
        
        self.iter_ = 0
        self.epoch = 0
        self.wds_sched_mult = wds_sched_mult
        self.norm_wds = norm_wds
        self.wds_history = list()

        # pre-calculating the number of epochs in current cycle
        self.cycle_epochs = dict()
        idx = 0
        for cycle in range(n_cycles):
            for _ in range(cycle_len):
                self.cycle_epochs[idx] = cycle_len
                idx += 1
            cycle_len *= cycle_mult

    def on_train_begin(self):
        self.iter_ = 0
        self.epoch = 0

    def on_batch_begin(self):
        _wds = self.init_wds

        # weight decay multiplier
        eta = 1.
        
        if self.wds_sched_mult is not None:
            _wds = self.wds_sched_mult(self)

        # normalize weight decay
        if self.norm_wds:
            _wds = _wds / np.sqrt(self.n_batches * self.cycle_epochs[self.epoch])
        
        self._wds = eta * _wds

        # Set weight_decay with zeros so that it is not applied in Adam, we will 
        # apply it outside in on_batch_end()
        self.layer_opt.set_wds_out(self._wds)
        
        # we have to save the existing weights before the optimizer changes the 
        # values
        self.iter_ += 1

    def on_epoch_end(self, metrics):
        self.epoch += 1

class RateDecayType(IntEnum):
    r""" Data class to enumerate each learning rate decay type. 
    """
    NO = 1
    LINEAR = 2
    COSINE = 3
    EXPONENTIAL = 4
    POLYNOMIAL = 5

class RateDecayScheduler(object):
    r"""
    Given initial and end values, this class generates the next value depending 
    on decay type and number of iterations. (by calling next_val().) 
    """
    def __init__(self, decay_type, n_iterations, start_val, end_val=None, 
                 extra=None):
        self.decay_type = decay_type 
        self.nb = n_iterations
        self.start_val = start_val 
        self.end_val = end_val 
        self.extra = extra
        self.iter_ = 0
        if self.end_val is None and not (self.dec_type in [1, 4]): 
            self.end_val = 0
    
    def next_val(self):
        self.iter_ += 1
        if self.dec_type == RateDecayType.NO:
            return self.start_val
        elif self.dec_type == RateDecayType.LINEAR:
            ratio = self.iter_ / self.nb
            return self.start_val + ratio * (self.end_val - self.start_val)
        elif self.dec_type == RateDecayType.COSINE:
            rate = np.cos(np.pi * (self.iter_) / self.nb) + 1
            return self.end_val + (self.start_val - self.end_val) / 2 * rate
        elif self.dec_type == RateDecayType.EXPONENTIAL:
            ratio = self.end_val / self.start_val
            return self.start_val * (ratio **  (self.iter_ / self.nb))
        elif self.dec_type == RateDecayType.POLYNOMIAL:
            return self.end_val + (self.start_val - self.end_val) \
                    * (1 - self.it / self.nb) ** self.extra    

class OptimScheduler(Recorder):
    r""" Learning rate Scheduler for training involving multiple phases.
    """
    def __init__(self, layer_opt, phases, nb_batches, stop = False):
        self.phases = phases 
        self.nb_batches = nb_batches
        self.stop = stop
        super().__init__(layer_opt, record_momentum=True)

    def on_train_begin(self):
        super().on_train_begin()
        self.phase = 0
        self.best = 1e9

    def on_batch_end(self, metrics):
        loss = metrics[0] if isinstance(metrics, list) else metrics
        if self.stop and (math.isnan(loss) or loss > self.best * 4):
            return True
        if (loss < self.best and self.iter_ > 10): self.best = loss
        super().on_batch_end(metrics)
        self.phases[self.phase].update()
    
    def on_phase_begin(self):
        self.phases[self.phase].phase_begin(self.layer_opt, 
                                            self.nb_batches[self.phase])
    
    def on_phase_end(self):
        self.phase += 1

    def plot_lr(self, show_text=True, show_momenta=True):
        r""" Plots the lr rate and momentum schedule.
        """
        phase_limits = [0]
        for nb_batch, phase in zip(self.nb_batches, self.phases):
            phase_limits.append(phase_limits[-1] + nb_batch * phase.epochs)
        plt.switch_backend('agg')
        np_plts = 2 if show_momenta else 1
        fig, axs = plt.subplots(1,np_plts, figsize=(6 * np_plts, 4))
        if not show_momenta: axs = [axs]
        for i in range(np_plts): axs[i].set_xlabel('iterations')
        axs[0].set_ylabel('learning rate')
        axs[0].plot(self.iters, self.lrs)
        if show_momenta:
            axs[1].set_ylabel('momentum')
            axs[1].plot(self.iters, self.momenta)
        if show_text:
            for i, phase in enumerate(self.phases):
                text = phase.opt_fn.__name__
                if phase.wds is not None: text += f'\nwds={phase.wds}'
                if phase.beta is not None: text += f'\nbeta={phase.beta}'
                for k in range(np_plts):
                    if i < len(self.phases) - 1:
                        draw_line(axs[k], phase_limits[i + 1])
                    draw_text(axs[k], 
                             (phase_limits[i] + phase_limits[i + 1]) / 2, 
                              text)
            plt.savefig(os.path.join(self.save_path, 'learning_rate.png'))
    
    def plot(self, n_skip=10, n_skip_end=5, linear=None):
        if linear is None: linear = self.phases[-1].lr_decay == DecayType.LINEAR
        plt.ylabel('loss')
        plt.plot(self.lrs[n_skip : -n_skip_end], 
                 self.losses[n_skip : -n_skip_end])
        if linear: plt.xlabel('learning rate')
        else:
            plt.xlabel('learning rate (log scale)')
            plt.xscale('log')
        plt.savefig(os.path.join(self.save_path, 'schedule.png'))

class TrainingPhase(object):
    r"""
    Object with training information for each phase, when multiple phases are 
    involved during training. Used by fit_opt_sched in learner.py
    """
    def __init__(self, epochs=1, optimizer=optim.SGD, lr=1e-2, momentum=0.9,
                 beta=None, wds=None, wd_loss=True, lr_decay=DecayType.NO, 
                 momentum_decay=DecayType.NO,):
        r"""
        Creates an object containing all the relevant informations for one part 
        of a model training.

        Arguments:
            epochs: Number of epochs to train like this.
            optimizer: An optimizer (example optim.Adam).
            lr: One learning rate or a tuple of the form (start_lr,end_lr) each 
                    of those can be a list/numpy array for differential learning 
                    rates.
            lr_decay: A DecayType object specifying how the learning rate should
                    change.
            momentum: One momentum (or beta1 in case of Adam), or a tuple of the
                    form (start, end).
            momentum_decay: A DecayType object specifying how the momentum 
                    should change.
            beta: beta2 parameter in Adam or alpha parameter in RMSProp.
            wds: Weight decay (can be an array for differential wds).
        """
        self.epochs = epochs 
        self.optimizer = optimizer 
        self.lr = lr
        self.momentum = momentum 
        self.beta = beta
        self.wds = wds
        if isinstance(lr_decay, tuple): self.lr_decay, self.extra_lr = lr_decay
        else: self.lr_decay, self.extra_lr = lr_decay, None
        if isinstance(momentum_decay, tuple): 
            self.momentum_decay, self.extra_momentum = momentum_decay
        else: self.momentum_decay, self.extra_momentum = momentum_decay, None
        self.wd_loss = wd_loss

    def phase_begin(self, layer_opt, nb_batches):
        self.layer_opt = layer_opt
        
        if isinstance(self.lr, tuple): start_lr, end_lr = self.lr
        else: start_lr, end_lr = self.lr, None
        self.lr_sched = DecayScheduler(self.lr_decay, nb_batches * self.epochs, 
                                       start_lr, end_lr, extra=self.extra_lr)
        
        if isinstance(self.momentum, tuple): 
            start_momentum, end_momentum = self.momentum
        else: start_momentum, end_momentum = self.momentum, None
        self.momentum_sched = DecayScheduler(self.momentum_decay, 
                                             nb_batches * self.epochs, 
                                             start_momentum, end_momentum, 
                                             extra=self.extra_mom)
        self.layer_opt.set_opt_fn(self.optimizer)
        self.layer_opt.set_lrs(start_lr)
        self.layer_opt.set_mom(start_momentum)
        if self.beta is not None: self.layer_opt.set_beta(self.beta)
        if self.wds is not None:
            if self.wd_loss: self.layer_opt.set_wds(self.wds)
            else: self.layer_opt.set_wds_out(self.wds)
    
    def update(self):
        new_lr = self.lr_sched.next_val()
        new_mom = self.momentum_sched.next_val()
        self.layer_opt.set_lrs(new_lr)
        self.layer_opt.set_mom(new_momentum)