from .imports import *
from .torch_imports import *
from .layer_optimizer import *
from .logging import *
from enum import IntEnum

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
        mult = self.lr_mult * self.iteration if self.linear \
                                             else self.lr_mult ** self.iteration
        return init_lrs * mult

    def on_batch_end(self, metrics):
        loss = metrics[0] if isinstance(metrics, list) else metrics
        if self.stop and (math.isnan(loss) or loss > self.best * 4):
            return True
        if (loss < self.best and self.iteration > 10): self.best = loss
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
        if self.iteration == self.nb:
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
        if self.iteration < self.nb / 20:
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
                 momentums=None):
        self.nb = nb
        self.div = div
        self.cut_div = cut_div
        self.on_cycle_end = on_cycle_end
        if momentums is not None:
            self.momentums = momentums
        super().__init__(layer_opt, record_momentum=(momentums is not None))

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
        res = self.momentums[1] + ratio * (self.momentums[0] - self.momentums[1])
        return res

class CircularLearningRateAlt(LearningRateUpdater):
    r""" 
    A variant of the Circular Learning Rate proposed in A disciplined approach 
    to neural network hyper-parameters: Part 1 -- learning rate, batch size, 
    momentum, and weight decay (arxiv.org/abs/1803.09820).
    """
    def __init__(self, layer_opt, nb, div=10, ratio=10, on_cycle_end=None, 
                 momentums=None):
        self.nb = nb
        self.div = div
        self.ratio = ratio
        self.on_cycle_end = on_cycle_end
        self.cycle_nb = int(nb * (1 - ratio / 100) / 2)
        if momentums is not None:
            self.momentums = momentums
        super().__init__(layer_opt, record_momentum=(momentums is not None))

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
            res = self.momentums[0]
        elif self.cycle_iter > self.cycle_nb:
            ratio = 1 - (self.cycle_iter - self.cycle_nb) / self.cycle_nb
            res = self.momentums[0] + ratio * \
                 (self.momentums[1] - self.momentums[0])
        else:
            ratio = self.cycle_iter / self.cycle_nb
            res = self.momentums[0] + ratio * \
                 (self.momentums[1] - self.momentums[0])
        return res

class WeightDecaySchedule(Callback):
    def __init__(self, layer_opt, batch_per_epoch, cycle_len, cycle_mult, 
                 n_cycles, norm_wds=False, wds_sched_mult=None):
        r"""
        Implements the weight decay schedule as proposed in Fixing Weight Decay 
        Regularization in Adam (arxiv.org/abs/1711.05101).

        Arguments:
            layer_opt: Object if class LayerOptimizer.
            batch_per_epoch: Number of batches in an epoch.
            cycle_len: Num epochs in initial cycle. 
                    Subsequent cycle_len = previous cycle_len * cycle_mult
            cycle_mult: The cycle multiplier.
            n_cycles: Number of cycles to be executed.
        """
        super().__init__()
        self.layer_opt = layer_opt
        self.batch_per_epoch = batch_per_epoch
        
        # weight decat schedule as set by user
        self.init_wds = np.array(layer_opt.wds)
        
        # Learning rates as initlaised by user
        self.init_lrs = np.array(layer_opt.lrs)
        
        # holds the new weight decay factors, calculated with on_batch_begin()
        self._wds = None
        
        self.iteration = 0
        self.epoch = 0
        self.wds_sched_mult = wds_sched_mult
        self.norm_wds = norm_wds
        self.wds_history = list()

        # pre-calculating the number of epochs in current cycle
        self.epoch_to_num_cycles = dict()
        i = 0
        
        for cycle in range(n_cycles):
            for _ in range(cycle_len):
                self.epoch_to_num_cycles[i] = cycle_len
                i += 1
            cycle_len *= cycle_mult

    def on_train_begin(self):
        self.iteration = 0
        self.epoch = 0

    def on_batch_begin(self):
        _wds = self.init_wds

        # weight decay multiplier
        eta = 1.
        
        if self.wds_sched_mult is not None:
            _wds = self.wds_sched_mult(self)

        # normalize weight decay
        if self.norm_wds:
            _wds = _wds / np.sqrt(self.batch_per_epoch * \
                                self.epoch_to_num_cycles[self.epoch])
        
        self._wds = eta * _wds

        # Set weight_decay with zeros so that it is not applied in Adam, we will 
        # apply it outside in on_batch_end()
        self.layer_opt.set_wds_out(self._wds)
        
        # we have to save the existing weights before the optimizer changes the 
        # values
        self.iteration += 1

    def on_epoch_end(self, metrics):
        self.epoch += 1

class DecayType(IntEnum):
    r""" Data class to enumerate each decay type. 
    """
    NO = 1
    LINEAR = 2
    COSINE = 3
    EXPONENTIAL = 4
    POLYNOMIAL = 5

class DecayScheduler(object):
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
        self.iteration = 0
        if self.end_val is None and not (self.dec_type in [1, 4]): 
            self.end_val = 0
    
    def next_val(self):
        self.iteration += 1
        if self.dec_type == DecayType.NO:
            return self.start_val
        elif self.dec_type == DecayType.LINEAR:
            ratio = self.iteration / self.nb
            return self.start_val + ratio * (self.end_val - self.start_val)
        elif self.dec_type == DecayType.COSINE:
            rate = np.cos(np.pi * (self.iteration) / self.nb) + 1
            return self.end_val + (self.start_val - self.end_val) / 2 * rate
        elif self.dec_type == DecayType.EXPONENTIAL:
            ratio = self.end_val / self.start_val
            return self.start_val * (ratio **  (self.iteration / self.nb))
        elif self.dec_type == DecayType.POLYNOMIAL:
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
        if (loss < self.best and self.iteration > 10): self.best = loss
        super().on_batch_end(metrics)
        self.phases[self.phase].update()
    
    def on_phase_begin(self):
        self.phases[self.phase].phase_begin(self.layer_opt, 
                                            self.nb_batches[self.phase])
    
    def on_phase_end(self):
        self.phase += 1

    def plot_lr(self, show_text=True, show_momentums=True):
        r""" Plots the lr rate and momentum schedule.
        """
        phase_limits = [0]
        for nb_batch, phase in zip(self.nb_batches, self.phases):
            phase_limits.append(phase_limits[-1] + nb_batch * phase.epochs)
        plt.switch_backend('agg')
        np_plts = 2 if show_momentums else 1
        fig, axs = plt.subplots(1,np_plts, figsize=(6 * np_plts, 4))
        if not show_momentums: axs = [axs]
        for i in range(np_plts): axs[i].set_xlabel('iterations')
        axs[0].set_ylabel('learning rate')
        axs[0].plot(self.iterations, self.lrs)
        if show_momentums:
            axs[1].set_ylabel('momentum')
            axs[1].plot(self.iterations, self.momentums)
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
            epochs: number of epochs to train like this.
            optimizer: an optimizer (example optim.Adam).
            lr: one learning rate or a tuple of the form (start_lr,end_lr) each 
                    of those can be a list/numpy array for differential learning 
                    rates.
            lr_decay: a DecayType object specifying how the learning rate should
                    change.
            momentum: one momentum (or beta1 in case of Adam), or a tuple of the
                    form (start, end).
            momentum_decay: a DecayType object specifying how the momentum 
                    should change.
            beta: beta2 parameter in Adam or alpha parameter in RMSProp.
            wds: weight decay (can be an array for differential wds).
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