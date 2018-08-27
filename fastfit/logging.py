from .imports import *
from .layer_optimizer import *
from .callbacks import Callback
from timeit import default_timer as timer
import copy


class LoggingCallback(Callback):
    r"""
    Class for maintaining status of a long-running job.

    Example usage:
    >>> learn.fit(0.01, 1, callbacks=[LoggingCallback(save_path='./logs')])
    """
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
    
    def on_train_begin(self):
        self.batch = 0
        self.epoch = 0
        self.phase = 0
        self.f = open(self.save_path, 'a', 1)
        self.log(f'\ton_train_begin')
    
    def on_batch_begin(self):
        self.log(f'{self.batch}\ton_batch_begin')
    
    def on_phase_begin(self):
        self.log(f'{self.phase}\ton_phase_begin')
    
    def on_epoch_end(self, metrics):
        self.log(f'{self.epoch}\ton_epoch_end: {metrics}')
        self.epoch += 1
    
    def on_phase_end(self):
        self.log(f'{self.phase}\ton_phase_end')
        self.phase += 1
    
    def on_batch_end(self, metrics):
        self.log(f'{self.batch}\ton_batch_end: {metrics}')
        self.batch += 1
    
    def on_train_end(self):
        self.log("\ton_train_end")
        self.f.close()
    
    def log(self, message):
        self.f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")}\t{message}\n')

class Recorder(Callback):
    r"""
    Saves and displays loss functions and other metrics. Default sched when none
    is specified in a learner. 
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
        self.losses, self.lrs, self.iter, self.epochs, self.times =  \
                                                        [[] for _ in range(5)]
        self.start_at = timer()
        self.val_losses, self.rec_metrics = [], []
        if self.record_momentum:
            self.momentums = []
        self.iteration = 0
        self.epoch = 0

    def on_epoch_end(self, metrics):
        self.epoch += 1
        self.epochs.append(self.iter)
        self.times.append(timer() - self.start_at)
        self.save_metrics(metrics)

    def on_batch_end(self, loss):
        self.iteration += 1
        self.lrs.append(self.layer_opt.lr)
        self.iter.append(self.iter)
        if isinstance(loss, list):
            self.losses.append(loss[0])
            self.save_metrics(loss[1:])
        else: self.losses.append(loss)
        if self.record_momentum: self.momentums.append(self.layer_opt.momentum)

    def save_metrics(self,vals):
        self.val_losses.append(delistify(vals[0]))
        if len(vals) > 2: self.rec_metrics.append(vals[1:])
        elif len(vals) == 2: self.rec_metrics.append(vals[1])

    def plot_loss(self, n_skip=10, n_skip_end=5):
        r"""
        Plots loss function. Plot will be displayed in console and both plot and 
        loss values are saved in save_path. 
        """
        plt.switch_backend('agg')
        plt.plot(self.iter[n_skip: -n_skip_end], self.losses[n_skip: -n_skip_end])
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
            axs[0].plot(self.iterations, self.lrs)
            axs[1].plot(self.iterations, self.momentums)   
        else:
            plt.xlabel('iterations')
            plt.ylabel('learning rate')
            plt.plot(self.iterations, self.lrs)
            plt.savefig(os.path.join(self.save_path, 'learning_rate_sched.png'))

class TensorBoard(Callback):
    # TO DO: implement this
    def __init__(self):
        super().__init__()
        return

class SaveBestModel(Recorder):
    
    r""" 
    Save weights of the best model based during training. If metrics are 
    provided, the first metric in the list is used to find the best model. If no
    metrics are provided, the loss is used.
           
    Example usage:
    Briefly, you have your model 'learn' variable and call fit.
    >>> learn.fit(lr, 2, cycle_len=2, cycle_mult=1, best_save_name='best')
    """
    def __init__(self, model, layer_opt, metrics, name='best'):
        super().__init__(layer_opt)
        self.name = name
        self.model = model
        self.best_loss = None
        self.best_acc = None
        self.save_method = self.no_metrics_save if metrics == None \
                                                else self.metrics_save
    def no_metrics_save(self, metrics):
        loss = metrics[0]
        if self.best_loss == None or loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')
    
    def metrics_save(self, metrics):
        loss, acc = metrics[0], metrics[1]
        if self.best_acc == None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.model.save(f'{self.name}')
        elif acc == self.best_acc and  loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')

    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        self.save_method(metrics)

def draw_line(ax,x):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.plot([x, x], [ymin, ymax], color='red', linestyle='dashed')

def draw_text(ax,x, text):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(x, (ymin + ymax) / 2, text, horizontalalignment='center', 
            verticalalignment='center', fontsize=14, alpha=0.5)

def curve_smoothing(vals, beta):
    avg_val = 0
    smoothed = []
    for (i, v) in enumerate(vals):
        avg_val = beta * avg_val + (1 - beta) * v
        smoothed.append(avg_val / (1 - beta ** (i + 1)))
    return smoothed

# github.com/ncullen93/torchsample
def summarize(m, inputs):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f'{class_name}-{module_idx + 1}'
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] \
                                                    for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and module.bias is not None:
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
            not isinstance(module, nn.ModuleList) and
            not (module == m)):
            hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register forward hooks
    m.apply(register_hook)
    
    xs = [to_gpu(Variable(x)) for x in inputs]
    m(*xs)

    # remove these hooks
    for h in hooks: h.remove()
    return summary
