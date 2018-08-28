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

def summarize(m, inputs):
    r""" Source: github.com/ncullen93/torchsample
    """
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
