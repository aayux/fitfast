from .imports import *
from .utils.core import *
from .utils.extras import *
from .layer_optimizer import *
from .tricks.swa import *
# from .fp16 import *

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')

def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]

def num_features(m):
    c = children(m)
    if len(c) == 0: return None
    for l in reversed(c):
        if hasattr(l, 'num_features'): return l.num_features
        res = num_features(l)
        if res is not None: return res

def torch_item(x): 
    return x.item() if hasattr(x, 'item') else x[0]

class Stepper():
    def __init__(self, m, opt, crit, clip=0, regularizer=None, fp16=False, 
                 loss_scale=1):
        self.m = m
        self.opt = opt
        self.crit = crit
        self.clip = clip
        self.regularizer = regularizer
        self.fp16 = fp16
        self.reset(True)
        # if self.fp16: self.fp32_params = copy_model_to_fp32(m, opt)
        self.loss_scale = loss_scale

    def reset(self, train=True):
        if train: apply_leaf(self.m, set_train_mode)
        else: self.m.eval()
        if hasattr(self.m, 'reset'): self.m.reset()
        #     if self.fp16: 
        #         self.fp32_params = copy_model_to_fp32(self.m, self.opt)

    def step(self, xs, y, epoch):
        extra = []
        output = self.m(*xs)
        if isinstance(output, tuple): output, *extra = output
        # if self.fp16: self.m.zero_grad()
        # else: 
        self.opt.zero_grad() 
        loss = raw_loss = self.crit(output, y)
        if self.loss_scale != 1: 
            # assert(self.fp16)
            loss = loss * self.loss_scale
        if self.regularizer: loss = self.regularizer(output, extra, raw_loss)
        loss.backward()
        
        # if self.fp16: update_fp32_grads(self.fp32_params, self.m)
        # if self.loss_scale != 1:
        #     for param in self.fp32_params: 
        #        param.grad.data.div_(self.loss_scale)
        
        # gradient clipping
        if self.clip:
            if IS_TORCH_04: 
                nn.utils.clip_grad_norm_(trainable_params_(self.m), self.clip)
            else: 
                nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        
        # weight decay after the gradient computation but before the step
        if 'wd' in self.opt.param_groups[0] \
                and self.opt.param_groups[0]['wd'] != 0:
            for group in self.opt.param_groups:
                lr, wd = group['lr'], group['wd']
                for p in group['params']:
                    if p.grad is not None: 
                        p.data = p.data.add(-wd * lr, p.data)
        
        self.opt.step()
        
        # if self.fp16: 
        #     copy_fp32_to_model(self.m, self.fp32_params)
        #     torch.cuda.synchronize()
        
        return torch_item(raw_loss.data)

    def evaluate(self, xs, y):
        preds = self.m(*xs)
        if isinstance(preds, tuple): preds = preds[0]
        return preds, self.crit(preds, y)

def set_train_mode(m):
    if (hasattr(m, 'running_mean') and (getattr(m,'bn_freeze', False) \
        or not getattr(m,'trainable', False))): m.eval()
    elif (getattr(m,'drop_freeze',False) and hasattr(m, 'p') \
          and ('drop' in type(m).__name__.lower())): m.eval()
    else: m.train()

def fit(model, data, n_epochs, opt, crit, metrics=None, callbacks=None, 
        stepper=Stepper, swa_model=None, swa_start=None, swa_eval_freq=None, 
        visualize=False, **kwargs):
    r""" 
    Fits the model.

    Arguments:
        model (model): Any pytorch module
        data (ModelData): See ModelData class and subclasses (can be a list).
        opts: an optimizer, for eg.: optim.Adam. 
        n_epochs(int or list): Number of epochs (or list of number of epochs). 
                If n_epochs is a list, it needs to be the layer_optimizer to get
                the optimizer as it changes. 
       crit: Loss function to optimize, for eg.: F.cross_entropy
    """

    seq_first = kwargs.pop('seq_first', False)
    all_val = kwargs.pop('all_val', False)
    get_epoch_vals = kwargs.pop('get_epoch_vals', False)
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom = 0.98
    batch_n = 0 
    avg_loss = 0.
    
    for cb in callbacks: cb.on_train_begin()
    names = ['epoch', 'trn_loss', 'val_loss'] + [f.__name__ for f in metrics]
    if swa_model is not None:
        swa_names = ['swa_loss'] + [f'swa_{f.__name__}' for f in metrics]
        names += swa_names
        # will use this to call evaluate later
        swa_stepper = stepper(swa_model, None, crit, **kwargs)

    layout = '{!s:10} ' * len(names)
    if not isinstance(n_epochs, Iterable): n_epochs = [n_epochs]
    if not isinstance(data, Iterable): data = [data]
    if len(data) == 1: data = data * len(n_epochs)
    for cb in callbacks: cb.on_phase_begin()
    model_stepper = stepper(model, opt.opt if hasattr(opt, 'opt') else opt, 
                            crit, **kwargs)
    epoch_vals = collections.OrderedDict()
    total_epochs = int(np.ceil(np.array(n_epochs).sum()))
    phase_count = np.array([ep * len(dat.trn_dl) \
                           for (ep, dat) in zip(n_epochs, data)]).cumsum()
    phase = 0
    for epoch in tnrange(total_epochs, desc='epoch'):
        # sometimes cumulated errors make this append
        if phase >= len(n_epochs): break
        model_stepper.reset(True)
        cur_data = data[phase]
        if hasattr(cur_data, 'trn_sampler'): 
            cur_data.trn_sampler.set_epoch(epoch)
        if hasattr(cur_data, 'val_sampler'): 
            cur_data.val_sampler.set_epoch(epoch)
        batch_n = len(cur_data.trn_dl)
        t = tqdm(iter(cur_data.trn_dl), leave=False, total=batch_n, miniters=0)
        if all_val: val_iter = IterateBatch(cur_data.val_dl)

        for (*x, y) in t:
            batch_n += 1
            for cb in callbacks: cb.on_batch_begin()
            loss = model_stepper.step(V(x),V(y), epoch)
            avg_loss = avg_loss * avg_mom + loss * (1 - avg_mom)
            debias_loss = avg_loss / (1 - avg_mom ** batch_n)
            t.set_postfix(loss=debias_loss, refresh=False)
            stop = False
            loss_ = debias_loss if not all_val \
            else [debias_loss] + validate_next(model_stepper, metrics, val_iter)
            
            for cb in callbacks: stop = stop or cb.on_batch_end(loss_)
            if stop: return
            if batch_n >= phase_count[phase]:
                for cb in callbacks: cb.on_phase_end()
                phase += 1
                if phase >= len(n_epochs):
                    t.close()
                    break
                for cb in callbacks: cb.on_phase_begin()
                if isinstance(opt, LayerOptimizer): model_stepper.opt = opt.opt
                if cur_data != data[phase]:
                    t.close()
                    break

        if not all_val:
            vals = validate(model_stepper, cur_data.val_dl, metrics, 
                            seq_first=seq_first)
            stop = False
            for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
            
            if swa_model is not None:
                if (epoch + 1) >= swa_start \
                and ((epoch + 1 - swa_start) % swa_eval_freq == 0 \
                or epoch == total_epochs - 1):
                    fix_batchnorm(swa_model, cur_data.trn_dl)
                    swa_vals = validate(swa_stepper, cur_data.val_dl, metrics)
                    vals += swa_vals

            if epoch > 0: 
                print_stats(epoch, [debias_loss] + vals, visualize, prev_val)
            else:
                print(layout.format(*names))
                print_stats(epoch, [debias_loss] + vals, visualize)
            
            prev_val = [debias_loss] + vals
            epoch_vals = append_stats(epoch_vals, epoch, [debias_loss] + vals)
        
        if stop: break
    
    for cb in callbacks: cb.on_train_end()
    
    if get_epoch_vals: return vals, epoch_vals
    else: return vals

def append_stats(epoch_vals, epoch, values, decimals=6):
    epoch_vals[epoch]=list(np.round(values, decimals))
    return epoch_vals

def print_stats(epoch, values, visualize, prev_val=[], decimals=6):
    layout = '{!s:^10}' + ' {!s:10}' * len(values)
    values = [epoch] + list(np.round(values, decimals))
    sym = ''
    if visualize:
        if epoch == 0: pass
        elif values[1] > prev_val[0] and values[2] > prev_val[1]:  sym = ' △ △'
        elif values[1] > prev_val[0] and values[2] < prev_val[1]:  sym = ' △ ▼'            
        elif values[1] < prev_val[0] and values[2] > prev_val[1]:  sym = ' ▼ △'            
        elif values[1] < prev_val[0] and values[2] < prev_val[1]:  sym = ' ▼ ▼'
    print(layout.format(*values) + sym)

class IterateBatch():
    def __init__(self, dl):
        self.idx = 0
        self.dl = dl
        self.iter = iter(dl)

    def __iter__(self): return self

    def next(self):
        res = next(self.iter)
        self.idx += 1
        if self.idx == len(self.dl):
            self.iter = iter(self.dl)
            self.idx = 0
        return res

def validate_next(stepper, metrics, val_iter):
    r"""Computes the loss on the next minibatch of the validation set.
    """
    stepper.reset(False)
    with no_grad_context():
        (*x,y) = val_iter.next()
        preds,l = stepper.evaluate(VV(x), VV(y))
        res = [delistify(to_np(l))]
        res += [f(datafy(preds), datafy(y)) for f in metrics]
    stepper.reset(True)
    return res

def batch_size(x, seq_first=False):
    if isinstance(x, (list, tuple)): x = x[0]
    return x.shape[1 if seq_first else 0]

def validate(stepper, dl, metrics, seq_first=False):
    batch_count, loss, res = [[] for _ in range(3)]
    stepper.reset(False)
    with no_grad_context():
        for (*x, y) in iter(dl):
            y = VV(y)
            preds, l = stepper.evaluate(VV(x), y)
            batch_count.append(batch_size(x, seq_first=seq_first))
            loss.append(to_np(l))
            res.append([f(datafy(preds), datafy(y)) for f in metrics])
    return [np.average(loss, 0, weights=batch_count)] \
            + list(np.average(np.stack(res), 0, weights=batch_count))

def get_prediction(x):
    if isinstance(x, (list, tuple)): x = x[0]
    return x.data

def predict(m, dl):
    preda, _ = predict_with_targs_(m, dl)
    return np.concatenate(preda)

def predict_batch(m, x):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    return m(VV(x))

def predict_with_targs_(m, dl):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    res = []
    for *x, y in iter(dl): 
        res.append([get_prediction(to_np(m(*VV(x)))), to_np(y)])
    return zip(*res)

def predict_with_targs(m, dl):
    preda, targa = predict_with_targs_(m, dl)
    return np.concatenate(preda), np.concatenate(targa)