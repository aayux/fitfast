from .imports import *
from .utils.core import *
from .utils.extras import *


def optimizer_params(parameters, lr, wd):
    return {'params': chain_params(parameters), 
            'lr': lr, 
            'weight_decay': wd}

class LayerOptimizer(object):
    def __init__(self, optimizer, layer_groups, lrs, wds=None):
        if not isinstance(layer_groups, (list, tuple)): 
            layer_groups=[layer_groups]
        lrs = listify(lrs, layer_groups)
        if wds is None: wds = 0.
        wds = listify(wds, layer_groups)
        self.layer_groups = layer_groups
        self.lrs = lrs
        self.wds = wds
        self.opt = optimizer(self.optimizer_params())

    def optimizer_params(self):
        assert len(self.layer_groups) == len(self.lrs), (f'size mismatch, '
            f'expected {len(self.layer_groups)} lrs, but got {len(self.lrs)}')
        assert len(self.layer_groups) == len(self.wds), (f'size mismatch, '
            f'expected {len(self.layer_groups)} wds, but got {len(self.wds)}')
        params = list(zip(self.layer_groups, self.lrs, self.wds))
        return [optimizer_params(*p) for p in params]

    @property
    def lr(self): return self.lrs[-1]

    @property
    def momentum(self):
        if 'betas' in self.opt.param_groups[0]:
            return self.opt.param_groups[0]['betas'][0]
        else:
            return self.opt.param_groups[0]['momentum']

    def set_lrs(self, lrs):
        lrs = listify(lrs, self.layer_groups)
        set_lrs(self.opt, lrs)
        self.lrs = lrs

    def set_wds_out(self, wds):
        wds = listify(wds, self.layer_groups)
        set_wds_out(self.opt, wds)
        set_wds(self.opt, [0] * len(self.layer_groups))
        self.wds = wds

    def set_wds(self, wds):
        wds = listify(wds, self.layer_groups)
        set_wds(self.opt, wds)
        set_wds_out(self.opt, [0] * len(self.layer_groups))
        self.wds = dwds
    
    def set_momentum(self, momentum):
        if 'betas' in self.opt.param_groups[0]:
            for pg in self.opt.param_groups: 
                pg['betas'] = (momentum, pg['betas'][1])
        else:
            for pg in self.opt.param_groups: pg['momentum'] = momentum
    
    def set_beta(self, beta):
        if 'betas' in self.opt.param_groups[0]:
            for pg in self.opt.param_groups: pg['betas'] = (pg['betas'][0],beta)
        elif 'alpha' in self.opt.param_groups[0]:
            for pg in self.opt.param_groups: pg['alpha'] = beta

    def set_optimizer(self, optimizer):
        if type(self.opt) != type(optimizer(self.optimizer_params())):
            self.opt = optimizer(self.optimizer_params())

def _zip_strict(l, r):
    assert len(l) == len(r), \
           f'size mismatch, expected {len(l)} r, but got {len(r)} r'
    return zip(l, r)

def set_lrs(opt, lrs):
    lrs = listify(lrs, opt.param_groups)
    for pg, lr in _zip_strict(opt.param_groups,lrs): pg['lr'] = lr

def set_wds_out(opt, wds):
    wds = listify(wds, opt.param_groups)
    assert len(opt.param_groups) == len(wds), \
    f'size mismatch, expected {len(opt.param_groups)} wds, but got {len(wds)}'
    
    for pg,wd in _zip_strict(opt.param_groups,wds): pg['wd'] = wd

def set_wds(opt, wds):
    wds = listify(wds, opt.param_groups)
    assert len(opt.param_groups) == len(wds), \
    f'size mismatch, expected {len(opt.param_groups)} wds, but got {len(wds)}'
    for pg,wd in _zip_strict(opt.param_groups,wds): pg['weight_decay'] = wd
