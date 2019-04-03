from ..imports import *

def seq2seq_regularizer(output, extra, loss, ar=0, tar=0):
    hs, dropped_hs = extra
    # AR: Activation Regularization
    if ar:
        loss = loss + (ar * dropped_hs[-1].pow(2).mean()).sum()
    # TAR: Temporal Activation Regularization (slowness)
    if tar:
        h = hs[-1]
        if len(h) > 1: 
            loss = loss + (tar * (h[1:] - h[:-1]).pow(2).mean()).sum()
    return loss