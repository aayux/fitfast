import warnings
from ..imports import *
from ..encoders import multi_batch_rnn

class LinearBlock(nn.Module):
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)

    def forward(self, x): return self.lin(self.drop(self.bn(x)))


class PoolingLinearClassifier(nn.Module):
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) \
                        for i in range(len(layers) - 1)])

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(bs, -1)

    def forward(self, input):
        raw_outputs, outputs = input
        output = outputs[-1]
        sl, bs, _ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x, raw_outputs, outputs


class Linear(object):
    def get_classifier(self, en, bptt, dims, max_seq, n_tokens, pad_token, em, 
                        nh, nl, drop_i=.6, drop_e=.5, drop_h=.3, drop_d=(.2, .1),
                        w_drop=.4, bidir=False):
        _encoder_init = multi_batch_rnn(en)
        encoder = _encoder_init(bptt, max_seq, n_tokens, em, nh, nl, 
                                pad_token=pad_token, bidir=bidir, drop_i=drop_i, 
                                drop_e=drop_e, drop_h=drop_h, w_drop=w_drop)
        decoder = PoolingLinearClassifier(layers, drop_d)
        return SequentialRNN(encoder, decoder)