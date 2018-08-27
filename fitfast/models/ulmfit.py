import warnings
from ..imports import *

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')


class MultiBatchRNN(AWDLSTMEncoder):
    def __init__(self, bptt, max_seq, *args, **kwargs):
        self.max_seq,self.bptt = max_seq,bptt
        super().__init__(*args, **kwargs)

    def concat(self, arrs):
        return [torch.cat([l[si] for l in arrs]) for si in range(len(arrs[0]))]

    def forward(self, input):
        sl, bs = input.size()
        for l in self.hidden:
            for h in l: h.data.zero_()
        raw_outputs, outputs = [], []
        for i in range(0, sl, self.bptt):
            r, o = super().forward(input[i: min(i + self.bptt, sl)])
            if i > (sl - self.max_seq):
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs), self.concat(outputs)


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
        sl,bs,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x, raw_outputs, outputs


def rnn_linear_classifier(bptt, max_seq, n_class, n_tokens, emb_sz, n_hid, 
                          n_layers, pad_token, layers, drops, bidir=False, 
                          dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, 
                          qrnn=False):
    encoder = MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, 
                            pad_token=pad_token, bidir=bidir, dropouth=dropouth,
                            dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, 
                            qrnn=qrnn)
    decoder = PoolingLinearClassifier(layers, drops)
    return SequentialRNN(encoder, decoder)