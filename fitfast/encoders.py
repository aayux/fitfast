from .imports import *
from .utils.core import set_grad_enabled
from .tricks.dropout import LockedDropout, WeightDrop, EmbeddingDropout

def repackage_var(h):
    r""" Wraps the input in new Variables to detach them from their history.
    """
    if IS_TORCH_04: 
        return h.detach() if type(h) == torch.Tensor \
                          else tuple(repackage_var(v) for v in h)
    else: return Variable(h.data) if type(h) == Variable \
                                  else tuple(repackage_var(v) for v in h)

class AWDLSTMEncoder(nn.Module):
    r"""
    A custom RNN encoder network that uses:
        - an embedding matrix to encode input,
        - a stack of LSTM or QRNN layers to drive the network, and
        - variational dropouts in the embedding and LSTM/QRNN layers

    The architecture for this network was inspired by the work done in 
    "Regularizing and Optimizing LSTM Language Models"
    arxiv.org/abs/1708.02182
    """

    initrange = 0.1

    def __init__(self, n_tokens, em, nh, nl, pad_token, drop_i, drop_e, drop_h, 
                 w_drop, bidir=False):
        r""" 
        Default constructor for the RNNEncoder class

        Arguments:
            bs (int): batch size of input data
            n_tokens (int): number of vocabulary (or tokens) in the source dataset
            emb (int): the embedding size to use to encode each token
            nh (int): number of hidden activation per LSTM layer
            nl (int): number of LSTM layers to use in the architecture
            pad_token (int): the int value used for padding text.
            drop_i (float): dropout to apply to the input layer.
            drop_e (float): dropout to apply to the embedding layer.
            drop_h (float): dropout to apply to the activations going from one 
                              LSTM layer to another
            w_drop (float): dropout used for a LSTM's internal (or hidden) 
                           recurrent weights.

        Returns: None
          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs = 1
        self.encoder = nn.Embedding(n_tokens, em, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.rnns = [nn.LSTM(em if l == 0 else nh, 
                    (nh if l != nl - 1 else em) // self.ndir, 1, 
                    bidirectional=bidir) for l in range(nl)]
        if w_drop: self.rnns = [WeightDrop(rnn, w_drop) for rnn in self.rnns]
        
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.em = em
        self.nh = nh
        self.nl = nl
        self.drop_e = drop_e
        self.drop_i = LockedDropout(drop_i)
        self.drop_hs = nn.ModuleList([LockedDropout(drop_h) for l in range(nl)])

    def forward(self, input):
        r"""
        Invoked during the forward propagation of the RNNEncoder module.
        
        Arguments:
            input (Tensor): input of shape [sentence length x batch_size]

        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors 
            evaluated from each RNN layer without using dropouth, list of 
            tensors evaluated from each RNN layer using dropouth,
        """
        sl, bs = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()
        with set_grad_enabled(self.training):
            embedding = self.encoder_with_dropout(input, 
                            dropout=self.drop_e if self.training else 0)
            embedding = self.drop_i(embedding)
            raw_output = embedding
            new_hidden, raw_outputs, outputs = [], [], []
            
            for l, (rnn, drop) in enumerate(zip(self.rnns, self.drop_hs)):
                current_input = raw_output
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    raw_output, new_h = rnn(raw_output, self.hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.nl - 1: raw_output = drop(raw_output)
                outputs.append(raw_output)

            self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def one_hidden(self, l):
        nh = (self.nh if l != self.nl - 1 else self.em) // self.ndir
        if IS_TORCH_04: 
            return Variable(self.weights.new(self.ndir, self.bs, nh).zero_())
        else: 
            return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), 
                            volatile=not self.training)

    def reset(self):
        self.weights = next(self.parameters()).data
        self.hidden = [(self.one_hidden(l), self.one_hidden(l)) \
                        for l in range(self.nl)]

class QRNNEncoder(nn.Module):
    r"""
    A custom RNN encoder network that uses:
        - an embedding matrix to encode input,
        - a stack of LSTM or QRNN layers to drive the network, and
        - variational dropouts in the embedding and QRNN layers

    The architecture for this network was inspired by the work done in 
    "Quasi-Recurrent Neural Network"
    arxiv.org/abs/1611.01576
    """

    initrange = 0.1

    def __init__(self, n_tokens, em, nh, nl, pad_token, drop_i, drop_e, drop_h, 
                 w_drop, bidir=False):
        r""" 
        Default constructor for the RNNEncoder class

        Arguments:
            bs (int): batch size of input data
            n_tokens (int): number of vocabulary (or tokens) in the source dataset
            emb (int): the embedding size to use to encode each token
            nh (int): number of hidden activation per LSTM layer
            nl (int): number of LSTM layers to use in the architecture
            pad_token (int): the int value used for padding text.
            drop_i (float): dropout to apply to the input layer.
            drop_e (float): dropout to apply to the embedding layer.
            drop_h (float): dropout to apply to the activations going from one 
                              LSTM layer to another
            w_drop (float): dropout used for a LSTM's internal (or hidden) 
                           recurrent weights.

        Returns: None

          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs = 1
        self.encoder = nn.Embedding(n_tokens, em, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)

        # Using QRNN requires cupy: github.com/cupy/cupy
        from .torchqrnn.qrnn import QRNNLayer
        self.rnns = [QRNNLayer(em if l == 0 else nh, 
                    (nh if l != n_layers - 1 else em) // self.ndir, 
                    save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, 
                    output_gate=True) for l in range(nl)]
        if w_drop:
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, w_drop, weights=['weight'])

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.em = em
        self.nh = nh
        self.nl = nl
        self.drop_e = drop_e
        self.drop_i = LockedDropout(drop_i)
        self.drop_hs = nn.ModuleList([LockedDropout(drop_h) for l in range(nl)])

    def forward(self, input):
        r"""
        Invoked during the forward propagation of the RNNEncoder module.
        
        Arguments:
            input (Tensor): input of shape [sentence length x batch_size]

        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors 
            evaluated from each RNN layer without using dropouth, list of 
            tensors evaluated from each RNN layer using dropouth,
        """
        sl, bs = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()
        with set_grad_enabled(self.training):
            embedding = self.encoder_with_dropout(input, 
                    dropout=self.drop_e if self.training else 0)
            embedding = self.drop_i(embedding)
            raw_output = embedding
            new_hidden, raw_outputs, outputs = [], [], []
            
            for l, (rnn, drop) in enumerate(zip(self.rnns, self.drop_hs)):
                current_input = raw_output
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    raw_output, new_h = rnn(raw_output, self.hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.nl - 1: raw_output = drop(raw_output)
                outputs.append(raw_output)

            self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def one_hidden(self, l):
        nh = (self.nh if l != self.nl - 1 else self.em) // self.ndir
        if IS_TORCH_04: 
            return Variable(self.weights.new(self.ndir, self.bs, nh).zero_())
        else: 
            return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), 
                            volatile=not self.training)

    def reset(self):
        [r.reset() for r in self.rnns]
        self.weights = next(self.parameters()).data
        self.hidden = [self.one_hidden(l) for l in range(self.nl)]

class TransformerEncoder(nn.Module):
    def __init__(self): pass
    def forward(self): pass