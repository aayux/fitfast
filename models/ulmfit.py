import warnings
from ..imports import *
from ..torch_imports import *
from ..dropout import LockedDropout, WeightDrop, EmbeddingDropout
from ..fit import Stepper
from ..core import set_grad_enabled

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')

def seq2seq_regularizer(output, xtra, loss, alpha=0, beta=0):
    hs, dropped_hs = xtra
    # AR: Activation Regularization
    if alpha:
        loss = loss + (alpha * dropped_hs[-1].pow(2).mean()).sum()
    # TAR: Temporal Activation Regularization (slowness)
    if beta:
        h = hs[-1]
        if len(h) > 1: 
            loss = loss + (beta * (h[1:] - h[:-1]).pow(2).mean()).sum()
    return loss


def repackage_var(h):
    r""" Wraps the input in new Variables to detach them from their history.
    """
    if IS_TORCH_04: 
        return h.detach() if type(h) == torch.Tensor \
                          else tuple(repackage_var(v) for v in h)
    else: return Variable(h.data) if type(h) == Variable \
                                  else tuple(repackage_var(v) for v in h)


class RNNEncoder(nn.Module):

    r"""
    A custom RNN encoder network that uses:
        - an embedding matrix to encode input,
        - a stack of LSTM or QRNN layers to drive the network, and
        - variational dropouts in the embedding and LSTM/QRNN layers

    The architecture for this network was inspired by the work done in 
    "Regularizing and Optimizing LSTM Language Models".
    (https://arxiv.org/pdf/1708.02182.pdf)
    """

    initrange = 0.1

    def __init__(self, ntoken, emb_sz, n_hid, n_layers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5, 
                 qrnn=False):
        r""" 
        Default constructor for the RNNEncoder class

        Arguments:
            bs (int): batch size of input data
            ntoken (int): number of vocabulary (or tokens) in the source dataset
            emb_sz (int): the embedding size to use to encode each token
            n_hid (int): number of hidden activation per LSTM layer
            n_layers (int): number of LSTM layers to use in the architecture
            pad_token (int): the int value used for padding text.
            dropouth (float): dropout to apply to the activations going from one 
                              LSTM layer to another
            dropouti (float): dropout to apply to the input layer.
            dropoute (float): dropout to apply to the embedding layer.
            wdrop (float): dropout used for a LSTM's internal (or hidden) 
                           recurrent weights.

        Returns: None
          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs = 1
        self.qrnn = qrnn
        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        if self.qrnn:
            # Using QRNN requires cupy: https://github.com/cupy/cupy
            from .torchqrnn.qrnn import QRNNLayer
            self.rnns = [QRNNLayer(emb_sz if l == 0 else n_hid, 
                        (n_hid if l != n_layers - 1 else emb_sz) // self.ndir, 
                        save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, 
                        output_gate=True) for l in range(n_layers)]
            if wdrop:
                for rnn in self.rnns:
                    rnn.linear = WeightDrop(rnn.linear, wdrop, weights=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, 
                        (n_hid if l != n_layers - 1 else emb_sz) // self.ndir, 1, 
                        bidirectional=bidir) for l in range(n_layers)]
            if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.emb_sz = emb_sz
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.dropoute = dropoute
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) \
                                        for l in range(n_layers)])

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
                    dropout=self.dropoute if self.training else 0)
            embedding = self.dropouti(embedding)
            raw_output = embedding
            new_hidden, raw_outputs, outputs = [], [], []
            
            for l, (rnn, drop) in enumerate(zip(self.rnns, self.dropouths)):
                current_input = raw_output
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    raw_output, new_h = rnn(raw_output, self.hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.n_layers - 1: raw_output = drop(raw_output)
                outputs.append(raw_output)

            self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def one_hidden(self, l):
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.ndir
        if IS_TORCH_04: 
            return Variable(self.weights.new(self.ndir, self.bs, nh).zero_())
        else: 
            return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), 
                            volatile=not self.training)

    def reset(self):
        if self.qrnn: [r.reset() for r in self.rnns]
        self.weights = next(self.parameters()).data
        if self.qrnn: self.hidden = [self.one_hidden(l) \
                                     for l in range(self.n_layers)]
        else: self.hidden = [(self.one_hidden(l), self.one_hidden(l)) \
                              for l in range(self.n_layers)]


class MultiBatchRNN(RNNEncoder):
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

class LinearDecoder(nn.Module):
    initrange = 0.1
    def __init__(self, n_out, n_hid, dropout, tie_encoder=None, bias=False):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout = LockedDropout(dropout)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.dropout(outputs[-1])
        decoded = self.decoder(output.view(output.size(0) * output.size(1), 
                               output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        return result, raw_outputs, outputs

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


class SequentialRNN(nn.Sequential):
    r"""
    The SequentialRNN layer is the native torch's Sequential wrapper that puts 
    the Encoder and Decoder modules sequentially in the model.
    """
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()


def rnn_language_model(n_tok, emb_sz, n_hid, n_layers, pad_token, dropout=0.4, 
                       dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, 
                       tie_weights=True, qrnn=False, bias=False):
    r"""
    Returns a SequentialRNN language model.

    A RNNEncoder layer is instantiated using the parameters provided. This is 
    followed by the creation of a LinearDecoder layer.

    Also by default (i.e., tie_weights=True), the embedding matrix used in the 
    RNNEncoder is used to  instantiate the weights for the LinearDecoder layer.

    Arguments:
        n_tok (int): number of unique vocabulary words (or tokens) in the source
                     dataset.
        emb_sz (int): the embedding size to use to encode each token.
        n_hid (int): number of hidden activation per LSTM layer.
        n_layers (int): number of LSTM layers to use in the architecture.
        pad_token (int): the int value used for padding text.
        dropouth (float): dropout to apply to the activations going from one 
                          LSTM layer to another.
        dropouti (float): dropout to apply to the input layer.
        dropoute (float): dropout to apply to the embedding layer.
        wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent 
                       weights.
        tie_weights (bool): decide if the weights of the embedding matrix in the
                            RNN encoder should be tied to the weights of the 
                            LinearDecoder layer.
        qrnn (bool): model is composed of LSTMS if False or QRNNs if True.
        bias (bool): Decoder has bias if True.
    
    Returns: A SequentialRNN model
    """
    rnn_encoder = RNNEncoder(n_tok, emb_sz, n_hid=n_hid, n_layers=n_layers, 
                         pad_token=pad_token, dropouth=dropouth, 
                         dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, 
                         qrnn=qrnn)
    tie_encoder = rnn_encoder.encoder if tie_weights else None
    decoder = LinearDecoder(n_tok, emb_sz, dropout, tie_encoder=tie_encoder, 
                            bias=bias)
    return SequentialRNN(rnn_encoder, decoder)

def rnn_linear_classifier(bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, 
                          pad_token, layers, drops, bidir=False, dropouth=0.3, 
                          dropouti=0.5, dropoute=0.1, wdrop=0.5, qrnn=False):
    encoder = MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, 
                            pad_token=pad_token, bidir=bidir, dropouth=dropouth,
                            dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, 
                            qrnn=qrnn)
    decoder = PoolingLinearClassifier(layers, drops)
    return SequentialRNN(encoder, decoder)

################################################################################
# TRASNFORMER DECODER NETWORK
# Based on Transformer Network introduced in "Attention Is All You Need"
# https://arxiv.org/abs/1706.03762

# class PositionwiseFeedForward(nn.Module):
#     r""" A two-feed-forward-layer module
#     """

#     def __init__(self, d_hid, d_inner_hid, dropout=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
#         self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
#         self.layer_norm = LayerNormalization(d_hid)
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         residual = x
#         output = self.relu(self.w_1(x.transpose(1, 2)))
#         output = self.w_2(output).transpose(2, 1)
#         output = self.dropout(output)
# return self.layer_norm(output + residual)

# class MultiHeadAttention(nn.Module):
#     r""" Multi-Head Attention module.
#     """

#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()

#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
#         self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
#         self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

#         self.attention = ScaledDotProductAttention(d_model)
#         self.layer_norm = LayerNormalization(d_model)
#         self.proj = Linear(n_head*d_v, d_model)

#         self.dropout = nn.Dropout(dropout)

#         init.xavier_normal(self.w_qs)
#         init.xavier_normal(self.w_ks)
#         init.xavier_normal(self.w_vs)

#     def forward(self, q, k, v, attn_mask=None):

#         d_k, d_v = self.d_k, self.d_v
#         n_head = self.n_head

#         residual = q

#         mb_size, len_q, d_model = q.size()
#         mb_size, len_k, d_model = k.size()
#         mb_size, len_v, d_model = v.size()

#         # treat as a (n_head) size batch
#         q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
#         k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
#         v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

#         # treat the result as a (n_head * mb_size) size batch
#         q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
#         k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
#         v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

#         # perform attention, result size = (n_head * mb_size) x len_q x d_v
#         outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))

#         # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
#         outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1) 

#         # project back to residual size
#         outputs = self.proj(outputs)
#         outputs = self.dropout(outputs)

#       return self.layer_norm(outputs + residual), attns

# class DecoderLayer(nn.Module):
#     ''' Compose with two layers '''

#     def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
#         super(EncoderLayer, self).__init__()
#         self.slf_attn = MultiHeadAttention(
#             n_head, d_model, d_k, d_v, dropout=dropout)
#         self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, 
#                                                   dropout=dropout)

#     def forward(self, enc_input, slf_attn_mask=None):
#         enc_output, enc_slf_attn = self.slf_attn(
#             enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
#         enc_output = self.pos_ffn(enc_output)
#       return enc_output, enc_slf_attn



# class TransformerDecoderClassifier(nn.Module):
#     def __init__(self, layers, drops):
#         super().__init__()
#         self.attentionlayer = nn.ModuleList([
#                   DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, 
#                                                     dropout=dropout)])

#     def pool(self, x, bs, is_max):
#         f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
#         return f(x.permute(1, 2, 0), (1, )).view(bs, -1)

#     def forward(self, input):
#         raw_outputs, outputs, targets = input
#         output = outputs[-1]
#         sl, bs, _ = output.size()
#         avgpool = self.pool(output, bs, False)
#         maxpool = self.pool(output, bs, True)
#         x = torch.cat([output[-1], maxpool, avgpool], 1)
        
#         for layer in self.attentionlayer:
#             x = l(x)
        
#         return x, raw_outputs, outputs