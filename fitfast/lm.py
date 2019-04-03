from .imports import *
from .encoders import AWDLSTMEncoder, QRNNEncoder
from .tricks.dropout import LockedDropout
from .text import SequentialRNN

class LinearDecoder(nn.Module):
    initrange = 0.1
    def __init__(self, no, nh, dropout, tie_encoder=None, bias=False):
        super().__init__()
        self.decoder = nn.Linear(nh, no, bias=bias)
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


class LanguageModeler(object):
    r""" Abstract base class for encode/decoder sequence modelers. 
    """
    def __init__(self): pass
    def get_language_model(self): pass


class AWDLSTMModeler(LanguageModeler):
    def get_language_model(self, n_tokens, em, nh, nl, pad_token, drop_i=.6, 
                           drop_e=.1, drop_h=.2, drop_d=.5, w_drop=.4, 
                           tie_weights=True, bias=False):
        r"""
        Returns a SequentialRNN language model.

        A RNNEncoder layer is instantiated using the parameters provided. This 
        is followed by the creation of a LinearDecoder layer.

        Also by default (i.e., tie_weights=True), the embedding matrix used in 
        the RNNEncoder is used to  instantiate the weights for the LinearDecoder 
        layer.

        Arguments:
            n_tokens (int): number of unique vocabulary words (or tokens) in the 
                            source dataset.
            em (int): the embedding size to use to encode each token.
            nh (int): number of hidden activation per LSTM layer.
            nl (int): number of LSTM layers to use in the architecture.
            pad_token (int): the int value used for padding text.
            drop_i (float): dropout to apply to the input layer.
            drop_e (float): dropout to apply to the embedding layer.
            drop_h (float): dropout to apply to the activations going from one 
                    LSTM layer to another.
            drop_d (float): dropout to apply to the linear decoder.
            w_drop (float): dropout used for a LSTM's internal (or hidden) 
                    recurrent weights.
            tie_weights (bool): decide if the weights of the embedding matrix in 
                    the RNN encoder should be tied to the weights of the 
                    LinearDecoder layer.
            bias (bool): Decoder has bias if True.
        
        Returns: A SequentialRNN model
        """
        rnn_encoder = AWDLSTMEncoder(n_tokens, em, nh=nh, nl=nl, 
                                     pad_token=pad_token, drop_i=drop_i, 
                                     drop_e=drop_e, drop_h=drop_h, w_drop=w_drop)
        tie_encoder = rnn_encoder.encoder if tie_weights else None
        decoder = LinearDecoder(n_tokens, em, drop_d, tie_encoder=tie_encoder, 
                                bias=bias)
        return SequentialRNN(rnn_encoder, decoder)


class QRNNModeler(LanguageModeler):
    def get_language_model(self, n_tokens, emb_sz, n_hid, n_layers, pad_token, 
                           dropout=0.4, dropouth=0.3, dropouti=0.5, dropoute=0.1,
                           wdrop=0.5, tie_weights=True, bias=False):
        r"""
        Returns a SequentialRNN language model.

        A RNNEncoder layer is instantiated using the parameters provided. This 
        is followed by the creation of a LinearDecoder layer.

        Also by default (i.e., tie_weights=True), the embedding matrix used in 
        the RNNEncoder is used to  instantiate the weights for the LinearDecoder 
        layer.

        Arguments:
            n_tokens (int): number of unique vocabulary words (or tokens) in the 
                            source dataset.
            emb_sz (int): the embedding size to use to encode each token.
            n_hid (int): number of hidden activation per LSTM layer.
            n_layers (int): number of LSTM layers to use in the architecture.
            pad_token (int): the int value used for padding text.
            dropouth (float): dropout to apply to the activations going from one 
                    LSTM layer to another.
            dropouti (float): dropout to apply to the input layer.
            dropoute (float): dropout to apply to the embedding layer.
            wdrop (float): dropout used for a LSTM's internal (or hidden) 
                    recurrent weights.
            tie_weights (bool): decide if the weights of the embedding matrix in 
                    the RNN encoder should be tied to the weights of the 
                    LinearDecoder layer.
            bias (bool): Decoder has bias if True.
        
        Returns: A SequentialRNN model
        """
        rnn_encoder = QRNNEncoder(n_tokens, emb_sz, n_hid=n_hid, 
                                     n_layers=n_layers, pad_token=pad_token, 
                                     dropouth=dropouth, dropouti=dropouti, 
                                     dropoute=dropoute, wdrop=wdrop)
        tie_encoder = rnn_encoder.encoder if tie_weights else None
        decoder = LinearDecoder(n_tokens, emb_sz, dropout, 
                                tie_encoder=tie_encoder, bias=bias)
        return SequentialRNN(rnn_encoder, decoder)