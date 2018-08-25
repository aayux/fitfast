from .torch_imports import *
from .core import *
from functools import wraps
import torch.nn.functional as F
from torch.autograd import Variable

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')

def dropout_mask(x, sz, dropout):
    r"""
    Applies a dropout mask whose size is determined by passed argument 'sz'.
    
    Args:
        x (nn.Variable): A torch Variable object
        sz (tuple(int, int, int)): The expected size of the new tensor
        dropout (float): The dropout fraction to apply

    This method uses the bernoulli distribution to decide which activations to 
    keep. Additionally, the sampled activations is rescaled is using the factor 
    1 / (1 - dropout).

    NOTE: bernoulli_ may be deprecated in the future.

    """
    return x.new(*sz).bernoulli_(1 - dropout) / (1 - dropout)

class EmbeddingDropout(nn.Module):

    r"""
    Applies dropout in the embedding layer by zeroing out some elements of the 
    embedding vector by utilising ses the custom dropout_mask layer.

    Args:
        embed (torch.nn.Embedding): An embedding torch layer
        words (torch.nn.Variable): A torch variable
        dropout (float): dropout fraction to apply to the embedding weights
        scale (float): additional scaling to apply to the modified embedding 
                       weights

    Returns: Tensor of size [batch_size x seq_length x embedding_size]
    """

    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, words, dropout=0.1, scale=None):
        if dropout:
            size = (self.embed.weight.size(0), 1)
            mask = Variable(dropout_mask(self.embed.weight.data, size, dropout))
            masked_embed_weight = mask * self.embed.weight
        else: masked_embed_weight = self.embed.weight

        if scale: masked_embed_weight = scale * masked_embed_weight

        padding_idx = self.embed.padding_idx
        if padding_idx is None: padding_idx = -1

        
        if IS_TORCH_04:
            x = F.embedding(words,
                masked_embed_weight, padding_idx, self.embed.max_norm,
                self.embed.norm_type, self.embed.scale_grad_by_freq, 
                self.embed.sparse)
        else:
            x = self.embed._backend.Embedding.apply(words,
                masked_embed_weight, padding_idx, self.embed.max_norm,
                self.embed.norm_type, self.embed.scale_grad_by_freq, 
                self.embed.sparse)

        return x

class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or not self.p: return x
        mask = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return Variable(mask, requires_grad=False) * x

class WeightDrop(torch.nn.Module):
    r"""
    A custom torch layer that serves as a wrapper on another torch layer.
    Primarily responsible for updating the weights in the wrapped module based
    on a specified dropout.
    """
    def __init__(self, module, dropout, weights=['weight_hh_l0']):
        r""" 
        Default constructor for the WeightDrop module

        Args:
            module (torch.nn.Module): A pytorch layer being wrapped
            dropout (float): a dropout value to apply
            weights (list(str)): the parameters of the wrapped module which 
                                 should be fractionally dropped.
        """
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        r""" 
        For each string defined in self.weights, the corresponding
        attribute in the wrapped module is referenced, then deleted, 
        and subsequently registered as a new parameter with a slightly 
        modified name.
        """
        if isinstance(self.module, torch.nn.RNNBase): 
            self.module.flatten_parameters = no_op
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))


    def _setweights(self):
        r""" 
        Uses pytorch's built-in dropout function to apply dropout to the 
        parameters of the wrapped module.
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=self.dropout, 
                                            training=self.training)
            if hasattr(self.module, name_w):
                delattr(self.module, name_w)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        r""" 
        Update weights and delegate the propagation of the tensor to the wrapped
        module's forward method.

        Returns: 
            Tensor obtained by running the forward method on the wrapped module.
        """
        self._setweights()
        return self.module.forward(*args)