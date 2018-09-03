from .utils.core import *

class BaseModel():
    r""" Base class for extending all model classes.
    """
    def __init__(self, model, name='base'): 
        self.model = model
        self.name = name
    
    def get_layer_groups(self, do_fc=False): return children(self.model)