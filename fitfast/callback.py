from .imports import *

class Callback:
    r""" An abstract class that all callback classes inherit from.
    """
    def on_train_begin(self): pass
    
    def on_batch_begin(self): pass
    
    def on_phase_begin(self): pass
    
    def on_epoch_end(self, metrics): pass
    
    def on_phase_end(self): pass
    
    def on_batch_end(self, metrics): pass
    
    def on_train_end(self): pass