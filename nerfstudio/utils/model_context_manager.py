from contextlib import ContextDecorator
from nerfstudio.models.base_model import Model 



class eval_context(ContextDecorator):
    """context for nerf model evaluation"""
    def __init__(self, model: Model):
        self.prev = False
        self._model = model
        
    
    def __enter__(self):
        self._model.setup_eval_ctx()

    def __exit__(self, *exc):
        self._model.clear_eval_ctx()