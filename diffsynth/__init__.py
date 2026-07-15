from .core import *
from .core.loader import load_state_dict
from .utils.data import save_video


class ModelManager:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "SlotMem.diffsynth.ModelManager is a compatibility placeholder "
            "and should not be instantiated in this runtime."
        )


class SVIVideoPipeline:
    @classmethod
    def from_model_manager(cls, *args, **kwargs):
        raise RuntimeError(
            "SlotMem.diffsynth.SVIVideoPipeline is a compatibility placeholder "
            "and should not be instantiated in this runtime."
        )
