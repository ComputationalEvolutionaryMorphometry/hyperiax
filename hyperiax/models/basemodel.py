from abc import ABC, abstractmethod
from .utils import map_reduction

class BaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._set_keys()
    
    @abstractmethod
    def _set_keys(self): ...

class ReducerModel(BaseModel):
    def __init__(self, reductions, up_preserves = []) -> None:
        super().__init__()
        self.reductions = {k: map_reduction(v) for k,v in reductions.items()}
        self.up_preserves = up_preserves
