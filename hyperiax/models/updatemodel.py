from .utils import map_reduction
from .basemodel import BaseModel
from abc import abstractmethod
from ..mcmc import ParameterStore

class UpdateModel(BaseModel):
    def __init__(self, reductions) -> None:
        #self.produces = produces
        self.reductions = {k: map_reduction(v) for k,v in reductions.items()}

    @abstractmethod
    def update(self, **kwargs): ...

    @abstractmethod
    def up(self, **kwargs): ...
