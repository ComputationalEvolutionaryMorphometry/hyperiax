from .basemodel import BaseModel
from abc import abstractmethod
from ..mcmc import ParameterStore

class UpdateModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update(self, *,parent_value,children_values,node_value): ...