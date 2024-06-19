from .basemodel import BaseModel
from abc import abstractmethod
from .utils import map_reduction

class UpDownModel(BaseModel):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def up(self, **kwargs): ...

    @abstractmethod
    def down(self, **kwargs): ...

    @abstractmethod
    def transform(self, **kwargs): ...

class UpModel(BaseModel):
    def __init__(self, reductions) -> None:
        #self.produces = produces
        self.reductions = {k: map_reduction(v) for k,v in reductions.items()}

    @abstractmethod
    def transform(self, **kwargs): ...

    @abstractmethod
    def up(self, **kwargs): ...

class DownModel(BaseModel):
    def __init__(self) -> None: ...

    @abstractmethod
    def down(self, **kwargs): ...

class FuseModel(BaseModel):
    def __init__(self): ...

    @abstractmethod
    def fuse(self, **kwargs): ...