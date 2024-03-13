from .basemodel import BaseModel
from abc import abstractmethod

class UpDownModel(BaseModel):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def up(self, **kwargs): ...

    @abstractmethod
    def down(self, **kwargs): ...

    @abstractmethod
    def fuse(self, **kwargs): ...

class UpModel(BaseModel):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def up(self, **kwargs): ...

    @abstractmethod
    def fuse(self, **kwargs): ...

class DownModel(BaseModel):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def down(self, **kwargs): ...