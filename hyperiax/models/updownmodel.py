from .basemodel import BaseModel
from abc import abstractmethod
from .utils import map_reduction
from inspect import getfullargspec

class UpModel(BaseModel):
    def __init__(self, reductions) -> None:
        #self.produces = produces
        self.reductions = {k: map_reduction(v) for k,v in reductions.items()}
        self._set_up_keys()

    @abstractmethod
    def transform(self, **kwargs): ...

    @abstractmethod
    def up(self, **kwargs): ...

    def _set_up_keys(self):
        up_arg_spec = getfullargspec(self.up)
        upkeys = up_arg_spec.args
        self.up_keys = [k for k in upkeys if k != 'self']

        transform_arg_spec = getfullargspec(self.transform)
        keys = transform_arg_spec.args
        transform_child_keys = [k.removeprefix('child_') for k in keys if k.startswith('child_')]
        transform_parent_keys = [k for k in keys if not k.startswith('child_') and k != 'self']

        self.transform_child_keys = transform_child_keys
        self.transform_parent_keys = transform_parent_keys

class DownModel(BaseModel):
    def __init__(self) -> None:
        self._set_down_keys()

    @abstractmethod
    def down(self, **kwargs): ...

    def _set_down_keys(self):
        arg_spec = getfullargspec(self.down)

        keys = arg_spec.args # TODO: we should remove things like key, up_msg, params

        down_parent_keys = [k.removeprefix('parent_') for k in keys if k.startswith('parent_')]
        down_child_keys = [k for k in keys if not k.startswith('parent_') and k != 'self']

        self.down_parent_keys = down_parent_keys
        self.down_child_keys = down_child_keys

class FuseModel(BaseModel):
    def __init__(self):
        raise DeprecationWarning('FuseModel is deprecated. Use UpModel instead')
        self._set_fuse_keys()

    @abstractmethod
    def fuse(self, **kwargs): ...

    def _set_fuse_keys(self): 
        ...