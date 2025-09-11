from .basemodel import BaseModel, ReducerModel
from abc import abstractmethod
from inspect import getfullargspec
from .utils import filter_keywords

class UpReducer(ReducerModel):
    def __init__(self, reductions, up_preserves = []) -> None:
        super().__init__(reductions=reductions, up_preserves=up_preserves)

    @abstractmethod
    def transform(self, **kwargs): ...

    @abstractmethod
    def up(self, **kwargs): ...

    def _set_keys(self):
        up_arg_spec = getfullargspec(self.up)
        upkeys = filter_keywords(up_arg_spec.args)
        self.up_keys = [k for k in upkeys if k != 'self']
        self.up_keys

        transform_arg_spec = getfullargspec(self.transform)
        keys = filter_keywords(transform_arg_spec.args)

        transform_child_keys = [k.removeprefix('child_') for k in keys if k.startswith('child_')]
        transform_parent_keys = [k for k in keys if not k.startswith('child_')]

        self.transform_child_keys = transform_child_keys
        self.transform_parent_keys = transform_parent_keys

class DownModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def down(self, **kwargs): ...

    def _set_keys(self):
        arg_spec = getfullargspec(self.down)

        keys = filter_keywords(arg_spec.args)

        down_parent_keys = [k.removeprefix('parent_') for k in keys if k.startswith('parent_')]
        down_child_keys = [k for k in keys if not k.startswith('parent_')]

        self.down_parent_keys = down_parent_keys
        self.down_child_keys = down_child_keys

class UpModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def up(self, **kwargs): ...

    def _set_keys(self):
        arg_spec = getfullargspec(self.up)

        keys = filter_keywords(arg_spec.args)

        up_child_keys = [k.removeprefix('child_') for k in keys if k.startswith('child_')]
        up_current_keys = [k for k in keys if not k.startswith('child_')]

        self.up_child_keys = up_child_keys
        self.up_current_keys = up_current_keys