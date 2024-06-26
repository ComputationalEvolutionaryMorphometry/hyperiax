from .basemodel import ReducerModel, BaseModel
from abc import abstractmethod
from ..mcmc import ParameterStore
from inspect import getfullargspec
from .utils import filter_keywords

class UpdateReducer(ReducerModel):
    def __init__(self, reductions) -> None:
        super().__init__(reductions=reductions)

    @abstractmethod
    def update(self, **kwargs): ...

    @abstractmethod
    def up(self, **kwargs): ...

    def _set_keys(self):
        up_arg_spec = getfullargspec(self.model.up)
        upkeys = up_arg_spec.args
        filter_keywords(upkeys)
        self.up_keys = upkeys


        update_arg_spec = getfullargspec(self.model.update)
        keys = filter_keywords(update_arg_spec.args)

        update_parent_keys = [k.removeprefix('parent_') for k in keys if k.startswith('parent_')]
        update_node_keys = [k for k in keys if not k.startswith('parent_') and not k.startswith('child_')]

        self.update_parent_keys = update_parent_keys
        self.update_node_keys = update_node_keys

class UpdateModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update(self, **kwargs): ...

    def _set_keys(self):
        update_arg_spec = getfullargspec(self.update)
        keys = filter_keywords(update_arg_spec.args)
        update_child_keys = [k.removeprefix('child_') for k in keys if k.startswith('child_')]
        update_parent_keys = [k.removeprefix('parent_') for k in keys if k.startswith('parent_')]
        update_current_keys = [k for k in keys if not k.startswith('parent_') and not k.startswith('child_')]

        self.update_child_keys = update_child_keys
        self.update_parent_keys = update_parent_keys
        self.update_current_keys = update_current_keys