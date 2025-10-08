from abc import ABC, abstractmethod
from typing import Dict, List
from inspect import getfullargspec

from .utils import map_reduction, filter_keywords


class BaseModel(ABC):
    """Abstract base class for all models that define a computational operation on a tree.

    This class provides the basic structure for models, ensuring that they are
    initialized with a mechanism to identify the data attributes they require for
    their computations. Subclasses must implement the `_set_keys` method.
    """

    def __init__(self) -> None:
        """Initializes the BaseModel and triggers the key-setting mechanism."""
        super().__init__()
        self._set_keys()

    @abstractmethod
    def _set_keys(self):
        """Abstract method to set the keys used in the model's methods.

        This method is responsible for inspecting the model's operational methods
        (e.g., `up`, `down`, `update`) to determine which data attributes they require
        from different parts of the tree (e.g., parent, child, current node).
        This allows the executor to dynamically provide the necessary data during
        tree traversals.
        """
        ...


class DownModel(BaseModel):
    """
    Abstract base class for models that perform downward passes on a tree.

    A downward pass typically involves propagating information from parent nodes
    to child nodes, starting from the root and moving towards the leaves.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def down(self, **kwargs) -> Dict:
        """
        Define the logic for a downward pass from a node to its children.

        The keyword arguments (**kwargs) for this method are dynamically supplied by
        the executor based on the data available in the tree nodes, which should be categorized
        into three types:
        1) arguments prefixed with 'current_' correspond to data from the current node,
        2) arguments prefixed with 'child_' correspond to data from the child nodes to be distributed,
        3) 'params_dict' corresponds to the parameters dictionary of the tree if applicable.

        Args:
            **kwargs: Keyword arguments representing data from the current and one of its children.

        Returns:
            A dictionary of computed values to be updated in the current node's children.
        """
        ...

    def _set_keys(self):
        """
        Inspects the 'down' method's signature to determine which keys (node attributes)
        it requires from parent nodes and from the current node. This allows the executor
        to provide the correct data during the pass.
        """
        arg_spec = getfullargspec(self.down)
        arg_keys = filter_keywords(arg_spec.args + arg_spec.kwonlyargs)

        current_keys = [
            k.removeprefix("current_") for k in arg_keys if k.startswith("current_")
        ]
        parent_keys = [
            k.removeprefix("parent_") for k in arg_keys if k.startswith("parent_")
        ]

        self.current_keys = current_keys
        self.parent_keys = parent_keys


class UpModel(BaseModel):
    """
    Abstract base class for models that perform upward passes on a tree.

    An upward pass typically involves propagating information from child nodes
    to parent nodes, starting from the leaves and moving towards the root.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def up(self, **kwargs) -> Dict:
        """
        Define the logic for an upward pass from a node to its parent.

        The keyword arguments (**kwargs) for this method are dynamically supplied by
        the executor, which shall be categorized into three types:
        1) arguments prefixed with 'current_' correspond to data from the current node,
        2) arguments prefixed with 'parent_' correspond to data from the parent node to be merged,
        3) 'params_dict' corresponds to the parameters dictionary of the tree if applicable.

        Args:
            **kwargs: Keyword arguments representing data from a current and its parent node.

        Returns:
            A dictionary of computed values to be passed up to the parent node.
        """
        ...

    def _set_keys(self):
        """
        Inspects the 'up' method's signature to determine which keys it requires
        from child nodes and the current node.
        """
        arg_spec = getfullargspec(self.up)
        arg_keys = filter_keywords(arg_spec.args + arg_spec.kwonlyargs)

        current_keys = [
            k.removeprefix("current_") for k in arg_keys if k.startswith("current_")
        ]
        child_keys = [
            k.removeprefix("child_") for k in arg_keys if k.startswith("child_")
        ]

        self.current_keys = current_keys
        self.child_keys = child_keys


class UpdateModel(BaseModel):
    """
    Abstract base class for models that perform updates using information
    from both child and parent nodes. This is often used in up-down algorithms.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update(self, **kwargs) -> Dict:
        """
        Perform an update using information from parent, child, and current nodes.
        **kwargs are dynamically supplied by the executor and categorized as follows:
        1) arguments prefixed with 'parent_' correspond to data from the parent node,
        2) arguments prefixed with 'child_' correspond to data from a child node,
        3) arguments prefixed with 'current_' correspond to data from the current node,
        4) 'params_dict' corresponds to the parameters dictionary of the tree if applicable.

        Args:
            **kwargs: Keyword arguments representing data from parent, child,
                      and current nodes, identified by 'parent_' and 'child_' prefixes.

        Returns:
            A dictionary of computed values to be updated in the current node.
        """
        ...

    def _set_keys(self):
        """
        Inspects the 'update' method's signature to determine which keys it requires
        from parent, child, and current nodes.
        """
        arg_spec = getfullargspec(self.update)
        arg_keys = filter_keywords(arg_spec.args + arg_spec.kwonlyargs)

        current_keys = [
            k.removeprefix("current_") for k in arg_keys if k.startswith("current_")
        ]
        child_keys = [
            k.removeprefix("child_") for k in arg_keys if k.startswith("child_")
        ]
        parent_keys = [
            k.removeprefix("parent_") for k in arg_keys if k.startswith("parent_")
        ]

        self.current_keys = current_keys
        self.child_keys = child_keys
        self.parent_keys = parent_keys


class ReducerModel(BaseModel):
    """
    Abstract base class for models that include a reduction step in an upward pass.

    This model is used when information from multiple children needs to be aggregated
    (e.g., summed, multiplied) before being used by the parent.
    """

    def __init__(
        self, reductions: Dict[str, str], up_preserves: List[str] = []
    ) -> None:
        """
        Initializes the ReducerModel.

        Args:
            reductions (Dict[str, str]): A mapping of data attribute names to the
                reduction operation ('sum', 'prod', 'min', 'max') to be applied.
            up_preserves (List[str], optional): A list of data attribute names from
                the children that should be preserved (concatenated) instead of
                reduced. Defaults to [].
        """
        super().__init__()
        self.reductions = {k: map_reduction(v) for k, v in reductions.items()}
        self.up_preserves = up_preserves


class UpdateReducer(ReducerModel):
    """
    Abstract base class for models that perform an upward pass with reduction,
    followed by an update step.
    """

    def __init__(self, reductions: Dict[str, str]) -> None:
        super().__init__(reductions)

    @abstractmethod
    def update(self, **kwargs) -> Dict:
        """
        Perform an update using reduced information from child nodes and data
        from parent and current nodes. **kwargs are dynamically supplied by the executor
        and categorized as follows:
        1) arguments prefixed with 'parent_' correspond to data from the parent node,
        2) arguments prefixed with 'child_' correspond to reduced data from child nodes,
        3) arguments prefixed with 'current_' correspond to data from the current node,
        4) 'params_dict' corresponds to the parameters dictionary of the tree if applicable.

        Args:
            **kwargs: Keyword arguments representing data from parent, current, and
                      reduced child data.

        Returns:
            A dictionary of computed values to be updated in the current node.
        """
        ...

    @abstractmethod
    def up(self, **kwargs) -> Dict:
        """
        Perform an upward pass to collect data from children for reduction. **kwargs
        are dynamically supplied by the executor and categorized as follows:
        1) arguments prefixed with 'current_' correspond to data from the current node,
        2) arguments prefixed with 'child_' correspond to data from a child node,
        3) 'params_dict' corresponds to the parameters dictionary of the tree if applicable.

        Args:
            **kwargs: Keyword arguments representing data from the current node.

        Returns:
            A dictionary of values to be reduced among siblings.
        """
        ...

    def _set_keys(self):
        """
        Inspects 'up' and 'update' method signatures to determine required keys.
        """
        up_arg_spec = getfullargspec(self.up)
        up_arg_keys = filter_keywords(up_arg_spec.args + up_arg_spec.kwonlyargs)

        up_current_keys = [
            k.removeprefix("current_") for k in up_arg_keys if k.startswith("current_")
        ]
        up_parent_keys = [
            k.removeprefix("parent_") for k in up_arg_keys if k.startswith("parent_")
        ]
        self.up_parent_keys = up_parent_keys
        self.up_current_keys = up_current_keys

        update_arg_spec = getfullargspec(self.update)
        update_arg_keys = filter_keywords(
            update_arg_spec.args + update_arg_spec.kwonlyargs
        )

        update_parent_keys = [
            k.removeprefix("parent_")
            for k in update_arg_keys
            if k.startswith("parent_")
        ]
        update_child_keys = [
            k.removeprefix("child_") for k in update_arg_keys if k.startswith("child_")
        ]
        update_current_keys = [
            k.removeprefix("current_")
            for k in update_arg_keys
            if k.startswith("current_")
        ]

        self.update_parent_keys = update_parent_keys
        self.update_child_keys = update_child_keys
        self.update_current_keys = update_current_keys


class UpReducer(ReducerModel):
    """
    Abstract base class for models that perform an upward pass with a reduction
    step, followed by a transformation at the parent node.
    """

    def __init__(
        self, reductions: Dict[str, str], up_preserves: List[str] = []
    ) -> None:
        super().__init__(reductions, up_preserves)

    @abstractmethod
    def transform(self, **kwargs) -> Dict:
        """
        Transform data at a node after reducing the 'up' results from its children.
        **kwargs are dynamically supplied by the executor and categorized as follows:
        1) arguments prefixed with 'current_' correspond to data from the current node,
        2) arguments prefixed with 'parent_' correspond to data from the parent node to be merged,
        3) 'params_dict' corresponds to the parameters dictionary of the tree if applicable.

        Args:
            **kwargs: Keyword arguments representing data from the current node and
                      the reduced data from its children (prefixed with 'child_').

        Returns:
            A dictionary of transformed values.
        """
        ...

    @abstractmethod
    def up(self, **kwargs) -> Dict:
        """
        Perform an upward pass to collect data from children for reduction.
        **kwargs are dynamically supplied by the executor and categorized as follows:
        1) arguments prefixed with 'current_' correspond to data from the current node,
        2) arguments prefixed with 'parent_' correspond to data from the parent node to be merged,
        3) 'params_dict' corresponds to the parameters dictionary of the tree if applicable.

        Args:
            **kwargs: Keyword arguments representing data from the current node.

        Returns:
            A dictionary of values to be reduced among siblings.
        """
        ...

    def _set_keys(self):
        """
        Inspects 'up' and 'transform' method signatures to determine required keys.
        """
        up_arg_spec = getfullargspec(self.up)
        up_keys = filter_keywords(up_arg_spec.args + up_arg_spec.kwonlyargs)
        up_current_keys = [
            k.removeprefix("current_") for k in up_keys if k.startswith("current_")
        ]
        self.up_current_keys = up_current_keys

        transform_arg_spec = getfullargspec(self.transform)
        transform_arg_keys = filter_keywords(
            transform_arg_spec.args + transform_arg_spec.kwonlyargs
        )

        transform_parent_keys = [
            k.removeprefix("parent_")
            for k in transform_arg_keys
            if k.startswith("parent_")
        ]
        transform_current_keys = [
            k.removeprefix("current_")
            for k in transform_arg_keys
            if k.startswith("current_")
        ]
        self.transform_current_keys = transform_current_keys
        self.transform_parent_keys = transform_parent_keys
