from typing import Callable, Any, Dict, List
from jax.typing import ArrayLike

from .basemodels import DownModel, UpModel, UpdateModel, UpReducer, UpdateReducer


class UpLambda(UpModel):
    """A lambda-based model for an upward pass on a tree.

    This class wraps a user-provided function, allowing for the definition of an
    `UpModel`'s logic without needing to create a new subclass. The signature of
    the provided function is inspected to determine the necessary data dependencies
    from the tree nodes.
    """

    def __init__(self, up_fn: Callable[..., Dict[str, ArrayLike]]) -> None:
        """Initializes the UpLambda model with a user-defined function.

        The signature of `up_fn` is inspected to determine what data is required
        from child and current nodes. The arguments in `up_fn` should be prefixed
        with `current_` or `child_` to specify the origin of the data.

        Args:
            up_fn: A function that performs the upward pass calculation. It receives
                data from a child and the current node as keyword arguments and
                returns a dictionary of values to be passed to the parent.
        """
        self.up = up_fn
        super().__init__()

    def up(self, *args, **kwargs):
        """Placeholder for the up method.

        This method is replaced at initialization by the `up_fn` provided by the user.
        Calling it directly will raise a `NotImplementedError`.
        """
        raise NotImplementedError(
            "The 'up' method should be provided during model initialization."
        )


class DownLambda(DownModel):
    """A lambda-based model for a downward pass on a tree.

    This class wraps a user-provided function, allowing for the definition of a
    `DownModel`'s logic without needing to create a new subclass. The signature of
    the provided function is inspected to determine the necessary data dependencies
    from the tree nodes.
    """

    def __init__(self, down_fn: Callable[[Any], Dict]) -> None:
        """Initializes the DownLambda model with a user-defined function.

        The signature of `down_fn` is inspected to determine what data is required
        from parent and current nodes. The arguments in `down_fn` should be prefixed
        with `current_` or `parent_` to specify the origin of the data.

        Args:
            down_fn: A function that performs the downward pass calculation. It
                receives data from the parent and current node as keyword arguments
                and returns a dictionary of values to be passed to the children.
        """
        self.down = down_fn
        super().__init__()

    def down(self, *args, **kwargs):
        """Placeholder for the down method.

        This method is replaced at initialization by the `down_fn` provided by the user.
        Calling it directly will raise a `NotImplementedError`.
        """
        raise NotImplementedError(
            "The 'down' method should be provided during model initialization."
        )


class UpLambdaReducer(UpReducer):
    """A lambda-based model for an upward pass with reduction and transformation.

    This model wraps user-provided functions for the `up` and `transform` steps
    of an `UpReducer` model. It allows for defining the model's logic without
    subclassing, combining data collection, reduction, and transformation.
    """

    def __init__(
        self,
        up_fn: Callable[..., Dict[str, ArrayLike]],
        transform_fn: Callable[..., Dict[str, ArrayLike]],
        reductions: Dict[str, str],
        up_preserves: List[str] = [],
    ) -> None:
        """Initializes the UpLambdaReducer model.

        The signatures of `up_fn` and `transform_fn` are inspected to determine
        the required data attributes for the operations.

        Args:
            up_fn: A function that collects data from a node to be passed upwards
                for reduction. Its arguments should be prefixed with `current_` or
                `child_`.
            transform_fn: A function that transforms data at a parent node after
                reducing the results from its children. Its arguments should be
                prefixed with `current_` or `child_` (for reduced data).
            reductions: A mapping of attribute names to reduction operations
                (e.g., 'sum', 'prod').
            up_preserves: A list of attribute names to preserve (concatenate)
                instead of reducing. Defaults to [].
        """
        self.up = up_fn
        self.transform = transform_fn
        super().__init__(reductions, up_preserves)

    def up(self, *args, **kwargs):
        """Placeholder for the up method.

        This method is replaced at initialization by the `up_fn` provided by the user.
        Calling it directly will raise a `NotImplementedError`.
        """
        raise NotImplementedError(
            "The 'up' method should be provided during model initialization."
        )

    def transform(self, *args, **kwargs):
        """Placeholder for the transform method.

        This method is replaced at initialization by the `transform_fn` provided by the user.
        Calling it directly will raise a `NotImplementedError`.
        """
        raise NotImplementedError(
            "The 'transform' method should be provided during model initialization."
        )


class UpdateLambdaReducer(UpdateReducer):
    """A lambda-based model for an up-pass with reduction followed by an update.

    This model wraps user-provided functions for the `up` and `update` steps,
    allowing for a common pattern of collecting and reducing data from children
    before updating the current node.
    """

    def __init__(
        self,
        up_fn: Callable[..., Dict[str, ArrayLike]],
        update_fn: Callable[..., Dict[str, ArrayLike]],
        reductions: Dict[str, str],
    ) -> None:
        """Initializes the UpdateLambdaReducer model.

        Args:
            up_fn: A function that collects data from a node to be passed upwards
                for reduction. Its arguments should be prefixed with `current_` or
                `child_`.
            update_fn: A function that performs an update using data from parent,
                current, and reduced child nodes. Its arguments should be prefixed
                with `parent_`, `current_`, or `child_`.
            reductions: A mapping of attribute names to reduction operations.
        """
        self.up = up_fn
        self.update = update_fn
        super().__init__(reductions)

    def up(self, *args, **kwargs):
        """Placeholder for the up method.

        This method is replaced at initialization by the `up_fn` provided by the user.
        Calling it directly will raise a `NotImplementedError`.
        """
        raise NotImplementedError(
            "The 'up' method should be provided during model initialization."
        )

    def update(self, *args, **kwargs):
        """Placeholder for the update method.

        This method is replaced at initialization by the `update_fn` provided by the user.
        Calling it directly will raise a `NotImplementedError`.
        """
        raise NotImplementedError(
            "The 'update' method should be provided during model initialization."
        )


class UpdateLambda(UpdateModel):
    """A lambda-based model for performing a local update on a node.

    This model wraps a user-provided function to be used as the `update` method,
    which is useful for operations that require data from a node's parent,
    children, and itself simultaneously.
    """

    def __init__(self, update_fn: Callable[..., Dict[str, ArrayLike]]) -> None:
        """Initializes the UpdateLambda model with a user-defined function.

        The signature of the provided `update_fn` is inspected to determine what data
        is required from parent, child, and current nodes. Arguments should be
        prefixed with `parent_`, `child_`, or `current_`.

        Args:
            update_fn: A function that performs the update. It receives data from
                parent, child, and current nodes as keyword arguments and should
                return a dictionary of new values for the current node.
        """
        self.update = update_fn
        super().__init__()

    def update(self, *args, **kwargs):
        """Placeholder for the update method.

        This method is replaced at initialization by the `update_fn` provided by the user.
        Calling it directly will raise a `NotImplementedError`.
        """
        raise NotImplementedError(
            "The 'update' method should be provided during model initialization."
        )
