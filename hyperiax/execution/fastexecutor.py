from __future__ import annotations
from functools import partial
from typing import Dict, Union

from jax import jit, Array
from jax.typing import ArrayLike
from jax.lax import slice_in_dim
import jax.random as jr

from ..tree import HypTree
from ..models import UpReducer, DownModel, UpModel


class OrderedExecutor(object):
    """Abstract for the ordered executor.

    Ordered executors performs operations in a certain direction (eg. up/down).

    It contains on abstract method `_determine_execution_order``, which gives the order in which nodes
    are to be executed in the tree. It is assumed that all operations are batched.
    """

    def __init__(
        self, model: Union[UpModel, DownModel, UpReducer], rng_key: Array = None
    ) -> None:
        self.rng_key = rng_key if rng_key else jr.key(0)
        self.model = model

    @partial(jit, static_argnames=["self", "tree"])
    def _down_inner(
        self,
        data: Dict[str, ArrayLike],
        tree: HypTree,
        **kwargs,
    ) -> Dict[str, ArrayLike]:
        """
        Perform downwards pass on the tree, this is a jitted inner function.
        This shall be called by the `down` method.

        Args:
            data (Dict[str, ArrayLike]): data stored in the tree
            tree (HypTree): the tree to execute on
            params_dict (Dict[str, Parameter]): parameters for the model

        Returns:
            Dict[str, ArrayLike]: the result of the downwards pass
        """
        for level_start, level_end in tree.levels[1:]:
            # ? Not sure if we need to pass the key down
            # current_data = {
            #     f"current_{k}": slice_in_dim(data[k], level_start, level_end)
            #     for k in self.model.current_keys
            # } | {"key": jnp.arange(level_start, level_end)}
            current_data = {
                f"current_{k}": slice_in_dim(data[k], level_start, level_end)
                for k in self.model.current_keys
            }
            parent_indices = tree.parents[
                level_start:level_end
            ]  # no need to lax since tree is untraced
            parent_data = {
                f"parent_{k}": data[k][parent_indices] for k in self.model.parent_keys
            }

            result = self.model.down(**current_data, **parent_data, **kwargs)

            for k, val in result.items():
                data[k] = data[k].at[level_start:level_end].set(val)
        return data

    @partial(jit, static_argnames=["self", "tree"])
    def _up_inner(
        self,
        data: Dict[str, ArrayLike],
        tree: HypTree,
        **kwargs,
    ) -> Dict[str, ArrayLike]:
        """
        Perform upwards pass without reduction on the branching nodes, this is a jitted inner function. This shall be called by the `up` method and ONLY if
        the tree's child indices have been precomputed.

        Args:
            data (Dict[str, ArrayLike]): data stored in the tree
            tree (HypTree): the tree to execute on
            params_dict (Dict[str, Parameter]): parameters for the model

        Returns:
            dict[str, ArrayLike]: the result of the upwards pass
        """
        iterator = zip(
            reversed(tree.levels),  # indices for levels
            reversed(tree.level_non_leaf_indices),  # precomputed non-leaf indices
        )
        # Process all levels, using precomputed non-leaf indices
        for (level_start, level_end), non_leaf_indices in iterator:
            if len(non_leaf_indices) == 0:
                continue  # Skip if all nodes in this level are leaves

            # Only gather data for non-leaf nodes
            current_data = {
                f"current_{k}": data[k][non_leaf_indices]
                for k in self.model.current_keys
            }
            # ? Not sure if we need to pass the key down
            # current_data = {
            #     f"current_{k}": data[k][non_leaf_indices]
            #     for k in self.model.current_keys
            # } | {"key": jnp.arange(level_start, level_end)}

            # Only gather child data for non-leaf nodes
            child_data = {
                f"child_{k}": data[k][tree.gather_child_idx[non_leaf_indices]]
                for k in self.model.child_keys
            }

            result = self.model.up(
                **current_data,
                **child_data,
                # ? Are these needed?
                # root_mask=tree.is_root[non_leaf_indices],
                # leaf_mask=jnp.zeros_like(
                #     non_leaf_indices, dtype=bool
                # ),  # All nodes we process are non-leaves
                **kwargs,
            )

            # Update data only for non-leaf nodes
            for k, val in result.items():
                data[k] = data[k].at[non_leaf_indices].set(val)

        return data

    @partial(jit, static_argnames=["self", "tree"])
    def _up_reduce_inner(
        self,
        data: Dict[str, ArrayLike],
        tree: HypTree,
        **kwargs,
    ) -> Dict[str, Array]:
        """
        Perform upwards pass with reductions on the branching nodes, this is a jitted inner function. This shall be called by the `up` method. In most cases when the tree's child indices have not been precomputed by setting `precompute_child_gather=False` in the `HypTree` constructor.

        Args:
            data (dict[str, ArrayLike]): data stored in the tree
            tree (HypTree): the tree to execute on
            params_dict (dict[str, Parameter]): parameters for the model

        Returns:
            dict[str, Array]: the result of the upwards and reduction
        """
        # Excluding the root level as it has no parents
        iterator = zip(
            reversed(tree.levels[1:]),  # indices for levels
            reversed(tree.pbuckets_ref[1:]),  # indices for upwards propagation
        )
        for (level_start, level_end), up_ref in iterator:
            # ? Not sure if we need to pass the key down
            # up_current_data = (
            #     {
            #         f"current_{k}": slice_in_dim(data[k], level_start, level_end)
            #         for k in self.model.up_current_keys  # the current node is acting as a child to the parent to be fused
            #     }
            #     | {"key": jnp.arange(level_start, level_end)}
            # )
            up_current_data = {
                f"current_{k}": slice_in_dim(data[k], level_start, level_end)
                for k in self.model.up_current_keys  # the current node is acting as a child to the parent to be fused
            }

            up_result = self.model.up(**up_current_data, **kwargs)

            # save specified values
            for k in self.model.up_preserves:
                data[k] = data[k].at[level_start:level_end].set(up_result[k])

            segments = tree.pbuckets[level_start:level_end]

            current_data = {
                f"current_{k}": self.model.reductions[k](
                    v, segments, num_segments=len(up_ref), indices_are_sorted=True
                )
                for k, v in up_result.items()
                if k in self.model.reductions
            }

            parent_data = {k: data[k][up_ref] for k in self.model.transform_parent_keys}

            result = self.model.transform(**parent_data, **current_data, **kwargs)
            for k, val in result.items():
                data[k] = data[k].at[up_ref].set(val)

        return data

    def down(self, tree: HypTree, **kwargs) -> None:
        """
        Runs the down pass on the tree

        Args:
            tree (HypTree): the tree to execute on
            params_dict (dict, optional): parameters for the model. Defaults to {}.

        Raises:
            ValueError: if the model is not of the correct type
        """
        if not issubclass(type(self.model), DownModel):
            raise ValueError(
                "down() requires model to be of type DownModel, but received {type(self.model)}"
            )

        new_data = self._down_inner(tree.data, tree, **kwargs)

        tree.data = {**tree.data, **new_data}

        return

    def up(self, tree: HypTree, **kwargs) -> None:
        """
        Runs the up pass on the tree. Depends on whether the model is an UpReducer or UpModel,
        the 'up then reduce' or 'up' function will be used.

        Args:
            tree (HypTree): the tree to execute on
            params_dict (dict[str, Parameter], optional): parameters for the model. Defaults to {}.

        Raises:
            ValueError: if the model is not of the correct type
        """
        # TODO: up function may produce new data in the future
        # out here we fetch the data stored in the tree
        # for k,val_shape in self.model.up_produces.items():
        #    if k not in tree.data:
        #        tree.add_property(k, val_shape)

        if issubclass(type(self.model), UpReducer):
            new_data = self._up_reduce_inner(tree.data, tree, **kwargs)

            tree.data = {**tree.data, **new_data}

            return
        elif issubclass(type(self.model), UpModel):
            new_data = self._up_inner(tree.data, tree, **kwargs)

            tree.data = {**tree.data, **new_data}

            return
        else:
            raise ValueError(
                "up() requires model to be of type UpReducer or UpModel, but received {type(self.model)}"
            )
