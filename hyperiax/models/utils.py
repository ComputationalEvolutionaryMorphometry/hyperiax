from typing import TypeVar, List
from jax import ops

Reduction_ops = TypeVar(
    "Reduction_ops", ops.segment_sum, ops.segment_prod, ops.segment_max, ops.segment_min
)


def map_reduction(reduction_str: str) -> Reduction_ops:
    """Maps a string identifier to a JAX reduction operation.

    This utility function provides a simple way to select a JAX segment operation
    (e.g., `segment_sum`, `segment_prod`) using a human-readable string.

    Args:
        reduction_str: The identifier for the reduction operation. Supported values
            are 'sum', 'prod', 'max', and 'min'.

    Returns:
        The corresponding JAX segment operation function.

    Raises:
        ValueError: If the `reduction_str` is not one of the supported operations.
    """
    map = {
        "sum": ops.segment_sum,
        "prod": ops.segment_prod,
        "max": ops.segment_max,
        "min": ops.segment_min,
    }
    if reduction_str not in map.keys():
        raise ValueError(f"Reduction '{reduction_str}' not supported")
    return map[reduction_str]


def filter_keywords(keys: List[str]) -> List[str]:
    """Filters reserved keywords from a list of strings.

    This function is used to clean up lists of function argument names, removing
    keywords that have special meaning within the execution framework (e.g., `self`,
    `params`) and are not intended to be treated as data dependencies from the tree.

    Args:
        keys: A list of strings, typically representing function argument names.

    Returns:
        A new list containing only the strings that are not reserved keywords.
    """
    reserved_keywords = ["self", "params", "key", "root_mask", "leaf_mask"]

    return [k for k in keys if k not in reserved_keywords]
