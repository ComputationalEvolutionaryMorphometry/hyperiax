from jax import numpy as jnp

def tuple_collate(vals):
    """Stacks tuples of identical shape into one tuple of said shape with a batch dimension

    Args:
        vals (list[tuple]): the tuples to be stacked

    Returns:
        tuple[list]: _description_
    """
    raise DeprecationWarning()
    if all(type(v) != tuple for v in vals):
        return jnp.stack(vals)
    res = tuple(map(jnp.stack, zip(*vals)))
    return res

def dict_collate(vals):
    ml = len(vals)
    keys = vals[0].keys()
    for d in vals:
        keys &= d.keys()
            
    return {key: jnp.stack([d[key] for d in vals]) for key in keys}


class DictTransposer:
    """Iterator that wraps a dictionary of lists

        Iteration leads a dictionary with corresponding elements to list entires.
    """
    def __init__(self, target):
        self.target = target

    def __iter__(self):
        self.iter_dict = {k: iter(v) for k,v in self.target.items()}
        return self

    def __next__(self):
        return {k:next(v) for k,v in self.iter_dict.items()}