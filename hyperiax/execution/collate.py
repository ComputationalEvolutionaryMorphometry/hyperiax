from jax import numpy as jnp

def tuple_collate(vals):
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
    def __init__(self, target):
        self.target = target

    def __iter__(self):
        self.iter_dict = {k: iter(v) for k,v in self.target.items()}
        return self

    def __next__(self):
        return {k:next(v) for k,v in self.iter_dict.items()}