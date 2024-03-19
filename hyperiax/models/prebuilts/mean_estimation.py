from ..updownmodel import UpModel
import jax
from jax import numpy as jnp
from functools import partial
from ..functional import pass_up

class PhyloMeanModel(UpModel):
    """Prebuilt model that calculates phylogenetic means in each node

    Assumes leafs all contain `estimated_value` before running it. This corresponds to actual observations.
    """
    def __init__(self,param_config = {}, **kwargs) -> None:
        super().__init__()

    @partial(jax.jit, static_argnums=0)
    def up(self, estimated_value, edge_length,**kwargs):
        return {'estimated_value': estimated_value, 'edge_length': edge_length}

    def fuse(self, child_estimated_value, child_edge_length, **kwargs):

        childrent_inv = 1 / child_edge_length

        result = jnp.einsum('c1,cd->d',childrent_inv, child_estimated_value)/childrent_inv.sum()
        return {'estimated_value': result}