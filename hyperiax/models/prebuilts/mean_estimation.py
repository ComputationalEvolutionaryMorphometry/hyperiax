from ..updownmodel import UpModel
import jax
from jax import numpy as jnp
from functools import partial
from ..functional import pass_up

class PhyloMeanModel(UpModel):
    """Prebuilt model that calculates phylogenetic means in each node

    Assumes leaves all contain `estimated_value` before running it. This corresponds to actual observations.
    """
    def __init__(self,param_config = {}, **kwargs) -> None:
        super().__init__()

    @partial(jax.jit, static_argnums=0)
    def up(self, estimated_value, edge_length,**kwargs):
        """inputs functions to fuse 

        :param estimated_value: value from node
        :param edge_length: value from node 
        :return: outputs same the estimated parameters to parent (from fuse) 
        """
        return {'estimated_value': estimated_value, 'edge_length': edge_length}

    def fuse(self, child_estimated_value, child_edge_length, **kwargs):
        """fuse 

        :param child_estimated_value: value from child node
        :param child_edge_length: value from child node
        :return: phylogenetic mean of the children returned to parent 
        """

        childrent_inv = 1 / child_edge_length

        result = jnp.einsum('c1,cd->d',childrent_inv, child_estimated_value)/childrent_inv.sum()
        return {'estimated_value': result}