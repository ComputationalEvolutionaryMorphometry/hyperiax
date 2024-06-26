from ..updownmodel import UpReducer
import jax
from jax import numpy as jnp
from functools import partial
from ..functional import pass_up

class PhyloMeanModel(UpReducer):
    """Prebuilt model that calculates phylogenetic means in each node

    Assumes leaves all contain `estimated_value` before running it. This corresponds to actual observations.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(reductions={'weighted_value': 'sum', 'inverse_edge_length': 'sum'}, **kwargs)

    def up(self, estimated_value, edge_length, **kwargs):
        return {'weighted_value': estimated_value/edge_length, 'inverse_edge_length':1/edge_length}


    def transform(self, child_weighted_value,child_inverse_edge_length, **kwargs):
        return {'estimated_value': child_weighted_value/child_inverse_edge_length}