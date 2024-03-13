from ..updownmodel import UpDownModel
import jax
from jax import numpy as jnp
from functools import partial

###
### # I cannot get this to work corretly.... 06/03/2024
###

class lddmm_alignment(UpDownModel):
    def __init__(self, **kwargs) -> None:
        from jaxgeometry_src.manifolds.landmarks import landmarks   
        from jaxgeometry_src.Riemannian import metric
        from jaxgeometry_src.dynamics import Hamiltonian
        from jaxgeometry_src.Riemannian import Log
        from jaxgeometry_src.dynamics import flow_differential
        super().__init__(up=self._up, down=None, fuse = self._fuse, **kwargs)

    @partial(jax.jit, static_argnums=0)

    def up(self, landmarks, edge_length,**args):
        return (landmarks, edge_length)

    def fuse(self, messages, **kwargs):
        def lddmm(childxs1,childxs2,parent_placement):
    
            # Initialize the flow, these are hard coded for now 
            sigma_k = 0.5 # just a number for now, should be changed 
            n_landmarks = jnp.shape(childxs1)[0]
            
            M = landmarks(n_landmarks,k_sigma=sigma_k*jnp.eye(2)) 
            # Riemannian structure

            metric.initialize(M)

            q = M.coords(jnp.vstack(childxs1).flatten())
            v =  (jnp.array(jnp.vstack(childxs2).flatten()),[0])
        
            Hamiltonian.initialize(M)
            # Logarithm map
            Log.initialize(M,f=M.Exp_Hamiltonian)

            # Estimate momentum 
            p = M.Log(q,v)[0]
        
            
            # Hammilton 
            (_,qps,charts_qp) = M.Hamiltonian_dynamics(q,p,dts(n_steps=100))
        
            
            #lift
            flow_differential.initialize(M)
            _,dphis,_ = M.flow_differential(qps,dts())
            dphi_t = dphis[int(100*parent_placement)]
            eta_t = jax.vmap(lambda A,v: jnp.dot(A.T,v))(dphi_t,p.reshape((M.N,M.m)))


            return qps[:,0][int(100*parent_placement)].reshape(-1,2),eta_t
        
        childrenxs, childrent = messages

        l_min = jnp.min(childrent)
        l_max = jnp.max(childrent)
        if l_min == l_max:
            parent_placement = 50
        else: 
            parent_placement = int((l_max-l_min)/l_max*100)

        lddmm_landmarks, lddmm_eta_left  = lddmm(childrenxs[0],childrenxs[1],parent_placement)
        _, lddmm_eta_right = lddmm(childrenxs[1],childrenxs[0],100-parent_placement)
        

        return {'landmarks':lddmm_landmarks,"eta_left":lddmm_eta_left,"eta_right":lddmm_eta_right}
    def __call__(self, tree):
        return super().up(tree)
