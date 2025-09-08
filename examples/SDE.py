# SDE utilities and guiding

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky

# for 2d + 3d cases with factorizable matrices
# multiply on the factorized matrix, e.g. covariance matrix
dot = lambda A,v: jnp.einsum('ij,jd->id',A,v.reshape((A.shape[0],-1))).flatten()
# multiple on inverse factorized matrix, e.g. inverse covariance matrix
solve = lambda A,v: jnp.linalg.solve(A,v.reshape((A.shape[0],-1))).flatten()

# time increments
def dts(T=1.,n_steps=100):
    return jnp.array([T/n_steps]*n_steps)

# Euler-Maruyama SDE integration
def forward(x,dts,dWs,b,sigma,params,a=None):
    def SDE(carry, val):
        t,X = carry
        dt,dW = val
        
        # SDE
        if sigma is not None:
            Xtp1 = X + b(t,X,params)*dt + dot(sigma(x,params),dW)
        else:
            assert(a is not None)
            Xtp1 = X + b(t,X,params)*dt + dot(cholesky(a(x,params),lower=True,check_finite=False),dW)
        tp1 = t + dt
        
        return((tp1,Xtp1),(t,X))    

    # sample
    (T,X), (ts,Xs) = jax.lax.scan(SDE,(0.,x),(dts,dWs))
    Xs = jnp.vstack((Xs,X))
    return Xs
