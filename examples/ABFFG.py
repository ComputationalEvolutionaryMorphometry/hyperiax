# Automatic backward filtering and forward guiding
import jax
import jax.numpy as jnp

from examples.SDE import dot, solve, forward

# functions for pullback and likelihood ratios, see https://arxiv.org/abs/2010.03509 section 6.1 for details
quadratic = lambda x,H: jnp.dot(x,jnp.dot(H,x))
logphi = lambda x,mu,Sigma: jax.scipy.stats.multivariate_normal.logpdf(x,mu,Sigma) # log Gaussian density given precision matrix H
phi = lambda x,mu,Sigma: jax.scipy.stats.multivariate_normal.pdf(x,mu,Sigma) # Gaussian density in standard form
omega = lambda Sigma: (jnp.linalg.det(Sigma)*(2*jnp.pi)**Sigma.shape[0])**(-.5) # normalization constant for Gaussian
omega_H = lambda H: jnp.sqrt(jnp.linalg.det(H)/((2*jnp.pi)**H.shape[0])) # normalization constant for Gaussian
logomega = lambda Sigma: .5*(-jnp.linalg.slogdet(Sigma)[1]-jnp.log(2*jnp.pi)*Sigma.shape[0]) # log normalization constant for Gaussian
logomega_H = lambda H: .5*(jnp.linalg.slogdet(H)[1]-jnp.log(2*jnp.pi)*H.shape[0]) # log normalization constant for Gaussian
logphi_H = lambda x,mu,H:  logomega_H(H)-.5*quadratic(x-mu,H) # log Gaussian density in standard form
phi_H = lambda x,mu,H: jnp.exp(logphi_H(x,mu,H))
#logphi_can = lambda y,F,H: logomega_H(H)-.5*jnp.dot(F,jnp.solve(H,F))-.5*jnp.einsum('i,ij,j->',y,H,y)+jnp.dot(F,y) # Gaussian density in canonical form with normalization
logphi_can = lambda y,F,H: logphi_H(y,jnp.linalg.solve(H,F),H)
phi_can = lambda y,F,H: jnp.exp(logphi_can(y,F,H))
logU = lambda y,c,F,H: c-.5*quadratic(y,H)+jnp.dot(F,y) # unnormalized Gaussian density in canonical form
U = lambda y,c,F,H: jnp.exp(logU(y,c,F,H))

# forward guided sampling, assumes already backward filtered (H,F parameters)
def forward_guided(x,H_T,F_T,tildea,dts,dWs,b,sigma,params):
    tildebeta = lambda t,params: 0.
    tildeb = lambda t,x,params: tildebeta(t,params) #+jnp.dot(tildeB,x) #tildeB is zero for now

    T = jnp.sum(dts)
    Phi_inv = lambda t: jnp.eye(H_T.shape[0])+H_T@tildea*(T-t)
    Ht = lambda t: solve(Phi_inv(t),H_T).reshape(H_T.shape) 
    Ft = lambda t: solve(Phi_inv(t),F_T).reshape(F_T.shape) 

    def bridge_SFvdM(carry, val):
        t, X, logpsi = carry
        #dt, dW, H, F = val
        dt, dW = val
        H = Ht(t); F = Ft(t)
        tilderx =  F-dot(H,X)
        _sigma = sigma(x,params)
        _a = jnp.einsum('ij,kj->ik',_sigma,_sigma)
        n = _a.shape[0]
        
        # SDE
        Xtp1 = X + b(t,X, params)*dt + dot(_a,tilderx)*dt + dot(_sigma,dW)
        tp1 = t + dt
        
        # logpsi
        amtildea = _a-tildea
        logpsicur = logpsi+(
                jnp.dot(b(t,X,params)-tildeb(t,X,params),tilderx)
                -.5*jnp.einsum('ij,ji->',amtildea,H)
                +.5*jnp.einsum('ij,jd,id->',
                           amtildea,tilderx.reshape((n,-1)),tilderx.reshape((n,-1)))
                    )*dt
        return((tp1,Xtp1,logpsicur),(t,X,logpsi))    

    # sample
    (T,X,logpsi), (ts,Xs,logpsis) = jax.lax.scan(bridge_SFvdM,(0.,x,0.),(dts,dWs))#,H,F))
    Xscirc = jnp.vstack((Xs, X))
    return Xscirc,logpsi