# Automatic backward filtering and forward guiding
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import time

from jax.random import split

from examples.SDE import dts, dot, solve, forward

from hyperiax.execution import OrderedExecutor
from hyperiax.models import UpLambdaReducer, DownLambda, UpLambda
from hyperiax.mcmc import ParameterStore, VarianceParameter

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

# sample new noise on tree
def update_noise(tree,key): 
    tree.data['noise'] = jax.random.normal(key, shape=tree.data['noise'].shape)

# add leaf noise to tree
def add_leaf_noise(tree,key,params):
    if tree.data['value'][0].ndim == 1:
        leaf_values = tree.data['value'][tree.is_leaf]+jnp.sqrt(params['obs_var'].value)*jax.random.normal(key,tree.data['value'][tree.is_leaf].shape)
    else:
        # Only add noise to the last timestep of the values
        leaf_values = tree.data['value'][tree.is_leaf].at[:,-1].add(jnp.sqrt(params['obs_var'].value)*jax.random.normal(key,tree.data['value'][tree.is_leaf][:,-1].shape))
    tree.data['value'] = tree.data['value'].at[tree.is_leaf].set(leaf_values)

# forward guided sampling, assumes already backward filtered (H,F parameters)
def forward_guided(x0,dts,dWs,b,sigma,params,B=None,beta=None,tildea0=None,tildeaT=None,F_T=None,H_T=None,F_t=None,H_t=None):
    # check inputs
    assert(tildeaT is not None)
    assert(F_T is not None or F_t is not None)
    assert(H_T is not None or H_t is not None)

    # dimension
    n = tildeaT.shape[0]
    d = x0.size//n # for tensor product diffusivity, e.g. for landmarks
    # time
    ts = jnp.cumsum(dts); T = ts[-1]

    if tildea0 is None:
        tildea = lambda t,T: tildeaT
        Phi_inv = lambda t: jnp.eye(n)+H_T@tildea*(T-t)
    else:
        tildea = lambda t,T: tildeaT*(t/T)+tildea0*(1-t/T)
        Phi_inv = lambda t:  jnp.eye(n)+H_T@((-(t**2-T**2)/(2*T))*tildeaT+((T-t)**2/(2*T))*tildea0)

    if B is None and beta is None: # closed form solution
        tildeb = lambda t,x,params: jnp.zeros((n*d,))
        Ht = lambda i: solve(Phi_inv(ts[i]),H_T).reshape(H_T.shape) 
        Ft = lambda i: solve(Phi_inv(ts[i]),F_T).reshape(F_T.shape) 
    else:
        tildeb = lambda t,x,params: beta(t,params)+jnp.dot(B(t,params),x)
        Ht = lambda i: H_t[i]
        Ft = lambda i: F_t[i]

    def bridge_SFvdM(carry,val):
        i, X, logpsi = carry
        dt, dW = val
        t = ts[i]
        H = Ht(i); F = Ft(i)
        tilderx =  F-dot(H,X)
        _sigma = sigma(X,params)
        _a = jnp.einsum('ij,kj->ik',_sigma,_sigma)
        
        # SDE
        Xtp1 = X + b(t,X,params)*dt + dot(_a,tilderx)*dt + dot(_sigma,dW)
        tp1 = t + dt
        
        # logpsi
        amtildea = _a-tildea(t,T)
        logpsitp1 = logpsi+(
                jnp.dot(b(t,X,params)-tildeb(t,X,params),tilderx)
                -.5*d*jnp.einsum('ij,ji->',amtildea,H)
                +.5*jnp.einsum('ij,jd,id->',
                           amtildea,tilderx.reshape((n,d)),tilderx.reshape((n,d)))
                    )*dt
        return((i+1,Xtp1,logpsitp1),(t,X,logpsi))    

    # sample
    (T,X,logpsi), (ts,Xs,logpsis) = jax.lax.scan(bridge_SFvdM,(0,x0,0.),(dts,dWs))#,H,F))
    Xscirc = jnp.vstack((Xs, X))
    return Xscirc,logpsi

def Gaussian_down_unconditional(sigma,params_fn=None):
    @jax.jit
    def down_unconditional(key,noise,edge_length,parent_value,params,**args):
        def f(key,noise,edge_length,parent_value):
            var = edge_length # variance is edge length
            _params = params if params_fn is None else params_fn(key,params)
            return {'value': parent_value+jnp.sqrt(var)*dot(sigma(parent_value,_params),noise)}

        return jax.vmap(f)(key,noise,edge_length,parent_value)
    downmodel_unconditional = DownLambda(down_fn=down_unconditional)
    return OrderedExecutor(downmodel_unconditional)

# construct down lambda reducers
def SDE_down_unconditional(n_steps,b,sigma,params_fn=None):
    # down_unconditional using vmap
    @jax.jit
    def down_unconditional(key,noise,edge_length,parent_value,params,**args):
        def f(key,noise,edge_length,parent_value):
            _params = params if params_fn is None else params_fn(key,params)
            var = edge_length # variance is edge length
            _dts = dts(T=var,n_steps=n_steps); _dWs = jnp.sqrt(_dts)[:,None]*noise
            Xs = forward(parent_value.reshape((n_steps+1,-1))[-1],_dts,_dWs,b,sigma,_params)
            return {'value': Xs}

        return jax.vmap(f)(key,noise,edge_length,parent_value)
    downmodel_unconditional = DownLambda(down_fn=down_unconditional)
    return OrderedExecutor(downmodel_unconditional)

def SDE_down_conditional(n_steps,b,sigma,a,B=None,beta=None,params_fn=None):
    @jax.jit
    def down_conditional_closed_form(key,noise,edge_length,v_0,v_T,F_T,H_T,parent_value,params,**args):
        def f(key,noise,edge_length,v_0,v_T,F_T,H_T,parent_value):
            _params = params if params_fn is None else params_fn(key,params)
            var = edge_length # variance is edge length
            _dts = dts(T=var,n_steps=n_steps)
            _dWs = jnp.sqrt(_dts)[:,None]*noise
            tildea0 = a(v_0,_params); tildeaT = a(v_T,_params)
            Xs,logpsi = forward_guided(parent_value.reshape((n_steps+1,-1))[-1],_dts,_dWs,b,sigma,_params,beta=beta,B=B,tildea0=tildea0,tildeaT=tildeaT,F_T=F_T,H_T=H_T)
            return {'value': Xs, 'logpsi': logpsi}

        return jax.vmap(f)(key,noise,edge_length,v_0,v_T,F_T,H_T,parent_value)
    def down_conditional(key,noise,edge_length,v_0,v_T,F_t,H_t,parent_value,params,**args):
        def f(key,noise,edge_length,v_0,v_T,F_t,H_t,parent_value):
            _params = params if params_fn is None else params_fn(key,params)
            var = edge_length # variance is edge length
            _dts = dts(T=var,n_steps=n_steps)
            _dWs = jnp.sqrt(_dts)[:,None]*noise
            tildea0 = a(v_0,_params); tildeaT = a(v_T,_params)
            Xs,logpsi = forward_guided(parent_value.reshape((n_steps+1,-1))[-1],_dts,_dWs,b,sigma,_params,beta=beta,B=B,tildea0=tildea0,tildeaT=tildeaT,F_t=F_t,H_t=H_t)
            return {'value': Xs, 'logpsi': logpsi}

        return jax.vmap(f)(key,noise,edge_length,v_0,v_T,F_t,H_t,parent_value)
    if B is None and beta is None:
        downmodel_conditional = DownLambda(down_fn=down_conditional_closed_form)
    else:
        downmodel_conditional = DownLambda(down_fn=down_conditional)
    return OrderedExecutor(downmodel_conditional)

def Gaussian_down_conditional(n,a,d=1,params_fn=None):
    @jax.jit
    def down_conditional(key,noise,edge_length,c_0,F_0,H_0,F_T,H_T,parent_value,params,**args):
        def f(key,noise,edge_length,c_0,F_0,H_0,F_T,H_T,parent_value):
            _params = params if params_fn is None else params_fn(key,params)
            x = parent_value
            var = edge_length # variance is edge length
            covar = var*a(parent_value,_params) # covariance matrix

            #invSigma = jnp.linalg.inv(covar)
            Sigma = covar
            #H = H_T+invSigma
            invH = jnp.linalg.solve(jnp.eye(n)+Sigma@H_T,Sigma)
            #mu = solve(H,F_T+dot(invSigma,x))
            mu = dot(invH,F_T+solve(Sigma,x))

            # for likelihood ratio
            #Sigma_T = jnp.linalg.inv(H_T)
            Sigma_T = invH
            v_T = dot(Sigma_T,F_T) # solve(H_T,F_T)

            # test
            #Sigma_0 = jnp.linalg.inv(H_0)
            #v_0 = dot(Sigma_0,F_0)
            v_0 = solve(H_0,F_0)

            inv_covar_Sigma_T = jnp.linalg.solve(jnp.eye(n)+H_T@Sigma_T,H_T) # inv(covar+Sigma_T)

            return {
                #'value': mu+jax.scipy.linalg.solve_triangular(jax.scipy.linalg.cholesky(H,lower=True),noise.reshape((n,d))).flatten(),
                'value': mu+dot(jax.scipy.linalg.cholesky(invH,lower=True),noise),
                #'value': mu+dot(jax.scipy.linalg.sqrtm(invH),noise),
                'logw': jnp.sum(jax.vmap(
                    #lambda v_T,parent_value,c_0,F_0: logphi(v_T,parent_value,covar+Sigma_T)-logU(parent_value,c_0,F_0,H_0),
                    lambda v_T,parent_value,c_0,F_0: logphi_H(v_T,parent_value,inv_covar_Sigma_T)-logU(parent_value,c_0,F_0,H_0),
                    (1,1,0,1))(v_T.reshape((n,d)),parent_value.reshape((n,d)),c_0.reshape(d),F_0.reshape((n,d)))),
                }

        return jax.vmap(f)(key,noise,edge_length,c_0,F_0,H_0,F_T,H_T,parent_value)
    downmodel_conditional = DownLambda(down_fn=down_conditional)
    return OrderedExecutor(downmodel_conditional)

#backwards filter
def backward_filter(dts,params,c_T,v_T,F_T,H_T,B=None,beta=None,tildea0=None,tildeaT=None,params_fn=None):
    # backward filter according to the following ODEs
    # dH(u) = (-B(u)'H(u) - H(u)B(u) + H(u)\tilde a(u)H(u)) du
    # dF(u) = (-B(u)'F(u) + H(u)\tilde a(u)F(u) + H(u)β(u)) du
    # dc(u) = (β(u)'F(u) + 1/2 F(u)'\tilde a(u)F(u) - 1/2 tr(H(u)\tilde a(u))) du

    # check inputs
    assert(tildeaT is not None)
    # make c_T an array if it is a scalar
    if jnp.isscalar(c_T):
        c_T = jnp.array([c_T])

    # dimension
    n = tildeaT.shape[0]
    d = v_T.size//n # for tensor product diffusivity, e.g. for landmarks
    # time
    # time
    T = dts.sum()
    ts = jnp.cumsum(jnp.concatenate((jnp.array([0.]), dts)))[:-1] # time discretization


    if tildea0 is None:
        tildea = lambda t: tildeaT
        Phi_inv = lambda t:  jnp.eye(n)+H_T@tildeaT*(T-t)
    else:
        tildea = lambda t: tildeaT*(t/T)+tildea0*(1-t/T)
        Phi_inv = lambda t:  jnp.eye(n)+H_T@((-(t**2-T**2)/(2*T))*tildeaT+((T-t)**2/(2*T))*tildea0)

    if B is None and beta is None: # closed form solution
        H_0 = solve(Phi_inv(0),H_T).reshape(H_T.shape)
        F_0 = solve(Phi_inv(0),F_T).reshape(F_T.shape)
        # log determinant of Phi_inv(0)
        log_det_phi_inv = jnp.linalg.slogdet(Phi_inv(0))[1]
        c_0 = jax.vmap(lambda v_T,c_T: c_T+.5*v_T.T@(H_T-H_0)@v_T-.5*log_det_phi_inv,(1,0))(v_T.reshape((n,d)),c_T)
        #c_0 = jax.vmap(lambda v_T: logphi_H(jnp.zeros(n),v_T,H_0),1)(v_T.reshape((n,d)))
        #c_0 = jax.vmap(lambda v_T,c_T: c_T-logphi_H(jnp.zeros(n),v_T,H_T)+logphi_H(jnp.zeros(n),v_T,H_0),(1,0))(v_T.reshape((n,d)),c_T)
        return {'c_0': c_0, 'F_0': F_0, 'H_0': H_0}
    else:
        # Define the ODE system
        def ode_system(y, tau, params):
            # Unpack state
            Ht, Ft, ct = y[:n*n].reshape((n,n)), y[n*n:n*n+n], y[-1:]
            t = T-tau
            # Get coefficients at time t
            Bt = B(t,params); betat = beta(t,params) if beta is not None else jnp.zeros(n)
            # Compute derivatives
            dH = -Bt.T@Ht - Ht@Bt + Ht@tildea(t)@Ht
            dF = -Bt.T@Ft + Ht@tildea(t)@Ft + Ht@betat
            dc = -(betat.T@Ft + 0.5*Ft.T@tildea(t)@Ft - 0.5*jnp.trace(Ht@tildea(t)))
            # Flatten and combine
            return -jnp.concatenate([dH.flatten(), dF, jnp.array([dc])])
        
        # Solve backwards in time from T to 0
        # Initial conditions at time T
        y0 = jnp.concatenate([H_T.flatten(), F_T, c_T])
        # Solve ODE
        solution = odeint(ode_system, y0, ts, params)
        ## direct Euler method
        #def scan_step(carry, dt):
        #    y_curr, t_curr = carry
        #    t_next = t_curr+dt
        #    # Compute derivative and update using Euler step
        #    dy = ode_system(y_curr, t_curr, params)
        #    y_next = y_curr+dt*dy
        #    return (y_next, t_next), y_next
        ## Use scan to iterate through time steps
        #(y,t), solution = jax.lax.scan( scan_step, (y0, 0.), dts)
        
        # Extract result and final values (at t=0)
        H_t = solution[::-1,:n*n].reshape((-1,n,n))
        F_t = solution[::-1,n*n:n*n+n]
        c_t = solution[::-1,-1:]
        H_0 = H_t[0]; F_0 = F_t[0]; c_0 = c_t[0]

        return {'c_0': c_0, 'F_0': F_0, 'H_0': H_0, 'F_t': F_t, 'H_t': H_t}
        
# initialize tree for up pass
def get_init_up(n,tree,d=1,root=None,n_steps=1,B=None,beta=None):
    tree.add_property('c_0', shape=(d,)); tree.add_property('F_0', shape=(n*d,)); tree.add_property('H_0', shape=(n,n)); tree.add_property('v_0', shape=(n*d,)); tree.add_property('c_T', shape=(d,)); tree.add_property('F_T', shape=(n*d,)); tree.add_property('H_T', shape=(n,n)); tree.add_property('v_T', shape=(n*d,)); tree.add_property('logw'); tree.add_property('logpsi')
    if B is not None or beta is not None:
        tree.add_property('F_t', shape=(n_steps,n*d)); tree.add_property('H_t', shape=(n_steps,n,n));
    if root is not None:
        tree.data['v_0'] = tree.data['v_0'].at[:].set(root); tree.data['v_T'] = tree.data['v_T'].at[:].set(root)
    def init_up(leaf_values,params,v_T=None):
        #tree.data['value'] = tree.data['value'].at[tree.is_leaf].set(leaf_values)
        tree.data['H_T'] = tree.data['H_T'].at[tree.is_leaf].set((jnp.eye(n)/params['obs_var'].value)[None,:,:])
        tree.data['F_T'] = tree.data['F_T'].at[tree.is_leaf].set(jax.vmap(lambda H,v: dot(H,v))(tree.data['H_T'][tree.is_leaf],leaf_values))
        Sigma = params['obs_var'].value*jnp.eye(n)
        tree.data['v_T'] = tree.data['v_T'].at[tree.is_leaf].set(leaf_values)
        tree.data['c_T'] = tree.data['c_T'].at[tree.is_leaf].set(jax.vmap(lambda v: jax.vmap(lambda v: logphi(jnp.zeros(n),v,Sigma),1)(v.reshape((n,d))))(leaf_values))
        # update v_0 from parent v_T
        if v_T is not None and root is not None:
            tree.data['v_0'] = v_T[tree.parents]
            tree.data['v_0'] = tree.data['v_0'].at[tree.is_root].set(root)
    return init_up

# construct up lambda reducer
def SDE_up(n_steps,a,B=None,beta=None,params_fn=None):
    @jax.jit
    def up(key,edge_length,v_0,c_T,v_T,F_T,H_T,params,**args):
        def f(key,edge_length,v_0,c_T,v_T,F_T,H_T):
            var = edge_length # variance is edge length
            T = var # running time of Brownian motion
            _params = params if params_fn is None else params_fn(key,params)
            return backward_filter(dts(T=T,n_steps=n_steps),_params,c_T,v_T,F_T,H_T,B=B,beta=beta,tildea0=a(v_0,_params),tildeaT=a(v_T,_params))
        return jax.vmap(f)(key,edge_length,v_0,c_T,v_T,F_T,H_T)
    def transform(child_c_0,child_F_0,child_H_0,**args):
        F_T = child_F_0
        H_T = child_H_0
        c_T = child_c_0
        v_T = jax.vmap(lambda H,F: solve(H,F))(H_T,F_T)
        return {'c_T': c_T, 'v_T': v_T, 'F_T': F_T, 'H_T': H_T}
    up_preserves = ['c_0','F_t','H_t'] if B is not None or beta is not None else []
    upmodel = UpLambdaReducer(up, transform,
                       reductions={
                           'c_0': 'sum',
                           'F_0': 'sum',
                           'H_0': 'sum',
                        },
                        up_preserves=up_preserves
            )
    return OrderedExecutor(upmodel)

def Gaussian_up(n,a,d=1,params_fn=None):
    @jax.jit
    def up(key,edge_length,c_T,F_T,H_T,params,**args):
        def f(key,edge_length,c_T,F_T,H_T):
            var = edge_length # variance is edge length
            _params = params if params_fn is None else params_fn(key,params)
            #Sigma_T = jnp.linalg.inv(H_T) # alt. Q_T
            #v_T = dot(Sigma_T,F_T)
            v_T = solve(H_T,F_T)
            covar = var*a(v_T,_params) # covariance matrix

            invPhi_0 = (jnp.eye(n)+H_T@covar)
            #Sigma_0 = Sigma_T@invPhi_0 # = Sigma_T+covar, alt. C_0
            Sigma_0 = jnp.linalg.solve(H_T,invPhi_0) # = Sigma_T+covar, alt. C_0
            #H_0 = jnp.linalg.inv(Sigma_0) # hat H
            H_0 = jnp.linalg.solve(invPhi_0,H_T) # hat H
            F_0 = solve(invPhi_0,F_T) # hat F
            v_0 = dot(Sigma_0,F_0)
            c_0 = jax.vmap(
                lambda v_T,c_T,F_T: 
                c_T-logphi_H(jnp.zeros(n),v_T,H_T)+logphi_H(jnp.zeros(n),v_T,H_0), 
                # = (c_T-logphi_can(jnp.zeros(n),F_T,H_T))+logphi_H(jnp.zeros(n),v_T,H_0),
                #jax.scipy.stats.multivariate_normal.logpdf(v_0,jnp.zeros(n*d),Sigma_0),
                (1,0,1))(v_T.reshape((n,d)),c_T.reshape(d),F_T.reshape((n,d)))

            return {'c_0': c_0, 'F_0': F_0, 'H_0': H_0}
        return jax.vmap(f)(key,edge_length,c_T,F_T,H_T)
    def transform(child_c_0,child_F_0,child_H_0,**args):
        F_T = child_F_0
        H_T = child_H_0
        return {'c_T': jax.vmap(lambda F_T,H_T: jax.vmap(lambda F_T: logphi_can(jnp.zeros(n),F_T,H_T),1)(F_T.reshape((n,d))))(F_T,H_T), 'F_T': F_T, 'H_T': H_T}
    upmodel = UpLambdaReducer(up, transform,
                       reductions={
                           'c_0': 'sum',
                           'F_0': 'sum',
                           'H_0': 'sum',
                        }
            )
    return OrderedExecutor(upmodel)

# Crank-Nicolson update with possibly node-dependent lambd
lambd = .9
update_CN = lambda noise,key: noise*lambd+jnp.sqrt((1-lambd**2))*jax.random.normal(key,noise.shape)

# log posterior, SDE case
def get_log_likelihood(tree,down_conditional,up,init_up,include_leaves=False):
    def log_likelihood(data,state):
        """Log likelihood of the tree."""
        params,(noise,v_T) = state
        # backwards filtering with current parameters
        init_up(data,params,v_T); up.up(tree,params.values())
        v = tree.data['value'][tree.is_root][:,0] if tree.data['value'].ndim==3 else tree.data['value'][tree.is_root]
        c,F,H = tree.data['c_T'][tree.is_root],tree.data['F_T'][tree.is_root],tree.data['H_T'][tree.is_root]
        v = jnp.squeeze(v); c = jnp.squeeze(c); F = jnp.squeeze(F); H = jnp.squeeze(H)
        tree_log_likelihood = c.sum()+F@v-.5*v.T@dot(H,v)
        # forwards guiding with current noise
        tree.data['noise'] = noise; down_conditional.down(tree,params.values())
        tree_logcorrection = jnp.sum(tree.data['logpsi']) if 'logpsi' in tree.data else jnp.sum(tree.data['logw'])
        # residuals
        # compute log likelihood
        residuals = tree.data['value'][tree.is_leaf][:,-1]-data if tree.data['value'].ndim==3 else tree.data['value'][tree.is_leaf]-data
        leaves_log_likelihood = jnp.sum(jax.scipy.stats.norm.logpdf(residuals,0,jnp.sqrt(params['obs_var'].value))) if include_leaves else 0.
        return tree_log_likelihood+tree_logcorrection+leaves_log_likelihood
    return log_likelihood

# posterior
def get_log_posterior(log_likelihood,skip_obs_var=False):
    def log_posterior(data,state):
        """Log posterior given the state and data."""
        parameters,_ = state
        log_prior = sum([v.log_prior() for k,v in parameters.params.items() if (not skip_obs_var or k!='obs_var')])
        log_like = log_likelihood(data,state)
        return log_prior + log_like

    return log_posterior

def get_proposal(tree,obs_var_sample_conditional=False):
    def proposal(data, state, key):
        subkeys = jax.random.split(key,2)
        params,(noise,v_T) = state

        # Use static variable to alternate between parameter and noise updates
        if not hasattr(proposal, 'update_params'):
            proposal.update_params = True
        if not hasattr(proposal, 'update_obs_var'):
            proposal.update_obs_var = False

        if proposal.update_params:
            if not obs_var_sample_conditional:
                new_params,log_correction = params.propose(subkeys[1])
            else:
                if not proposal.update_obs_var:
                    # update all parameters except obs_var
                    subkeys = jax.random.split(subkeys[1],len(params.params))
                    proposals_and_corrections = {k: v.propose(rngkey) for rngkey,(k,v) in zip(subkeys,params.params.items()) if k!='obs_var'}
                    new_params = {k: v[0] for k,v in proposals_and_corrections.items()}
                    new_params['obs_var'] = params['obs_var']
                    log_correction = sum(v[1] for v in proposals_and_corrections.values())
                else:
                    # obs_var
                    residuals = tree.data['value'][tree.is_leaf][:,-1]-data if tree.data['value'].ndim==3 else tree.data['value'][tree.is_leaf]-data
                    alpha_post = params['obs_var'].alpha+.5*residuals.size # alpha from prior
                    beta_post = params['obs_var'].beta+.5*jnp.sum(residuals**2) # beta from prior
                    new_obs_var = beta_post/jax.random.gamma(subkeys[-1],alpha_post) # inverse gamma sample
                    new_params = {k: v for k,v in params.params.items() if k!='obs_var'}
                    new_params['obs_var'] = VarianceParameter(**{**params['obs_var'].__dict__,'value':new_obs_var}) if not params['obs_var'].keep_constant else params['obs_var']
                    log_correction = 0.
                new_params = ParameterStore(new_params)
                proposal.update_obs_var = not proposal.update_obs_var
            new_state = new_params,(noise,tree.data['v_T'])
        else:
            new_noise = update_CN(noise,subkeys[1])
            new_state = params,(new_noise,v_T)
            log_correction = 0.

        proposal.update_params = not proposal.update_params
        return new_state,log_correction
    return proposal

# test and time the operations
def test_up_down(down_unconditional,down_conditional,up,init_up,tree,params,leaf_values,key):
    init_up(leaf_values,params)
    up.up(tree,params.values())
    down_conditional.down(tree,params.values())

    # time the operations
    subkey, key = split(key)
    update_noise(tree,subkey)
    down_unconditional.down(tree,params.values())

    start_time = time.time()
    down_unconditional.down(tree,params.values())
    jax.block_until_ready(tree.data['value'])
    print(f"Time elapsed: {(time.time() - start_time)*1000:.4f} ms")

    start_time = time.time()
    down_conditional.down(tree,params.values())
    jax.block_until_ready(tree.data['value'])
    print(f"Time elapsed: {(time.time() - start_time)*1000:.4f} ms")
    
    start_time = time.time()
    up.up(tree,params.values())
    jax.block_until_ready(tree.data['H_0'])
    print(f"Time elapsed: {(time.time() - start_time)*1000:.4f} ms")
    None