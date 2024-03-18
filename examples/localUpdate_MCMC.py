# %%
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# %%
def len_to_cov(len_branch,param):
    arr=jnp.tile(len_branch, param['d'])
    return(jnp.diag(arr))

#parameters = {'tree_var': .5,'k_alpha': 1.,'k_sigma': 1.,'obs_var': 1e-2}
def MH_Gibbs(state,len_branches,param,random_key):
    #set up values
    x1=state['A1'] ##mean of B and A2,A3
    x2=state['A2']
    x3=state['A3']
    x=state['B']
    l1=len_branches['B']
    l2=len_branches['A2']
    l3=len_branches['A3']
    sigma_l1=len_to_cov(l1,param)
    sigma_l2=len_to_cov(l2,param)
    sigma_l3=len_to_cov(l3,param)
    sigma_B=sigma_l1
    sigma_AB=jnp.concatenate((sigma_l1, sigma_l1), axis=0)
    sigma_BA=jnp.concatenate((sigma_l1, sigma_l1), axis=1)
    sigma_A=jnp.block([[sigma_l1+sigma_l2,sigma_l1],[sigma_l1,sigma_l1+sigma_l3]])

    #get the inv_sigma
    inv_sigma_A=jnp.linalg.inv(sigma_A)
    inv_sigma_l2=jnp.linalg.inv(sigma_l2)
    inv_sigma_l3=jnp.linalg.inv(sigma_l3)

    #conditional mean and cov_matrix for B
    miu_con=x1+sigma_BA@inv_sigma_A@(jnp.block([[x2],[x3]])-jnp.block([[x1],[x1]]))
    sigma_con=sigma_B-sigma_BA@inv_sigma_A@sigma_AB

    #sample the new B value
    key=jax.random.PRNGKey(random_key)
    new_B = random.multivariate_normal(key, miu_con.T, sigma_con).T
    #calculate the lik_ratio of A2&A3 based on new B / current B
    lik_ration_new_top=jnp.exp(0.5*((x2-x).T@inv_sigma_l2@(x2-x)+
                                    (x3-x).T@inv_sigma_l3@(x3-x)-
                                    (x2-new_B).T@inv_sigma_l2@(x2-new_B)-
                                    (x3-new_B).T@inv_sigma_l3@(x3-new_B)))

    #ratio and uniform comparison
    key, subkey = random.split(key)
    stdunif = random.uniform(subkey, minval=0.0, maxval=1.0)
    B_new=x
    if(lik_ration_new_top> 1 ):
        B_new=new_B
    return(B_new)

def tree_MCMC_1chain(iter,state,len_branches,param):
    B_chain=jnp.array([state['B']])
    for i in range(iter): 
        new_B=MH_Gibbs(state,len_branches,param,i)
        B_chain=jnp.concatenate((B_chain, new_B[jnp.newaxis, :]), axis=0)
        state['B']=new_B
    return(B_chain)

# %%
# 1 dimension
len_branches={'B':1.0,'A2':1.0,'A3':1.0}
param={'d':1,'k_alpha':0.5,'k_sigma':1}
key1=jax.random.PRNGKey(1)
key2=jax.random.PRNGKey(2)
key3=jax.random.PRNGKey(3)
num_samples = 1
x1=jnp.array([0.0]).reshape(1, -1)
x=jax.random.multivariate_normal(key1, x1, len_to_cov(len_branches['B'],param), shape=(num_samples,))
x2=jax.random.multivariate_normal(key2, x, len_to_cov(len_branches['A2'],param), shape=(num_samples,))
x3=jax.random.multivariate_normal(key3, x, len_to_cov(len_branches['A2'],param), shape=(num_samples,))


state = {'A1': x1.T,'B': jnp.array([-0.5]).reshape(-1, 1),'A2': x2.T,'A3': x3.T}
iter=1000

B_chain1=tree_MCMC_1chain(iter,state,len_branches,param)
B_chain=jnp.squeeze(B_chain1)


# %% 
y=B_chain[1:200]
plt.plot(y, marker='o', linestyle='-', markersize=2, linewidth=0.5)
plt.hlines(x,xmin=0,xmax=1+len(y),colors='red')
plt.title('Traceplot of x')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# %%
# 2 dimension
len_branches={'B':3.0,'A2':2.0,'A3':3.0}
param={'d':2,'k_alpha':0.5,'k_sigma':1}
key1=jax.random.PRNGKey(1)
key2=jax.random.PRNGKey(2)
key3=jax.random.PRNGKey(3)
num_samples = 1
x1=jnp.array([0.0,0.0]).reshape(1, -1)
x=jax.random.multivariate_normal(key1, x1, len_to_cov(len_branches['B'],param), shape=(num_samples,))
x2=jax.random.multivariate_normal(key2, x, len_to_cov(len_branches['A2'],param), shape=(num_samples,))
x3=jax.random.multivariate_normal(key3, x, len_to_cov(len_branches['A2'],param), shape=(num_samples,))


state = {'A1': x1.T,'B': jnp.array([-0.5,0.5]).reshape(-1, 1),'A2': x2.T,'A3': x3.T}
iter=10

B_chain1=tree_MCMC_1chain(iter,state,len_branches,param)




