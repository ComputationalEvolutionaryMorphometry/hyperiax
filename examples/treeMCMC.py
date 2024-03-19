import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

def state2cov(state, alpha, sigma, length):
    # TODO: Jacky to write, input is landmark vector, output is cov matrix
    return jnp.identity(jnp.size(state))

def prob(stateC, stateP, stateL, stateR, invcovC, invcovP):
    # TODO: Jacky to write, compute conditional probability of current state vector (stateC) given Parent, left and right child, and inverse of cov matrices
    return 1
    
def sample(stdnorm, stateP, stateL, stateR, invcovC, invcovP):
    # TODO: Jacky to write, sample current state vector (stateC) given Parent, left and right child, and inverse of cov matrices, stdnorm is a vector of samples from standard normal distribution
    return stdnorm

state2cov_jit = jit(state2cov)
prob_jit = jit(prob)
sample_jit = jit(sample)

class TreeMCMC:
    def initNode(self, parentID, treeDict, depth):
        nodeID = self.nodeCnt
        self.nodeCnt += 1
        self.parent.append(parentID)
        self.leftChild.append(-1)
        self.rightChild.append(-1)
        self.states.append(treeDict["state"])
        self.lengths.append(treeDict["length"] if parentID != -1 else 0)
        if "left" in treeDict:
            self.leftChild[nodeID] = self.initNode(nodeID, treeDict["left"], depth + 1)
            self.rightChild[nodeID] = self.initNode(nodeID, treeDict["right"], depth + 1)
            if parentID != -1:
                if depth % 2 == 0:
                    self.A.append(nodeID)
                else:
                    self.B.append(nodeID)
        return nodeID
    
    def __init__(self, treeDict, parameters):
        self.parameters = parameters
        self.nodeCnt = 0
        self.A = []
        self.B = []
        self.parent = []
        self.leftChild = []
        self.rightChild = []
        self.states = []
        self.lengths = []
        self.initNode(-1, treeDict, 0)
        
    def MCMC(self, iterations):
        key = random.PRNGKey(0)
        states = jnp.array(self.states)
        lengths = jnp.array(self.lengths)
        alpha = self.parameters["alpha"]
        sigma = self.parameters["sigma"]
        A = jnp.array(self.A)
        B = jnp.array(self.B)
        parent = jnp.array(self.parent)
        PA = parent[A]
        PB = parent[B]
        leftChild = jnp.array(self.leftChild)
        LA = leftChild[A]
        LB = leftChild[B]
        rightChild = jnp.array(self.rightChild)
        RA = rightChild[A]
        RB = rightChild[B]
        
        #invcovs = random.normal(key, (states.shape[0], states.shape[1], states.shape[1]))
        probs = jnp.ones(lengths.shape)
        covs = vmap(state2cov_jit)(states, alpha * jnp.ones(lengths.shape), sigma * jnp.ones(lengths.shape), lengths)
        invcovs = vmap(jnp.linalg.inv)(covs)
        for C, P, L, R in [(A, PA, LA, RA), (B, PB, LB, RB)]:
            statesC, statesP, statesL, statesR = (states[C], states[P], states[L], states[R])
            invcovC, invcovP = (invcovs[C], invcovs[P])
            probsC = vmap(prob_jit)(statesC, statesP, statesL, statesR, invcovC, invcovP)
            probs = probs.at[C].set(probsC)
            
        for i in range(iterations):
            for C, P, L, R in [(A, PA, LA, RA), (B, PB, LB, RB)]:
                statesC, statesP, statesL, statesR = (states[C], states[P], states[L], states[R])
                invcovC, invcovP = (invcovs[C], invcovs[P])
                probsC = probs[C]
                lengthsC = lengths[C]
                
                key, subkey = random.split(key)
                stdnorm = random.normal(subkey, jnp.shape(statesC))
                newStatesC = vmap(sample_jit)(stdnorm, statesP, statesL, statesR, invcovC, invcovP)
                newCovsC = vmap(state2cov_jit)(newStatesC, alpha * jnp.ones(lengthsC.shape), sigma * jnp.ones(lengthsC.shape), lengthsC)
                newInvcovsC = vmap(jnp.linalg.inv)(newCovsC)
                newProbsC = vmap(prob_jit)(newStatesC, statesP, statesL, statesR, newInvcovsC, invcovP)
                
                key, subkey = random.split(key)
                stdunif = random.uniform(subkey, jnp.shape(C))
                accepted = jnp.where(newProbsC > stdunif * probsC)
                states = states.at[C[accepted]].set(newStatesC[accepted])
                invcovs = invcovs.at[C[accepted]].set(newInvcovsC[accepted])
                probs = probs.at[C[accepted]].set(newProbsC[accepted])
            
            print(states[A], states[B])
        
        self.states = states
        
        
def __main__():
    n = 3
    treeDict = {"state": np.random.normal(size=n), "length": np.random.normal(), "left": {"state": np.random.normal(size=n), "length": np.random.normal()}, "right":
               {"state": np.random.normal(size=n), "length": np.random.normal(), "left": {"state": np.random.normal(size=n), "length": np.random.normal()}, "right":
               {"state": np.random.normal(size=n), "length": np.random.normal(), "left": {"state": np.random.normal(size=n), "length": np.random.normal()}, "right":
               {"state": np.random.normal(size=n), "length": np.random.normal(), "left": {"state": np.random.normal(size=n), "length": np.random.normal()}, "right":
               {"state": np.random.normal(size=n), "length": np.random.normal()}}}}}
    parameters = {"alpha": 1, "sigma": 1}
    tree = TreeMCMC(treeDict, parameters)
    tree.MCMC(3)

__main__()