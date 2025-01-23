# Shape related utiliti functions

import numpy as np
import jax.numpy as jnp
import trimesh

# kernels
k_Gaussian = lambda x,params: params['k_alpha']/2*jnp.exp(-.5*jnp.sum(jnp.square(x)/params['k_sigma'],2))
r = lambda x,params: jnp.sqrt(1e-7+jnp.sum(jnp.square(x/params['k_sigma']),2))
def k_K0(x,params): 
    """ Laplace K0 kernel"""
    r_ = r(x,params)
    return params['k_alpha']*jnp.exp(-r_)
def k_K1(x,params): 
    """ Laplace K1 kernel"""
    r_ = r(x,params)
    return params['k_alpha']*2*(1+r_)*jnp.exp(-r_)
def k_K2(x,params): 
    """ Laplace K2 kernel"""
    r_ = r(x,params)
    return params['k_alpha']/2*4*(3+3*r_+r_**2)*jnp.exp(-r_)
def k_K3(x,params): 
    """ Laplace K3 kernel"""
    r_ = r(x,params)
    return params['k_alpha']*8*(15+15*r+6*r**2+r**3)*jnp.exp(-r)
def k_K4(x,params): 
    """ Laplace K4 kernel"""
    r_ = r(x,params)
    return params['k_alpha']*16*(105+105*r_+45*r_**2+10*r_**3+r_**4)*jnp.exp(-r_)

# 3d

# place points approximately uniformly on a sphere
def fibonacci_sphere(samples=1):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)

def mesh_sphere(target_vertices=100):
    # Start with a low number of subdivisions
    subdivisions = 0
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1)
    # Increase subdivisions until the number of vertices is at least the target
    while len(sphere.vertices) < target_vertices:
        subdivisions += 1
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1)
    # If we overshoot the target, reduce subdivisions
    while len(sphere.vertices) > target_vertices and subdivisions > 0:
        subdivisions -= 1
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1)
    return sphere
