# Shape plots

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2d
def plot_shape_2d(q,color=None,ax=None,label=None):
    d = 2
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    q = q.reshape((-1,d))
    ax.plot(q[:,0],q[:,1],'.',color=color,label=label)
    ax.axis('equal')
    return ax

# 3d
def plot_shape_3d(points,ax=None):
    d = 3
    points = points.reshape((-1,3))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax

def plot_mesh(mesh,ax=None):
    # Plot the mesh
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color='b', edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax
