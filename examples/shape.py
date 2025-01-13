# Shape related utiliti functions

import numpy as np
import trimesh

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
