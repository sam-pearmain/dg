import meshio
from mesh import Mesh

points = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]
cells = [
    ("triangle", [[0, 1, 2], [1, 3, 2]])
]

mesh = meshio.Mesh(points, cells)
jax_mesh = Mesh(mesh)

print(jax_mesh.dimensions, jax_mesh.element_type)
print(jax_mesh.nodes)
print(jax_mesh.connectivity)

jax_mesh.write("jax_mesh.vtk")