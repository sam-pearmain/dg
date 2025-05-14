import meshio
import matplotlib as plt
import jax.numpy as jnp
from jax import Array
from element import Segement, Triangle, Quadrilateral, ElementType

class Mesh():
    """A JAX initialisation of a mesh"""
    def __init__(self, mesh: meshio.Mesh):
        self.nodes      = jnp.array(mesh.points, dtype = jnp.float64)
        self.elements   = jnp.array(mesh.cells, dtype = jnp.int32)
        self.dimensions = self.nodes.shape[1]

    def n_nodes(self) -> int:
        """Returns the number of nodes in the mesh"""
        return self.nodes.shape[0]
    
    def n_elements(self) -> int:
        """Returns the number of elements in the mesh"""
        return self.connectivity.shape[0]


def simple_1d(length: float, n_nodes: int) -> Mesh:
    n_elements = n_nodes - 1
    nodes = jnp.linspace(0, length, n_nodes).reshape((n_nodes, 1))

    idx_col1 = jnp.arange(0, n_elements)
    idx_col2 = jnp.arange(1, n_elements + 1)
    
    connectivity = jnp.stack([idx_col1, idx_col2], axis = 1)
    mesh = Mesh(nodes, connectivity)
    return mesh

def simple_2d_rect(length: float, height: float, nx: int, ny: int) -> Mesh:
    x_coords = jnp.linspace(0, length, nx)
    y_coords = jnp.linspace(0, height, ny)

    xx, yy = jnp.meshgrid(x_coords, y_coords)
    nodes = jnp.stack([xx.ravel(), yy.ravel()], axis = 1)

    connectivity_list = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i # bottom left
            n1 = j * nx + (i + 1) # bottom right
            n2 = (j + 1) * nx + i # top left
            n3 = (j + 1) * nx + (i + 1) # top right
            connectivity_list.append([n0, n1, n2, n3])

    connectivity = jnp.array(connectivity_list, dtype = int)
    mesh = Mesh(nodes, connectivity)
    return mesh