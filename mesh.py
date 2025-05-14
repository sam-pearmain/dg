import jax
import jax.numpy as jnp
import matplotlib as plt
from element import Segement, Triangle, Quadrilateral, ElementType

class Mesh():
    def __init__(self, nodes: jax.Array, connectivity: jax.Array):
        self.nodes = nodes
        self.connectivity = connectivity
        self.dimensions = nodes.shape[1]

        self.elements = []

    def get_n_nodes(self) -> int:
        return self.nodes.shape[0]
    
    def get_n_elements(self) -> int:
        return self.connectivity.shape[0]

    def construct_elements(self):
        if self.dimensions == 1:
            for idx, connection in enumerate(self.connectivity):
                self.elements.append(Segement(idx, connection))
        elif self.dimensions == 2:
            for idx, connection in enumerate(self.connectivity):
                self.elements.append(
                    Triangle(idx, connection) if connection.len() == 3 else Quadrilateral(idx, connection)
                )
        elif self.dimensions == 3:
            raise NotImplementedError
        else:
            raise ValueError("invalid dimensions")

    def plot(self):
        fig = plt.figure()
        fig.show()

# just a simple function to create a equally spaced 1D mesh
def _simple_1d(length: float, n_nodes: int) -> Mesh:
    n_elements = n_nodes - 1
    nodes = jnp.linspace(0, length, n_nodes).reshape((n_nodes, 1))
    idx_col1 = jnp.arange(0, n_elements)
    idx_col2 = jnp.arange(1, n_elements + 1)
    connectivity = jnp.stack([idx_col1, idx_col2], axis = 1)
    mesh = Mesh(nodes, connectivity)
    mesh.construct_elements()
    return mesh