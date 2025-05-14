import jax.numpy as jnp
import matplotlib as plt
from element import Element, ElementType

class Mesh():
    def __init__(self, dimensions: int, n_nodes: int, e_type: ElementType):
        self.dimensions = dimensions
        self.nodes = jnp.zeros((n_nodes, dimensions), dtype = jnp.float64)
        self.connectivity = jnp.zeros(())
        
        self.elements = []
        self.element_type = e_type

    def get_n_nodes(self) -> int:
        return self.nodes.shape[0]

    def construct_elements(self):
        if self.dimensions == 1:
            for i in range(self.get_n_nodes() - 1):
                node_indices = jnp.array([i, i + 1])
                element = Element(i, ElementType.Segment)
                element.node_ids = node_indices
                self.elements.append(element)

    def plot(self):
        fig = plt.figure()
        fig.show()

# just a simple function to create a equally spaced 1D mesh
def _simple_1d(length: float, n_nodes: int) -> Mesh:
    mesh = Mesh(dimensions = 1, n_nodes = n_nodes, e_type = ElementType.Segment)
    node_coords = jnp.linspace(0, length, n_nodes)
    mesh.nodes = node_coords.reshape((n_nodes, 1))
    mesh.construct_elements()
    return mesh