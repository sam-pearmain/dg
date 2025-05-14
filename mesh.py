import jax
import jax.numpy as jnp
import matplotlib as plt
from element import Element, ElementType

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
        ## this needs to be changed, should instead be supplied with connectivity and nodes and should construct the elements based off that information ##
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