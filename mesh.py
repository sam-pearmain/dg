from meshio import Mesh as MeshIOMesh
from meshio import CellBlock as MeshIOCellBlock
import jax.numpy as jnp
from element import Segement, Triangle, Quadrilateral, ElementType, SUPPORTED_ELEMENTS

class Block():
    """A JAX-based representation of a block of elements"""
    def __init__(self, cell_block: MeshIOCellBlock):
        if cell_block.type not in SUPPORTED_ELEMENTS:
            raise NotImplementedError(f"cell type: {cell_block.type} not supported")
        self.element_type = SUPPORTED_ELEMENTS[cell_block.type]
        self.data = jnp.asarray(cell_block.data)

    def __len__(self):
        return len(self.data)

class Mesh():
    """A JAX-based mesh object"""
    def __init__(self, mesh: MeshIOMesh):
        element_types = {cell_block.type for cell_block in mesh.cells}
        if element_types > 1:
            raise ValueError("jax mesh only supports single element-type meshes")
        
        self.nodes = jnp.asarray(mesh.points, dtype = jnp.float64)
        self.connectivity = jnp.asarray(
            jnp.concatenate([cell_block.data for cell_block in mesh.cells]),
            dtype = jnp.int32
        )

        self.element_type = SUPPORTED_ELEMENTS[element_types[0]]
        self.dimensions = self.element_type.dimensions()

        
        