from meshio import Mesh as MeshIOMesh
from meshio import CellBlock as MeshIOCellBlock
import matplotlib as plt
import jax.numpy as jnp
from jax import Array
from element import Segement, Triangle, Quadrilateral, ElementType, ELEMENTS

class Block():
    """A JAX-based representation of a block of elements"""
    def __init__(self, cell_block: MeshIOCellBlock):
        if cell_block.type not in ELEMENTS:
            raise NotImplementedError(f"cell type: {cell_block.type} not supported")
        self.element_type = ELEMENTS[cell_block.type]
        self.data = jnp.asarray(cell_block.data)

    def __len__(self):
        return len(self.data)

class Mesh():
    """A JAX-based mesh object"""
    def __init__(self, mesh: MeshIOMesh):
        self.nodes = jnp.asarray(mesh.points)

        
        