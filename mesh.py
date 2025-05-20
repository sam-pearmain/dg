import numpy as np
import jax.numpy as jnp
import meshio
from meshio import Mesh as MeshIOMesh
from element import SUPPORTED_ELEMENTS

class Mesh():
    """A JAX-based mesh object"""
    def __init__(self, mesh: MeshIOMesh):
        element_types = {cell_block.type for cell_block in mesh.cells}

        if not element_types:
            raise ValueError("input mesh has no defined elements")

        if len(element_types) > 1:
            raise ValueError("only single element-type meshes are supported")
        
        meshio_cell_type_str = list(element_types)[0]
        
        if meshio_cell_type_str not in SUPPORTED_ELEMENTS:
            raise NotImplementedError(f"element type '{meshio_cell_type_str}' not supported")
        
        connectivity_arrays = []
        for cell_block in mesh.cells:
            connectivity_arrays.append(jnp.asarray(cell_block.data, dtype = jnp.int32))

        self.nodes = jnp.asarray(mesh.points, dtype = jnp.float64)
        self.connectivity = jnp.asarray(
            jnp.concatenate(connectivity_arrays),
            dtype = jnp.int32
        )
        self.element_type = SUPPORTED_ELEMENTS[meshio_cell_type_str]
        self.dimensions = self.element_type.dimensions()

    @classmethod
    def read(cls, filepath: str):
        """Reads a mesh from a file using meshio and returns a Mesh object"""
        meshio_mesh = meshio.read(filepath)
        return cls(meshio_mesh)

    def write(self, filepath: str, file_format=None):
        """Writes the mesh to a given filepath using the meshio API"""
        cell_type = None
        for key, value in SUPPORTED_ELEMENTS.items():
            if value == self.element_type:
                cell_type = key
                break 
        
        if cell_type is None:
            raise ValueError(f"could not find a corresponding meshio cell type for {self.element_type}")
        
        points = np.asarray(self.nodes)
        cells = [(cell_type, np.asarray(self.connectivity))]
        
        meshio_mesh = MeshIOMesh(points, cells)
        meshio_mesh.write(filepath, file_format)