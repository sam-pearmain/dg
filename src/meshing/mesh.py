import jax.numpy as jnp
import numpy as np
import meshio

from jax import Array
from typing import Optional, Dict, Union
from meshio import Mesh as MeshIOMesh
from utils import todo, Uninit
from meshing.interface import BoundaryType, SUPPORTED_BOUNDARIES
from meshing.element import ElementType, SUPPORTED_ELEMENTS

class Mesh():
    """A JAX-based geometric mesh object"""
    # todo: perhaps we want to have this be aa generic class with both 
    # hp-adaptive meshes and static meshes built on top 
    nodes:         Array
    connectivity:  Array
    boundaries:    Dict[BoundaryType, Array]
    groups:        Dict[str, Array]
    element_type:  ElementType
    element_order: Union[Dict[int, Array], Uninit]

    def __init__(self, mesh: MeshIOMesh, element_type: ElementType | str):
        if isinstance(element_type, str):
            try:
                element_type = SUPPORTED_ELEMENTS[element_type]
            except KeyError:
                raise NotImplementedError(f"{element_type} not supported")
        
        element_types = {cell_block.type for cell_block in mesh.cells}

        if not element_types:
            raise ValueError("input mesh has no defined elements")
                
        self.boundaries = {}

        connectivity_arrays = []
        for cell_block in mesh.cells:
            if cell_block.type in SUPPORTED_ELEMENTS:
                if element_type == SUPPORTED_ELEMENTS[cell_block.type]:
                    connectivity_arrays.append(
                        jnp.asarray(cell_block.data, dtype = jnp.int32)
                    )

                if element_type.face_type == SUPPORTED_ELEMENTS[cell_block.type]:
                    # since the boundaries will always be defined by the face type
                    # for our chosen mesh element
                    pass
                    

        self.nodes = jnp.asarray(mesh.points, dtype = jnp.float64)
        self.connectivity = jnp.asarray(
            jnp.concatenate(connectivity_arrays),
            dtype = jnp.int32
        )
        self.element_type = element_type
        self.element_order = Uninit

    def __repr__(self):        
        return (
            f"<Mesh object with {self.n_nodes} nodes, {self.n_elements} elements>\n"
            f" - ElementType: {self.element_type}\n"
            f" - Dimensions:  {self.dimensions}\n"
            f" - Coordinate Span: {self.bounds}"
        )

    @classmethod
    def read(cls, filepath: str, element_type: ElementType | str, file_format: Optional[str] = None):
        """Reads a mesh from a file using meshio and returns a Mesh object"""
        meshio_mesh = meshio.read(filepath, file_format)
        return cls(meshio_mesh, element_type)

    def write(self, filepath: str, file_format = None):
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

    def initialise_element_order(self, order: int):
        """Initialises the dict that stores the local order of each element, {order : elem_ids}"""
        element_ids = jnp.arange(self.n_elements, dtype = jnp.int32)
        self.element_order = {order : element_ids}

    @property
    def n_dofs(self) -> Optional[int]:
        if isinstance(self.element_order, Uninit):
            todo("how do we handle this, either return None or raise an error, idk")
        
        n_dofs = 0
        for order, element_ids in self.element_order.items():
            n_dofs += element_ids.shape[0] * self.element_type.n_dofs(order)
            
        return n_dofs

    @property
    def n_nodes(self) -> int:
        """Returns the number of nodes within the mesh"""
        return self.nodes.shape[0]
    
    @property
    def n_elements(self) -> int:
        """Returns the number of elements within the mesh"""
        return self.connectivity.shape[0]
    
    @property
    def dimensions(self) -> int:
        """Returns the dimensions of the mesh"""
        return self.element_type.dimensions()
    
    @property
    def bounds(self):
        """Returns the max/min dimensional bounds of the mesh"""
        min_coords = jnp.min(self.nodes, axis = 0)
        max_coords = jnp.max(self.nodes, axis = 0)

        if self.dimensions == 1:
            return (
                (min_coords[0].item(), max_coords[0].item())
            )
        elif self.dimensions == 2:
            return (
                (min_coords[0].item(), max_coords[0].item()),
                (min_coords[1].item(), max_coords[1].item())
            )
        elif self.dimensions == 3:
            return (
                (min_coords[0].item(), max_coords[0].item()),
                (min_coords[1].item(), max_coords[1].item()),
                (min_coords[2].item(), max_coords[2].item())
            )
        else:
            raise ValueError("not sure if this will ever occur")
        