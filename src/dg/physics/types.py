from typing import TypeVar, Protocol, Generic
from jaxtyping import Array, Float64
from dg.physics.base import PDE, ConvectiveTerms, DiffusiveTerms

class ConvectivePDE(ConvectiveTerms, PDE):
    """A generic type for convective PDEs"""
    pass

class DiffusivePDE(DiffusiveTerms, PDE):
    """A generic type for diffusive PDEs"""
    pass

class ConvectiveDiffusivePDE(ConvectiveTerms, DiffusiveTerms, PDE):
    """A generic type for convection-diffusion PDEs"""
    pass

ConvectivePDEType = TypeVar(
    "ConvectivePDEType", 
    bound = ConvectivePDE, 
    contravariant = True,
)

DiffusivePDEType = TypeVar(
    "DiffusivePDEType", 
    bound = DiffusivePDE,
    contravariant = True,   
)
ConvectiveDiffusivePDEType = TypeVar(
    "ConvectiveDiffusivePDEType", 
    bound = ConvectiveDiffusivePDE, 
    contravariant = True    
)
