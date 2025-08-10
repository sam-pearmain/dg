from typing import TypeVar, Annotated
from dg.physics.base import Physics, ConvectiveTerms, DiffusiveTerms

_ConvectivePhysics = TypeVar(
    "_ConvectivePhysics", 
    ConvectiveTerms, 
    Physics
)

_DiffusivePhysics = TypeVar(
    "_DiffusivePhysics", 
    DiffusiveTerms, 
    Physics
)

_ConvectiveDiffusivePhysics = TypeVar(
    "_ConvectiveDiffusivePhysics", 
    ConvectiveTerms, 
    DiffusiveTerms
)