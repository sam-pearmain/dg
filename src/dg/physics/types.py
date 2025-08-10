from typing import TypeVar, Annotated
from dg.physics.base import Physics, ConvectiveTerms, DiffusiveTerms

ConvectivePhysics = TypeVar(
    "ConvectivePhysics", 
    Physics,
    ConvectiveTerms, 
)

DiffusivePhysics = TypeVar(
    "DiffusivePhysics", 
    Physics,
    DiffusiveTerms, 
)

ConvectiveDiffusivePhysics = TypeVar(
    "ConvectiveDiffusivePhysics", 
    Physics,
    ConvectiveTerms, 
    DiffusiveTerms,
)