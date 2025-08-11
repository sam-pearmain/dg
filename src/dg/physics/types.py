from typing import TypeVar, Protocol
from jaxtyping import Float64, Array

from dg.physics.base import Physics, ConvectiveTerms, DiffusiveTerms

ConvectivePhysics = TypeVar("ConvectivePhysics", ConvectiveTerms, Physics)