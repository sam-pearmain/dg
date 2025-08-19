from typing import TypeVar
from jaxtyping import Array, Float64

from dg.physics.base import (
    Convective, Diffusive, ConvectiveFlux, DiffusiveFlux, 
    ConvectiveNumericalFlux, DiffusiveNumericalFlux, 
    ConvectivePDETrait, DiffusivePDETrait
)

C = TypeVar('C', bound = ConvectivePDETrait)
class ConvectiveTerms(ConvectiveFlux[C], ConvectiveNumericalFlux[C], Trait[C]):
    def __init__(self) -> None:
        super().__init__()