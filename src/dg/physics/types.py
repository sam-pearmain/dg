from typing import TypeVar, Self

from dg.physics.pde import PDE
from dg.physics.flux import Convective, Diffusive, Flux

# -- fluxes -- #

P = TypeVar('P', bound = PDE)
class ConvectiveFlux(Convective[P], Flux[P]): ... # type: ignore

P = TypeVar('P', bound = PDE)
class DiffusiveFlux(Diffusive[P], Flux[P]): ... # type: ignore 

P = TypeVar('P', bound = PDE)
class ConvectiveDiffusiveFlux(Convective[P], Diffusive[P], Flux[P]): ... # type: ignore

# -- pdes -- #

F = TypeVar('F', bound = ConvectiveFlux)
class ConvectivePDE(PDE[F]):
    flux: F

F = TypeVar('F', bound = DiffusiveFlux)
class DiffusivePDE(PDE[F]):
    flux: F

F = TypeVar('F', bound = ConvectiveDiffusiveFlux)
class ConvectiveDiffusivePDE(PDE[F]):
    flux: F