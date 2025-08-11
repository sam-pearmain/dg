from jax import jit

from abc import ABC, abstractmethod
from enum import Enum
from jaxtyping import Array, Float64
from typing import List, Any, Tuple, Type, Protocol

from dg.physics.flux import ConvectiveNumericalFlux, DiffusiveNumericalFlux
from dg.physics.types import ConvectivePDE, DiffusivePDE, ConvectivePDEType, DiffusivePDEType

class PDE(ABC):
    """
    The physics abstract base class is the skeleton for any weak DG formulation of a PDE 
    in the form: ∂u/∂t + ∇F_conv(u) + ∇F_diff(u, ∇u) = S
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def n_dimensions(self) -> int:
        """Returns the number of physical dimensions of the system"""
        ...

    @property
    @abstractmethod
    def state_variables(self) -> Type[Enum]:
        """The state variables, u"""
        ...

    @property
    def n_state_variables(self) -> int:
        """Returns the number of state vars of the system"""
        return len(self.state_variables)
    
    def get_state_variable_names(self) -> List[str]:
        """Returns a list of the names of all the state variables"""
        return list(self.state_variables.__members__.keys())

    def get_state_variable_index(self, var_name: str) -> int:
        """Returns the index of a specific state variable"""
        return self.get_state_variable_names().index(var_name)
    
    def has_convective_terms(self) -> bool:
        return False
    
    def has_diffusive_terms(self) -> bool:
        return False

# -- convective terms -- 

class ConvectiveTerms(ABC):
    """
    A mixin class for PDEs with convective terms
    """
    _convective_numerical_flux: ConvectiveNumericalFlux

    def __init__(
            self, 
            convective_numerical_flux: ConvectiveNumericalFlux, 
            **kwargs: Any
        ) -> None:
        super().__init__(**kwargs)
        self._convective_numerical_flux = convective_numerical_flux

    @jit
    @abstractmethod
    def compute_convective_flux(
        self,
        u: Float64[Array, "n_q n_s"]
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the convective flux, F(u)"""
        ...

    def compute_convective_numerical_flux(
        self,
        physics: ConvectivePDEType,
        u_l: Float64[Array, "n_fq n_s"],
        u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the convective numerical flux at either inteior or boundary faces"""
        return self._convective_numerical_flux(
            physics, u_l, u_r, normals
        )

    def has_convective_terms(self) -> bool:
        return True

# -- diffusive terms --

class DiffusiveTerms(ABC):
    """
    A mixin class for PDEs with diffusive terms
    """
    _diffusive_numerical_flux: DiffusiveNumericalFlux

    def __init__(
            self, 
            diffusive_numerical_flux: DiffusiveNumericalFlux, 
            **kwargs: Any
        ) -> None:
        super().__init__(**kwargs)
        self._diffusive_numerical_flux = diffusive_numerical_flux

    @jit
    @abstractmethod
    def compute_diffusive_flux(
        self,
        u: Float64[Array, "n_q n_s"],
        grad_u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the diffusive flux, F_diff(u)"""
        ...

    def compute_diffusive_numerical_flux(
        self,
        physics: DiffusivePDEType,
        u_l: Float64[Array, "n_fq n_s"],
        u_r: Float64[Array, "n_fq n_s"],
        grad_u_l: Float64[Array, "n_fq n_s"],
        grad_u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the diffusive numerical flux at either inteior or boundary faces"""
        return self._diffusive_numerical_flux(
            physics, u_l, u_r, grad_u_l, grad_u_r, normals
        )

    def has_diffusive_terms(self) -> bool:
        return True

# -- tests --

def tests():
    import jax.numpy as jnp
    from enum import Enum, auto

    class DummyPhysics(ConvectivePDE):
        a: float

        @property
        def state_variables(self) -> Type[Enum]:
            class DummyStateVars(Enum):
                u = auto()
            return DummyStateVars
        
        @property
        def n_dimensions(self) -> int:
            return 1

        @jit
        def compute_convective_flux(self, u: Array) -> Array:
            return self.a * u
    
    class UpwindFlux(ConvectiveNumericalFlux[DummyPhysics]):
        def __call__(self, physics: DummyPhysics, u_l: Array, u_r: Array, normals: Array) -> Array:
            return u_l

    class LaxFriedrichsFlux(ConvectiveNumericalFlux):
        def __call__(
                self, 
                physics: DummyPhysics, 
                u_l: Array, 
                u_r: Array, 
                normals: Array
            ) -> Array:
            return 0.5 * (
                physics.compute_convective_flux(u_l) +
                physics.compute_convective_flux(u_r) -
                jnp.abs(physics.a * normals) * 2
            )

    erm = DummyPhysics(
        UpwindFlux(),
        a = 1.0
    )

    erm2 = DummyPhysics(
        LaxFriedrichsFlux(),
        a = 1.0
    )

if __name__ == "__main__":
    tests()