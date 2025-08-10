from jax import jit

from abc import ABC, abstractmethod
from enum import Enum
from jaxtyping import Array, Float64
from typing import List, Any, Tuple, Type

from dg.physics.flux import ConvectiveNumericalFlux, DiffusiveNumericalFlux
from dg.physics.types import ConvectivePhysics, DiffusivePhysics, ConvectiveDiffusivePhysics

class Physics(ABC):
    """
    The physics abstract base class is the skeleton for any weak DG formulation of a PDE 
    in the form: ∂u/∂t + ∇F_conv(u) + ∇ F_diff(u, ∇u) = S
    """
    def __init__(self, **physical_constants: Any) -> None:
        super().__init__()
        for key, value in physical_constants.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def state_variables(self) -> Type[Enum]:
        """The state variables, u"""
        ...

    @property
    @abstractmethod
    def n_dimensions(self) -> int:
        """Returns the number of physical dimensions of the system"""
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
    Convective flux terms within the governing equations
    """
    _convective_numerical_flux_function: ConvectiveNumericalFlux

    def __init__(
            self, 
            convective_numerical_flux: ConvectiveNumericalFlux, 
            **kwargs: Any
        ) -> None:
        super().__init__(**kwargs)
        self._convective_numerical_flux_function = convective_numerical_flux

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
        physics: ConvectivePhysics,
        u_l: Float64[Array, "n_fq n_s"],
        u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the convective numerical flux at either inteior or boundary faces"""
        return self._convective_numerical_flux_function(
            physics, u_l, u_r, normals
        )

    def has_convective_terms(self) -> bool:
        return True

# -- diffusive terms --

class DiffusiveTerms(ABC):
    """
    Diffusive flux terms within the governing equations
    """
    _diffusive_numerical_flux_function: DiffusiveNumericalFlux

    def __init__(
            self, 
            diffusive_numerical_flux: DiffusiveNumericalFlux, 
            **kwargs: Any
        ) -> None:
        super().__init__(**kwargs)
        self._diffusive_numerical_flux_function = diffusive_numerical_flux

    @jit
    @abstractmethod
    def compute_diffusive_flux(
        self,
        u: Float64[Array, "n_q n_s"]
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the diffusive flux, F(u)"""
        ...

    def compute_diffusive_numerical_flux(
        self,
        physics: DiffusivePhysics,
        u_l: Float64[Array, "n_fq n_s"],
        u_r: Float64[Array, "n_fq n_s"],
        grad_u_l: Float64[Array, "n_fq n_s"],
        grad_u_r: Float64[Array, "n_fq n_s"],
        normals: Float64[Array, "n_fq n_d"]
    ) -> Float64[Array, "n_fq n_s"]:
        """Computes the diffusive numerical flux at either inteior or boundary faces"""
        return self._diffusive_numerical_flux_function(
            physics, u_l, u_r, grad_u_l, grad_u_r, normals
        )

    def has_diffusive_terms(self) -> bool:
        return True

# -- tests --

def tests():
    import jax.numpy as jnp
    from enum import Enum, auto
    from dg.physics.base import ConvectiveTerms, Physics

    class DummyPhysics(ConvectiveTerms, Physics):
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
    
    class UpwindFlux(ConvectiveNumericalFlux):
        def compute_convective_numerical_flux(
                self, 
                physics: ConvectivePhysics, 
                u_l: Array, 
                u_r: Array, 
                normals: Array
            ) -> Array:
            return jnp.asarray(1)
        
        def compatible_physics_types(self) -> Tuple[type[Physics], ...]:
            return (DummyPhysics,)

    class LaxFriedrichsFlux(ConvectiveNumericalFlux):
        def compute_convective_numerical_flux(
                self, 
                physics: ConvectivePhysics, 
                u_l: Array, 
                u_r: Array, 
                normals: Array
            ) -> Array:
            return 0.5 * (physics.compute_convective_flux(u_l))

    erm = DummyPhysics(
        UpwindFlux(),
        a = 1.0
    )

if __name__ == "__main__":
    tests()