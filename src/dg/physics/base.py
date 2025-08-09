from jax import jit

from abc import ABC, abstractmethod
from enum import Enum
from jaxtyping import Array, Float64
from typing import List, Any, Type

from dg.physics.flux import ConvectiveNumericalFlux

class Physics(ABC):
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

# -- convective flux -- 

class ConvectiveTerms(ABC):
    """
    Convective flux terms within the governing equations
    """
    _convective_numerical_flux_function: ConvectiveNumericalFlux

    def __init__(
            self, 
            conv_num_flux_func: ConvectiveNumericalFlux
        ) -> None:
        super().__init__()
        self._convective_numerical_flux_function = conv_num_flux_func

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
        physics: Physics,
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
    
    class Upwind(ConvectiveNumericalFlux):
        def compute_convective_numerical_flux(
                self, 
                physics: Physics, 
                u_l: Array, 
                u_r: Array, 
                normals: Array
            ) -> Array:
            return jnp.asarray(1)

    erm = DummyPhysics(a = 1.0)

if __name__ == "__main__":
    tests()