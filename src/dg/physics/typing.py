from abc import abstractmethod
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Type, Tuple, TypeVar

from jax import jit
from jaxtyping import Array, Float64

from dg.utils.trait import Trait

class PDE(Trait):
    """The core PDE trait"""
    @property
    def n_dimensions(self) -> int: ...

    @property
    def state_variables(self) -> Type[Enum]: ...

    @property
    def n_state_variables(self) -> int: return len(self.state_variables)

    def tree_flatten(self) -> Tuple[List[Any], Dict[str, Any]]: ...

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Dict[str, Any], 
        children: List[Any]
    ) -> 'PDE': ...

class Convective(Trait):
    """The core trait for convective terms within a PDE"""
    @jit
    @abstractmethod
    def compute_diffusive_flux(
        self,
        u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the convective flux, F_conv(u)"""
        ...

class Diffusive(Trait):
    @abstractmethod
    def compute_diffusive_flux(
        self,
        u: Float64[Array, "n_q n_s"],
        grad_u: Float64[Array, "n_q n_s"],
    ) -> Float64[Array, "n_q n_s"]:
        """Computes the diffusive flux, F_diff(u)"""
        ...

C = TypeVar("C", bound = Convective, covariant = True)
D = TypeVar("D", bound = Diffusive,  covariant = True)

class ConvectiveNumericalFlux(Trait[C]):
    pass

def tests():
    # ===================================================================
    # 1. DEFINE THE TRAITS (The Contracts)
    # ===================================================================

    # A simple behavior
    class HasName(Trait):
        def get_name(self) -> str: ...

    # Another simple behavior
    class HasValue(Trait):
        def get_value(self) -> int: ...

    # A combined trait for anything that has both a name and a value
    class HasNameAndValue(HasName, HasValue):
        pass

    # ===================================================================
    # 2. DEFINE THE GENERIC OPERATOR (The Function)
    # ===================================================================

    # A TypeVar bound to our combined trait. This is the equivalent of
    # a Rust generic parameter `T: HasNameAndValue`.
    T = TypeVar('T', bound=HasNameAndValue, contravariant=True)

    # A generic trait for any 'thing' that can process an object of type T.
    class Processor(Trait[T]):
        def process(self, item_to_process: T) -> str: ...

    # ===================================================================
    # 3. CREATE CONCRETE IMPLEMENTATIONS
    # ===================================================================

    # A concrete data structure that fulfills the `HasNameAndValue` contract.
    # Note: It does NOT inherit from Processor. It is the DATA.
    class MyDataObject(HasName, HasValue):
        def get_name(self) -> str:
            return "Sam's Data"

        def get_value(self) -> int:
            return 42

    # A concrete implementation of the Processor. This is the BEHAVIOR.
    # It is specialized to work with `MyDataObject`.
    class MyDataProcessor(Processor[MyDataObject]): # type: ignore
        def process(self, item_to_process: MyDataObject) -> str:
            # It can call methods from both traits because the TypeVar `T`
            # was bound to `HasNameAndValue`.
            name = item_to_process.get_name()
            value = item_to_process.get_value()
            return f"Processing '{name}' which has a value of {value}."

    # ===================================================================
    # 4. EXECUTE THE TEST
    # ===================================================================

    # Create an instance of our data and our processor
    data_object = MyDataObject()
    processor = MyDataProcessor()

    # Use the processor to operate on the data
    result = processor.process(data_object)
    print(result)

    # --- Type Checking Proof ---
    # A type checker would allow this because MyDataObject fulfills the contract.
    # It would flag an error if you tried to pass an incompatible object.

if __name__ == "__main__":
    tests()
