from typing import Generic, TypeVar

T = TypeVar('T')

class PhysicalConstant(Generic[T]):
    def __init__(self, value: T) -> None:
        self._value = value

    def __get__(self, instance, owner) -> T:
        return self._value

    def __set__(self, instance, owner):
        raise AttributeError("Cannot mutate a PhysicalConstant")

def tests():
    import timeit

    setup_descriptor = """
from __main__ import PhysicalConstant

class Constants:
    a = PhysicalConstant(1.2)
    b = PhysicalConstant(2.4)

constants = Constants()
    """

    setup_plain = """
class Constants2:
    a = 1.2
    b = 2.4

constants = Constants2()
    """

    statement = "1 - constants.a * constants.b"
    number_of_executions = 1_000_000

    time_descriptor = timeit.timeit(
        stmt=statement, 
        setup=setup_descriptor, 
        number=number_of_executions
    )

    time_plain = timeit.timeit(
        stmt=statement, 
        setup=setup_plain, 
        number=number_of_executions
    )

    print(f"--- Benchmark Results ({number_of_executions:,} executions) ---\n")
    print(f"Using PhysicalConstant descriptor: {time_descriptor:.4f} seconds")
    print(f"Using plain class attributes:      {time_plain:.4f} seconds")
    print("-" * 45)

    if time_plain > 0:
        overhead_factor = time_descriptor / time_plain
        print(f"\nThe descriptor-based approach is {overhead_factor:.2f}x slower.")
    else:
        print("\nPlain attribute access was too fast to measure accurately.")


if __name__ == "__main__":
    tests()