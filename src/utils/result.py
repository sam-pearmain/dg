from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar("T")
E = TypeVar("E")

class Result(Generic[T, E]):
    """A shameless copy of Rust's Result type"""
    def is_ok(self) -> bool: ...
    def is_err(self) -> bool: ...
    def unwrap(self) -> T: ...
    def unwrap_or(self, default: T) -> T: ...

@dataclass
class Ok(Result[T, E]):
    value: T

    def is_ok(self) -> bool: 
        return True
    
    def is_err(self) -> bool: 
        return False
    
    def unwrap(self) -> T: 
        return self.value
    
    def unwrap_or(self, default: T) -> T: 
        return self.value
    
@dataclass
class Err(Result[T, E]):
    error: E

    def is_ok(self) -> bool: 
        return False
    
    def is_err(self) -> bool: 
        return True
    
    def unwrap(self) -> T: 
        if isinstance(self.error, Exception):
            raise self.error
        raise RuntimeError(self.error)
    
    def unwrap_or(self, default: T) -> T: 
        return default