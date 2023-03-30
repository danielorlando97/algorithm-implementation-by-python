from .vector import Vector
from typing import Callable, Generic, TypeVar
from abc import ABC, abstractclassmethod
T = TypeVar('T')


class LinealFunction(Callable, Generic[T], ABC):
    """
    Lineal functions are a function that defined from R_n to R.
    So, f is a lineal function if:
        f(x) = y | x in R_n and y in R
    """

    @abstractclassmethod
    def __call__(self, x: Vector[T]) -> T:
        """"""


class DotAsLinealFunction(LinealFunction):
    def __init__(self, a: Vector[T]) -> None:
        super().__init__()

        self.coefficients = a

    def __call__(self, x: Vector[T]) -> T:
        return self.coefficients ^ x


class AffineLinealFunction(DotAsLinealFunction):
    """
    A function is a affine function when it can be written as:
        f(x) = a ^ x + b | a in R_n and b in R
    """

    def __init__(self, a: Vector[T], b=0) -> None:
        super().__init__(a)
        self.b = b

    def __call__(self, x: Vector[T]) -> T:
        return super().__call__(x) + self.b
