from .vector import Vector
from typing import TypeVar, List, Generic, Iterator
import math

T = TypeVar('T')
R = TypeVar('R')

class Matrix(Generic[T]):
    def __init__(self, *vectors: List[Vector[T]]) -> None:
        self.columns = list(vectors)
        self.n = len(self.columns)
        self.m = len(self.columns[0])

    def __getindex__(self, index):
        i, j = index
        return self.columns[j][i]
    
    @staticmethod
    def null(n : int, m: int) -> 'Matrix[int]':
        return Matrix(Vector(0 for _ in range(m)) for _ in range(n))
    
    @staticmethod
    def I(n : int, m: int) -> 'Matrix[int]':
        return Matrix(Vector(int(i == j) for i in range(m)) for j in range(n))
    
    @property
    def is_superior_triangle(self) -> bool:
        for i in range(self.m):
            for j in range(self.n):
                if i > j and self[(i, j)] != 0:
                    return False
                
        return True
    
    @property
    def is_inferior_triangle(self) -> bool:
        for i in range(self.m):
            for j in range(self.n):
                if i < j and self[(i, j)] != 0:
                    return False
                
        return True
    
    @property
    def is_diagonal(self) -> bool:
        for i in range(self.m):
            for j in range(self.n):
                if i != j and self[(i, j)] != 0:
                    return False           
                
        return True
    
    @property
    def transpose(self) -> 'Matrix[T]':
        return Matrix(Vector(c[i] for c in self.columns) for i in range(self.m))
    
    def __add__(self, other: 'Matrix[T]') -> 'Matrix[T]':
        assert self.n == other.n and self.m == other.m

        return Matrix(
            Vector(a + b for a, b in zip(self.columns[i], other.columns[i])) 
            for i in range(self.n)    
        )

    def __eq__(self, other: 'Matrix[T]') -> bool:
        assert self.n == other.n and self.m == other.m

        for i in range(self.m):
            for j in range(self.n):
                if self[(i, j)] != self[i,j]:
                    return False
                
        return True
    
    def __mul__(self, k: float | int) -> 'Matrix[T]':
        return Matrix(Vector(self[(i, j)] * k for i in range(self.m)) for j in range(self.n))
    
    def __rmul__(self, k: float | int) -> 'Matrix[T]':
        return self.__mul__(k)
    
    def __xor__(self, other: 'Vector[T]') -> T:

        assert isinstance(other, Vector)
        components = []
        for j in range(self.m):
            c = 0
            for i in range(self.n):
                c += other[i] * self[i, j]

            components.append(c)

        return Vector(components)


    def components(self) -> Iterator[T]:
        for i in range(self.m):
            for j in range(self.n): 
                yield self[i, j]


def norm_frobenius(m: Matrix[T]) ->  float:
    result = 0
    for c in m.components:
        result += c * c

    return math.sqrt(result)
