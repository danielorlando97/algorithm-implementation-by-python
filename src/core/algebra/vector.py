from typing import Generic, TypeVar, Iterable, Iterator, Union, List
from math import sqrt
from functools import singledispatchmethod

T = TypeVar('T')


class Vector(Generic[T]):

    """
    The vector class is a representation of the 
    main operations of vector algebra. 

    It is also the input type for many of the 
    project algorithms. 

    This class is fully compatible with the 
    functools library and its caching functions.
    """

    def __init__(self, components: Iterable[T]) -> None:
        """
        A vector is constructed from any iterator. 
        From this iterator we create an immutable 
        tuple that represents our vector and each of 
        its components.
        """

        super().__init__()

        self.components = tuple(components)
        self.size = len(self.components)

    @staticmethod
    def neutral_value(dim):
        return Vector([0 for _ in range(dim)])

    def __getitem__(self, index: int) -> T:
        return self.components[index]

    def __len__(self):
        return self.size

    def __hash__(self) -> int:
        return hash(self.components)

    def __iter__(self) -> Iterator[T]:
        return self.components.__iter__()

    def __eq__(self, __value: object) -> bool:
        """
        Two vectors x and y in R_n are equals if 
        each tuple of components x_i and y_i are equals,
        such as i <= n

        ```python
        >>> Vector([1,2,3]) == Vector([1,2,3])
        >>> True
        >>> Vector([1,2,3]) == Vector([1,4,3])
        >>> False
        ```
        """

        if isinstance(__value, Vector):
            return self.size == __value.size and self.components == __value.components
        if isinstance(__value, Iterable):
            return self.components == tuple(__value)

        return False

    ###################################################################################
    #                                                                                 #
    #                                                                                 #
    #                                                                                 #
    #                      Properties                                                 #
    #                  - is canonic                                                   #
    #                  - is an affine lineal comb                                     #
    #                  - is an convex lineal comb                                     #
    #                                                                                 #
    #                                                                                 #
    ###################################################################################

    @property
    def is_canonic(self):
        """
        A vector is canonic if it has only one component different than 0 
        and this component is 1

                ```python
        >>> Vector([1,0,1]).is_canonic
        >>> False
        >>> Vector([1,0,0]).is_canonic
        >>> True
        ```
        """

        ones = [c for c in self.components if c != 0]
        return len(ones) == 1 and ones[1] == 1

    # A lineal combination is a equation where there are
    # a vector of variables and a vector of coefficients.
    # In addition, the vector of coefficients is
    # a lineal combination of the vector of variables

    @property
    def is_an_affine_lineal_comb(self):
        """
        A vector can be a lineal compilation of other vector.
        In that case, this vector is a affine lineal combination if 
        the sum of all its components is equal 1 

        ```python
        >>> Vector([1,0,1]).is_an_affine_lineal_comb
        >>> False
        >>> Vector([1,0,0]).is_an_affine_lineal_comb
        >>> True
        ```
        """

        return sum(self.components) == 1

    @property
    def is_a_convex_combination(self):
        """
        A vector can be a lineal compilation of other vector.
        In that case, this vector is a affine lineal combination if 
        the sum of all its components is equal 1 

        ```python
        >>> Vector([1,0,1]).is_a_convex_combination
        >>> False
        >>> Vector([1,0,0]).is_a_convex_combination
        >>> True
        >>> Vector([1,1,-1]).is_a_convex_combination
        >>> False
        ```
        """

        s = 0
        for item in self.components:
            if item < 0:
                return False
            s += item

        return s == 1

    ###################################################################################
    #                                                                                 #
    #                                                                                 #
    #                                                                                 #
    #                      Operations                                                 #
    #                  - add between vectors                                          #
    #                  - sub between vectors                                          #
    #                  - mul with real vector                                         #
    #                  - intern product between vector                                #
    #                  - Euclidean Norm                                               #
    #                  - Root Mean Square                                             #
    #                  - Average                                                      #
    #                  - Standard Deviation                                           #
    #                                                                                 #
    #                                                                                 #
    ###################################################################################

    def __add__(self, other: 'Vector[T]') -> 'Vector[T]':
        """
        The add operation between two or more vector is defined like:
        Let's x and y vector in R_n then:
        x + y = Vector([x_1 + y_1, x_2 + y_2, ...., x_n + y_n])

        ```python
        >>> Vector([1,2,3]) + Vector([1,2,3])
        >>> Vector([2,4,6])
        ```
        """

        return Vector(map(lambda x: x[0] + x[1], zip(self.components, other)))

    def __sub__(self, other: 'Vector[T]') -> 'Vector[T]':
        """
        The sub operation between two or more vector is defined like:
        Let's x and y vector in R_n then:
        x - y = Vector([x_1 - y_1, x_2 - y_2, ...., x_n - y_n])

        ```python
        >>> Vector([1,2,3]) - Vector([1,2,3])
        >>> Vector([0,0,0])
        ```
        """
        return Vector(map(lambda x: x[0] - x[1], zip(self, other)))

    def __mul__(self, other: float) -> 'Vector':
        """
        There are some kind of multiply operations in the vectorial spaces. But 
        the most simple is the multiply for a real value. This operation is defined like:
        Let's x a vector in R_n and k a value in R 
        x * k = Vector([x_1 * k, x_2 * k, ...., x_n * k])

        ```python
        >>> Vector([1,2,3]) * 2
        >>> Vector([2,4,6])
        ```
        """

        return Vector(map(lambda x: x * other, self.components))

    def __xor__(self, other: 'Vector[T]') -> T:
        """
        The dot operator is defined like:
        Let's x and y vector from R_n:

        (x * y) = x_T y = x_1 * y_1 + x_2 * y_2 + ... x_n * y_n
        So, how sometimes this operation is written like 
        the x's transpose per y, then it has selected the operator ^ 
        to express x's transpose per
        """

        assert isinstance(other, Vector)
        return sum(map(lambda x: x[0] * x[1], zip(self, other)))

    def euclidean_norm(self):
        """Traditional distance"""

        return sqrt(self ^ self)

    def rms(self):
        """root mean square"""

        return sqrt((self ^ self)/self.size)

    def avg(self):

        return sum(self.components)/self.size

    def std(self):

        return Vector(map(lambda x: x - self.avg(), self.components)).rms()

    def is_orthogonal_with(self, x: 'Vector[T]') -> bool:
        return self ^ x == 0

    ###################################################################################
    #                                                                                 #
    #                                                                                 #
    #                                                                                 #
    #                      Tools                                                      #
    #               - Euclidean Distance                                              #
    #               - Euclidean Root Mean Square Deviation                            #
    #                                                                                 #
    #                                                                                 #
    ###################################################################################


def euclidean_distance(x: Vector, y: Vector) -> float:
    return (x - y).euclidean_norm()


def euclidean_rms_deviation(x: Vector, y: Vector) -> float:
    return (x - y).rms()
