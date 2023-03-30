from unittest import TestCase
from random import randint
from .vector import Vector
from .lineal_functions import DotAsLinealFunction


class LinealFunctionTest(TestCase):
    """
    This class is a unit test about the properties of lineal functions.
    """

    def setUp(self):
        self.v = Vector([randint(1, 100) for _ in range(10)])

    def op(self, v):
        """Each norm should override this function and call is function"""

    def test_homogeneity(self):
        assert self.op(2 * self.v) == 2 * self.op(self.v)

    def test_triangular_unequally(self):
        assert self.op(self.v + self.w) <= self.op(self.v) + self.op(self.w)


class DotAsFunctionTest(LinealFunctionTest):
    def setUp(self):
        super().setUp()
        self.f = DotAsLinealFunction(
            Vector([randint(1, 100) for _ in range(10)])
        )

    def op(self, v):
        return self.f(v)
