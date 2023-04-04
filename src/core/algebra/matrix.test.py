from unittest import TestCase
from random import randint
from .vector import Vector
from .matrix import Matrix

class TransposeTest(TestCase):
    def setUp(self) -> None:
        self.A = Matrix(Vector(randint(0, 10) for _ in range(10)) for _ in range(10))

        return super().setUp()  
    
    def check_property(self):
        B = self.A.transpose

        for i in range(self.A.m):
            for j in range(self.A.n):
                assert B[(j, i)] == self.A[(i, j)]


class MatrixSumTest(TestCase):

    def setUp(self):
        self.A = Matrix(Vector(randint(0, 10) for _ in range(10)) for _ in range(10))
        self.B = Matrix(Vector(randint(0, 10) for _ in range(10)) for _ in range(10))
        self.C = Matrix(Vector(randint(0, 10) for _ in range(10)) for _ in range(10))


    def test_commutativity(self):

        assert self.A + self.B == self.B + self.A

    def test_associativity(self):

        assert (self.A + self.B) + self.C == self.A + (self.B + self.C)

    def test_neutral_value(self):

        assert self.A + Matrix.null == self.A

    def test_transpose(self):
        assert (self.A + self.B).transpose == self.A.transpose + self.B.transpose


class MatrixMulTest(TestCase):

    def setUp(self):
        self.A = Matrix(Vector(randint(0, 10) for _ in range(10)) for _ in range(10))


    def test_associativity(self):

        assert (2 * 3) * self.A == 2 * (3 * self.A)

    def test_distributary(self):

        assert self.A * (2 + 3) == 2 * self.A + 3 * self.A

    def test_transpose(self):
        assert (self.A * 2).transpose == self.A.transpose * 2