from unittest import TestCase
from random import randint
from .vector import Vector


class VectorialSpaceTest(TestCase):
    """
    This class is a unit test about the properties of a vectorial space.
    So, if some vector implementation pass this test means that this class 
    is a vectorial spaces
    """

    def setUp(self):
        self.v = Vector([randint(1, 100) for _ in range(10)])
        self.w = Vector([randint(1, 100) for _ in range(10)])
        self.z = Vector([randint(1, 100) for _ in range(10)])

        self.neutral_value = Vector.neutral_value(10)

    def test_commutativity(self):
        """
        Every vector space have to be commutative with respect to sum
        """

        assert self.w + self.v == self.v + self.w

    def test_associativity(self):
        """
        Every vector space have to be associativity with respect to sum
        """

        assert (self.w + self.v) + self.z == self.w + (self.v + self.z)

    def test_neutral_value(self):
        """
        Every vector space have to have a neutral value with respect to sum
        """

        assert self.v + self.neutral_value == self.v

    def test_neutral_value(self):
        """
        In every vector space, for every vector have to be a unique inverted vector
        and their sum have to be the neutral value
        """

        assert self.v + -1 * self.v == self.neutral_value

    def test_distributary(self):
        """
        Every vector space have to be distributary with respect to multiply with a real value
        """

        assert 2 * (self.w + self.v) == 2 * self.w + 2 * self.v

    def test_associativity_by_real_value(self):
        """
        Every vector space have to be associativity with respect to multiply for real value
        """

        assert (2 * 3) * self.v == 2 * (3 * self.v)

    def test_neutral_value_in_multiply_for_real_values(self):
        assert self.v * 1 == self.v


class DotProductTest(TestCase):
    """
    This class is a unit test about the properties of the dot product between vectors.
    """

    def setUp(self):
        self.v = Vector([randint(1, 100) for _ in range(10)])
        self.w = Vector([randint(1, 100) for _ in range(10)])
        self.z = Vector([randint(1, 100) for _ in range(10)])

    def test_commutativity(self):

        assert self.w ^ self.v == self.v ^ self.w

    def test_associativity_by_real_value(self):

        assert (2 * self.w) ^ self.v == 2 * (self.w ^ self.v)

    def test_distributary_by_vectorial_sum(self):

        assert (self.w + self.v) ^ self.z == self.w ^ self.z + self.v ^ self.z


class NormTest(TestCase):
    """
    This class is a unit test about the properties of norms.

    """

    def setUp(self):
        self.v = Vector([randint(1, 100) for _ in range(10)])
        self.w = Vector([randint(1, 100) for _ in range(10)])
        self.z = Vector([randint(1, 100) for _ in range(10)])

        self.neutral_value = Vector.neutral_value(10)

    def op(self, v):
        """Each norm should override this function and call is function"""

    def test_positive(self):
        n = self.op(self.v)
        assert n >= 0
        assert not n == 0 or n == self.neutral_value

    def test_homogeneity(self):
        assert self.op(2 * self.v) == 2 * self.op(self.v)

    def test_triangular_unequally(self):
        assert self.op(self.v + self.w) <= self.op(self.v) + self.op(self.w)


class EuclideanTest(NormTest):
    def op(self, v):
        return v.euclidean_norm()
