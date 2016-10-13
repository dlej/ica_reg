import numpy as np
import unittest

def gen_orthonormal(m, n=None):
    if n is None:
        n = m
    if n > m:
        m, n = n, m
        transpose = True
    else:
        transpose = False

    Q = np.zeros((m, n))
    for j in range(n):
        q = np.random.randn(m)
        q -= Q.dot(Q.T.dot(q))
        q /= np.linalg.norm(q)
        Q[:, j] = q

    return Q.T if transpose else Q



class TestOrthonormalGen(unittest.TestCase):

    def test_square(self):
        Q = gen_orthonormal(3)
        self.assertTrue(np.allclose(Q.dot(Q.T), np.eye(3)))
        self.assertTrue(np.allclose(Q.T.dot(Q), np.eye(3)))

    def test_tall(self):
        Q = gen_orthonormal(5, 3)
        self.assertTrue(np.allclose(Q.T.dot(Q), np.eye(3)))
        self.assertFalse(np.allclose(Q.dot(Q.T), np.eye(5)))

    def test_wide(self):
        Q = gen_orthonormal(3, 5)
        self.assertFalse(np.allclose(Q.T.dot(Q), np.eye(5)))
        self.assertTrue(np.allclose(Q.dot(Q.T), np.eye(3)))


if __name__ == '__main__':
    unittest.main()
