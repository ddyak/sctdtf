""" Helper functions for matrix multiplication """

import numpy as np
import unittest

VERB = False

def mtxab(A, B):
    """ AB utility function """
    if VERB:
        print('mtxab {} {}'.format(A.shape, B.shape))
    return np.einsum('kij, kjl -> kil', A, B)

def mtxabt(A, B):
    """ AB^T utility function """
    if VERB:
        print('mtxabt {} {}'.format(A.shape, B.shape))
    return np.einsum('kij, klj -> kil', A, B)

def mtxabc(A, B, C):
    """ ABC utility function """
    if VERB:
        print('mtxabc {} {} {}'.format(A.shape, B.shape, C.shape))
    return np.einsum('kij, kjl, klm -> kim', A, B, C)

def mtxabtc(A, B, C):
    """ AB^TC utility function """
    if VERB:
        print('mtxabtc {} {} {}'.format(A.shape, B.shape, C.shape))
    return np.einsum('kij, klj, klm -> kim', A, B, C)

def mtxabat(A, B):
    """ ABA^T utility function """
    if VERB:
        print('mtxabat {} vs {}'.format(A.shape, B.shape))
    return np.einsum('kij, kjl, kml -> kim', A, B, A)

def chi2_item(r, cInv):
    """ rT cInv r """
    if VERB:
        print('chi2_item {} {}'.format(r.shape, cInv.shape))
    if len(cInv.shape) == 3:
        return np.einsum('ki, kij, kj -> k', r, cInv, r)
    return np.einsum('ki, ij, kj -> k', r, cInv, r)

class TestMtx(unittest.TestCase):
    def test_mtxab(self):
        N, dim = 10, 6
        A = np.random.rand(N, dim, dim)
        B = np.array([np.linalg.inv(it) for it in A])
        expected = np.array([np.eye(dim) for _ in range(N)])
        self.assertTrue(np.allclose(expected, mtxab(A, B)))

        dim1, dim2, dim3 = 4, 5, 6
        A = np.random.rand(N, dim1, dim2)
        B = np.random.rand(N, dim2, dim3)
        self.assertTrue((N, dim1, dim3) == mtxab(A, B).shape)

    def test_mtxabt(self):
        N, dim = 10, 6
        A = np.random.rand(N, dim, dim)
        B = np.array([np.linalg.inv(it.T) for it in A])
        expected = np.array([np.eye(dim) for _ in range(N)])
        self.assertTrue(np.allclose(expected, mtxabt(A, B)))

        dim1, dim2, dim3 = 4, 5, 6
        A = np.random.rand(N, dim1, dim2)
        B = np.random.rand(N, dim3, dim2)
        self.assertTrue((N, dim1, dim3), mtxabt(A, B).shape)

    def test_mtxabc(self):
        N, dim1, dim2, dim3, dim4 = 10, 4, 5, 6, 7
        A = np.random.rand(N, dim1, dim2)
        B = np.random.rand(N, dim2, dim3)
        C = np.random.rand(N, dim3, dim4)
        self.assertTrue(np.allclose(mtxab(mtxab(A,B), C), mtxabc(A, B, C)))
        self.assertTrue(np.allclose(mtxab(A, mtxab(B,C)), mtxabc(A, B, C)))

    def test_mtxabtc(self):
        N, dim1, dim2, dim3, dim4 = 10, 4, 5, 6, 7
        A = np.random.rand(N, dim1, dim2)
        B = np.random.rand(N, dim3, dim2)
        C = np.random.rand(N, dim3, dim4)
        self.assertTrue(np.allclose(mtxab(mtxabt(A,B), C), mtxabtc(A, B, C)))

    def test_mtxabat(self):
        N, dim1, dim2 = 10, 4, 5
        A = np.random.rand(N, dim1, dim2)
        B = np.random.rand(N, dim2, dim2)
        self.assertTrue(np.allclose(mtxabt(mtxab(A,B), A), mtxabat(A, B)))

if __name__ == '__main__':
    # VERB = True
    unittest.main()
