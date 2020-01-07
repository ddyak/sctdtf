""" Helper functions for matrix multiplication """

import numpy as np

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
    return np.einsum('kij, kli, klm -> kjm', A, B, C)

def mtxabat(A, B):
    """ ABA^T utility function """
    if VERB:
        print('mtxabat {} vs {}'.format(A.shape, B.shape))
    return np.einsum('kij, kjl, kml -> kim', A, B, A)
