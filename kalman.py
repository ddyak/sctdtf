#! /usr/bin/env python3

import numpy as np

def predicted_resud_uncert(Vk, Hk, Cl):
    """ Uncertainty of the predicted residual
        $V_k + H_k C_{k-1} H_k^T$ """
    return Vk + np.einsum('kij, kil, klm -> ljm', Hk, Cl, Hk)

def gain(Cl, Hk, Vk):
    """ Gain matrix
        $C_{k-1} H^T_k (R_k^{k-1})^{-1}$ """
    Rlk = predicted_resud_uncert(Vk, Hk, Cl)
    return np.einsum('kij, kli, klm -> ljm', Cl, Hk, np.linalg.inv(Rlk))

def covariance_full():
    pass

def covariance_short():
    pass

def covariance_normal():
    pass