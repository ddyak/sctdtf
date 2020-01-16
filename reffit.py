#! /usr/bin/env python3

""" Complete chi2 fit for reference """

import numpy as np
from scipy.linalg import block_diag

from event_generator import UNIT, MASS_DICT, p3top4, mass_sq, mass, make_hist
import eintools as et

def gmass(p4pip, p4pim):
    """ Mass constraint """
    return MASS_DICT['K0_S']**2 - mass_sq(p4pip + p4pim)

def gmomentum(p4pip, p4pim, p4k):
    """ Momentum conservation constraint """
    return p4k - p4pip - p4pim

def chi2(p3pip0, p3pim0, p4pip, p4pim, covInv):
    """ Calculates chi2 """
    return et.chi2_item(p4pip[:, 1:] - p3pip0, covInv) +\
           et.chi2_item(p4pim[:, 1:] - p3pim0, covInv)

def gradient(p3pip0, p3pim0, p4pip, p4pim, p4k, covInv, lam):
    """ 15D Gradient """
    lamm, lame, lamp =\
        lam[:, 0].reshape(-1, 1), lam[:, 1].reshape(-1, 1), lam[:, 2:]
    (epip, p3pip), (epim, p3pim), (ek, p3k) =\
        [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4pip, p4pim, p4k]]

    ddp1 = np.dot(p3pip - p3pip0, covInv) - lame*p3pip/epip - lamp
    ddp2 = np.dot(p3pim - p3pim0, covInv) - lame*p3pim/epim - lamp
    ddek = -2*lamm*ek  + lame
    ddpk = +2*lamm*p3k + lamp
    dlamm = gmass(p4pip, p4pim)
    dlamep = gmomentum(p4pip, p4pim, p4k)

    return 2*np.column_stack([ddp1, ddp2, ddek, ddpk, dlamm, dlamep])


# TODO calculate gradient coefficients

def cascade_gradient(p3_ks_pip0, p3_ks_pim0, p4_ks_pip, p4_ks_pim, p4k, lamk,
                    p3_phi_pip0, p3_phi_pim0, p4_phi_pip, p4_phi_pim, p4phi, lamphi, 
                    p4d, lamd, covInv):
    """ 39D Gradient """
    lamm_k, lame_k, lamp_k =\
        lamk[:, 0].reshape(-1, 1), lamk[:, 1].reshape(-1, 1), lamk[:, 2:]
    (epip, p3_ks_pip), (epim, p3_ks_pim), (ek, p3k) =\
        [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4_ks_pip, p4_ks_pim, p4k]]

    lamm_phi, lame_phi, lamp_phi =\
        lamphi[:,0].reshape(-1, 1), lamphi[:, 1].reshape(-1, 1), lamphi[:, 2:]
    # (epip, p3_ks_pip), (epim, p3_ks_pim), (ek, p3k) =\
        # [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4_ks_pip, p4_ks_pim, p4k]]

    lamm_d, lame_d, lamp_d =\
        lamd[:,0].reshape(-1, 1), lamd[:, 1].reshape(-1, 1), lamd[:, 2:]
    # (epip, p3_ks_pip), (epim, p3_ks_pim), (ek, p3k) =\
        # [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4_ks_pip, p4_ks_pim, p4k]]


    dd_ks_p1 = np.dot(p3_ks_pip - p3_ks_pip0, covInv) - lame_k*p3_ks_pip/epip - lamp_k
    dd_ks_p2 = np.dot(p3_ks_pim - p3_ks_pim0, covInv) - lame_k*p3_ks_pim/epim - lamp_k
    ddek = -2*lamm_k*ek  + lame_k - lame_d
    ddpk = +2*lamm_k*p3k + lamp_k - lamp_d
    dlamm = gmass(p4_ks_pip, p4_ks_pim)
    dlamep = gmomentum(p4_ks_pip, p4_ks_pim, p4k)
#####################################################################
    lamm, lame, lamp =\
        lam[:, 0].reshape(-1, 1), lam[:, 1].reshape(-1, 1), lam[:, 2:]
    (epip, p3pip), (epim, p3pim), (ek, p3k) =\
        [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4pip, p4pim, p4k]]

    ddp1 = np.dot(p3pip - p3pip0, covInv) - lame*p3pip/epip - lamp
    ddp2 = np.dot(p3pim - p3pim0, covInv) - lame*p3pim/epim - lamp
    ddek = -2*lamm*ek  + lame
    ddpk = +2*lamm*p3k + lamp
    dlamm = gmass(p4pip, p4pim)
    dlamep = gmomentum(p4pip, p4pim, p4k)
####################################################################
    lamm, lame, lamp =\
        lam[:, 0].reshape(-1, 1), lam[:, 1].reshape(-1, 1), lam[:, 2:]
    (epip, p3pip), (epim, p3pim), (ek, p3k) =\
        [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4pip, p4pim, p4k]]

    ddp1 = np.dot(p3pip - p3pip0, covInv) - lame*p3pip/epip - lamp
    ddp2 = np.dot(p3pim - p3pim0, covInv) - lame*p3pim/epim - lamp
    ddek = -2*lamm*ek  + lame
    ddpk = +2*lamm*p3k + lamp
    dlamm = gmass(p4pip, p4pim)
    dlamep = gmomentum(p4pip, p4pim, p4k)


    return 2*np.column_stack([ddp1, ddp2, ddek, ddpk, dlamm, dlamep])

def hessian(p4pip, p4pim, p4k, covInv, lam):
    """ 15x15 Hessian """
    N = p4pip.shape[0]
    lamm, lame = lam[:, 0].reshape(-1, 1), lam[:, 1].reshape(-1, 1)
    (epip, p3pip), (epim, p3pim), (ek, p3k) =\
        [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4pip, p4pim, p4k]]

    p3overEpip = p3pip / epip
    p3overEpim = p3pim / epim

    dphess = lambda e, pOvE : np.einsum('ki, ij -> kij', lame / e, np.ones((3,3))) *\
        (np.einsum('ki, kj -> kij', pOvE, pOvE) - np.eye(3))

    hess = np.zeros((N, 15, 15))
    hess[:, 0:3,  0:3] = covInv + dphess(epip, p3overEpip)
    hess[:, 3:6,  3:6] = covInv + dphess(epim, p3overEpim)
    hess[:,6:10, 6:10] = 2*np.einsum('ki, ij -> kij', lamm, np.diag([-1, 1, 1 ,1]))
    hess[:,  10,    6] = hess[:,   6,   10] = -2*ek.ravel()  # dlamm dek
    hess[:,  10, 7:10] = hess[:,7:10,   10] =  2*p3k         # dlamm dpk
    hess[:,  11,  0:3] = hess[:, 0:3,   11] = -p3pip / epip  # dp1 dlam_eps
    hess[:,  11,  3:6] = hess[:, 3:6,   11] = -p3pim / epim  # dp2 dlam_eps
    hess[:, 11:, 6:10] = hess[:,6:10,  11:] =  np.eye(4)     # dek dlam_eps
    hess[:, 12:,  0:3] = hess[:, 0:3,  12:] = -np.eye(3)     # dlam_p dp1
    hess[:, 12:,  3:6] = hess[:, 3:6,  12:] = -np.eye(3)     # dlam_p dp2
    return 2*hess

def cascade_hessian(p4_k_pip, p4_k_pim, p4k, lam_k, p4_phi_pip, p4_phi_pim, p4phi, lam_phi, p4d, lam_d, covInv):
    """ 39x39 Hessian """
    N = p4_k_pip.shape[0]
    
    lamm_k, lame_k = lam_k[:, 0].reshape(-1, 1), lam_k[:, 1].reshape(-1, 1)
    (e_k_pip, p3_k_pip), (e_k_pim, p3_k_pim), (ek, p3k) =\
        [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4_k_pip, p4_k_pim, p4k]]

    p3overE_k_pip = p3_k_pip / e_k_pip
    p3overE_k_pim = p3_k_pim / e_k_pim

    dphess = lambda e, pOvE : np.einsum('ki, ij -> kij', lame_k / e, np.ones((3,3))) *\
        (np.einsum('ki, kj -> kij', pOvE, pOvE) - np.eye(3))

    hess = np.zeros((N, 39, 39))
    hess[:,   0:3,  0:3] = covInv + dphess(e_k_pip, p3overE_k_pip)
    hess[:,   3:6,  3:6] = covInv + dphess(e_k_pim, p3overE_k_pim)
    hess[:,  6:10, 6:10] = 2*np.einsum('ki, ij -> kij', lamm_k, np.diag([-1, 1, 1 ,1]))
    hess[:,    10,    6] = hess[:,   6,   10] = -2*ek.ravel()  # dlamm dek
    hess[:,    10, 7:10] = hess[:,7:10,   10] =  2*p3k         # dlamm dpk
    hess[:,    11,  0:3] = hess[:, 0:3,   11] = -p3_k_pip / e_k_pip  # dp1 dlam_eps
    hess[:,    11,  3:6] = hess[:, 3:6,   11] = -p3_k_pim / e_k_pim  # dp2 dlam_eps
    hess[:, 11:15, 6:10] = hess[:,6:10,  11:15] =  np.eye(4)     # dek dlam_eps
    hess[:, 12:15,  0:3] = hess[:, 0:3,  12:15] = -np.eye(3)     # dlam_p dp1
    hess[:, 12:15,  3:6] = hess[:, 3:6,  12:15] = -np.eye(3)     # dlam_p dp2

    lamm_phi, lame_phi = lam_phi[:, 0].reshape(-1, 1), lam_phi[:, 1].reshape(-1, 1)
    (e_phi_pip, p3_phi_pip), (e_phi_pim, p3_phi_pim), (ek, p3k) =\
        [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4_phi_pip, p4_phi_pim, p4phi]]

    p3overE_phi_pip = p3_phi_pip / e_phi_pip
    p3overE_phi_pim = p3_phi_pim / e_phi_pim

    dphess = lambda e, pOvE : np.einsum('ki, ij -> kij', lame_phi / e, np.ones((3,3))) *\
        (np.einsum('ki, kj -> kij', pOvE, pOvE) - np.eye(3))

    hess[:, 15:18,  15:18] = covInv + dphess(e_phi_pip, p3overE_phi_pip)
    hess[:, 18:21,  18:21] = covInv + dphess(e_phi_pim, p3overE_phi_pim)
    hess[:, 21:25, 21:25] = 2*np.einsum('ki, ij -> kij', lamm_phi, np.diag([-1, 1, 1 ,1]))
    hess[:,  25,    21] = hess[:,   21,   25] = -2*ek.ravel()  # dlamm dek
    hess[:,  25, 22:25] = hess[:,22:25,   25] =  2*p3k         # dlamm dpk
    hess[:,  26,  15:18] = hess[:, 15:18,   26] = -p3_phi_pip / e_phi_pip  # dp1 dlam_eps
    hess[:,  26,  18:21] = hess[:, 18:21,   26] = -p3_phi_pim / e_phi_pim  # dp2 dlam_eps
    hess[:, 26:30, 21:25] = hess[:,21:25,  26:] =  np.eye(4)     # dek dlam_eps
    hess[:, 27:30,  15:18] = hess[:, 15:18,  27:30] = -np.eye(3)     # dlam_p dp1
    hess[:, 27:30,  18:21] = hess[:, 18:21,  27:30] = -np.eye(3)     # dlam_p dp2

    lamm_d, lame_d = lam_d[:, 0].reshape(-1, 1), lam_d[:, 1].reshape(-1, 1)
    (ek, p3k) = [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4d]]

    dphess = lambda e, pOvE : np.einsum('ki, ij -> kij', lame_d / e, np.ones((3,3))) *\
        (np.einsum('ki, kj -> kij', pOvE, pOvE) - np.eye(3))

    hess[:, 30:34, 30:34] = 2*np.einsum('ki, ij -> kij', lamm_d, np.diag([-1, 1, 1 ,1]))
    hess[:,   34,     30] = hess[:,     30,   34] = -2*ek.ravel()
    hess[:,   34,  31:34] = hess[:,  31:34,   34] =  2*p3k         # dlamm dpk

    hess[:, 35:39,  30:34] = hess[:, 30:34,  35:39] =  np.eye(4)     # dek dlam_eps
    hess[:, 35:39,  6:10] = hess[:,   6:10,  35:39] = -np.eye(4)     # dlam_p dp1
    hess[:, 27:30,  21:25] = hess[:, 21:25,  35:39] = -np.eye(4)     # dlam_p dp2
 
    return 2*hess


def fit_to_ks(p3pip, p3pim, cov, nit=5):
    """ cov [3 x 3] """
    N = p3pip.shape[0]
    mpi = MASS_DICT['pi+']
    p3pip0, p3pim0 = p3pip.copy(), p3pim.copy()
    p4pip, p4pim = [p3top4(p, mpi) for p in [p3pip, p3pim]]
    p4ks = p4pip + p4pim
    covInv = np.linalg.inv(cov)

    print('Inverse covariance\n{}'.format(covInv))
    print('N = {}'.format(N))

    lam = 1*np.ones((N, 5))

    logs = {key : [] for key in ['chi2', 'xi', 'grad', 'hess', 'det', 'cov']}
    def save_log(xi, p4pip, p4pim, grad, hess):
        logs['chi2'].append(chi2(p3pip0, p3pim0, p4pip, p4pim, covInv))
        logs['xi'].append(xi[:, :10].copy())
        logs['grad'].append(grad.copy())
        logs['hess'].append(hess.copy())
        logs['det'].append(np.linalg.det(hess))
        logs['cov'].append(2 * np.linalg.inv(hess))

    def calc(xi):
        p3pip, p3pim, p4ks, lam = xi[:, :3], xi[:, 3:6], xi[:, 6:10], xi[:, 10:]
        p4pip, p4pim = [p3top4(p, mpi) for p in [p3pip, p3pim]]
        grad = gradient(p3pip0, p3pim0, p4pip, p4pim, p4ks, covInv, lam)
        hess = hessian(p4pip, p4pim, p4ks, covInv, lam)
        return (p4pip, p4pim, grad, hess)

    xi = np.column_stack([p3pip, p3pim, p4ks, lam])
    for iter in range(nit):
        print('Iteration {}'.format(iter))
        p4pip, p4pim, grad, hess = calc(xi)
        save_log(xi, p4pip, p4pim, grad, hess)
        xi -= np.einsum('kij, ki -> kj', np.linalg.inv(hess), grad)

    p4pip, p4pim, grad, hess = calc(xi)
    save_log(xi, p4pip, p4pim, grad, hess)

    return logs


def fit_to_d0(p3_ks_pip, p3_ks_pim, p3_phi_pip, p3_phi_pim, cov, nit=5):
    """ cov [3 x 3] """
    N = p3_ks_pip.shape[0]
    mks, mphi, mpi = MASS_DICT['K0_S'], MASS_DICT['phi'], MASS_DICT['pi+']
    
    p3_ks_pip0, p3_ks_pim0 = p3_ks_pip.copy(), p3_ks_pim.copy()
    p4_ks_pip, p4_ks_pim = [p3top4(p, mpi) for p in [p3_ks_pip, p3_ks_pim]]
    p4ks = p4_ks_pip + p4_ks_pim
    covInv = np.linalg.inv(cov)

    print('Inverse covariance\n{}'.format(covInv))
    print('N = {}'.format(N))

    lam = 1*np.ones((N, 5))

    logs = {key : [] for key in ['chi2', 'xi', 'grad', 'hess', 'det', 'cov']}
    def save_log(xi, p4_ks_pip, p4_ks_pim, p4_ks_pip, p4_ks_pim, grad, hess):
        logs['chi2'].append(chi2(p3pip0, p3pim0, p4pip, p4pim, covInv))
        logs['xi'].append(xi[:, :10].copy())
        logs['grad'].append(grad.copy())
        logs['hess'].append(hess.copy())
        logs['det'].append(np.linalg.det(hess))
        logs['cov'].append(2 * np.linalg.inv(hess))

    def calc(xi):
        p3pip, p3pim, p4ks, lam = xi[:, :3], xi[:, 3:6], xi[:, 6:10], xi[:, 10:]
        p4pip, p4pim = [p3top4(p, mpi) for p in [p3pip, p3pim]]
        grad = gradient(p3pip0, p3pim0, p4pip, p4pim, p4ks, covInv, lam)
        hess = hessian(p4pip, p4pim, p4ks, covInv, lam)
        return (p4pip, p4pim, grad, hess)

    xi = np.column_stack([p3pip, p3pim, p4ks, lam])
    for iter in range(nit):
        print('Iteration {}'.format(iter))
        p4pip, p4pim, grad, hess = calc(xi)
        save_log(xi, p4pip, p4pim, grad, hess)
        xi -= np.einsum('kij, ki -> kj', np.linalg.inv(hess), grad)

    p4pip, p4pim, grad, hess = calc(xi)
    save_log(xi, p4pip, p4pim, grad, hess)

    return logs



def main():
    from event_generator import generate, generate_cascade
    cov = np.diag([3,3,5])**2 * UNIT**2
    N = 10**4

    ptot = np.array([1000, 0, 0])

    (p3_ks_pip, p3_ks_pim, p3_phi_pip, p3_phi_pim), \
    p3_ks_pip_gen, p3_ks_pim_gen, p3_phi_pip_gen, p3_phi_pim_gen \
        = generate_cascade(N, cov, ptot=ptot)

    logs = pfit_to_d0(p3_ks_pip, p3_ks_pim, p3_phi_pip, p3_phi_pim, cov, nit=5)


    # for energy in np.logspace(0, 3, 5):
    #     ptot = np.array([energy, 0, 0])
    #     (p3pip, p3pim), p3pipGen, p3pimGen = generate(N, cov, ptot=ptot)

    #     logs = fit_to_ks(p3pip, p3pim, cov, nit=10)
    #     np.savez('logs/fitres_{:.3f}_MeV'.format(energy),
    #         chi2=logs['chi2'],
    #         xi=logs['xi'],
    #         Ck=logs['cov'],
    #         grad=logs['grad'],
    #         hess=logs['hess'],
    #         det=logs['det'],
    #     pipgen=p3pipGen,
    #     pimgen=p3pimGen,
    #         cov=cov
    #     )


if __name__ == '__main__':
    main()
