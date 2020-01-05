#! /usr/bin/env python3

""" Complete chi2 fit for reference """

import numpy as np
from scipy.linalg import block_diag

from event_generator import MASS_DICT, p3top4, mass_sq, mass, make_hist

mksSq = MASS_DICT['K0_S']**2

def gmass(p4pip, p4pim):
    """ Mass constraint """
    return mksSq - mass_sq(p4pip + p4pim)

def gmomentum(p4pip, p4pim, p4k):
    """ Momentum conservation constraint """
    return p4k - p4pip - p4pim

def chi2(p3pip0, p3pim0, p4pip, p4pim, p4k, covInv, lam):
    """ Calculates chi2 with mass hypothesys and momentum conservation """
    dp3pip = p4pip[:, 1:] - p3pip0
    dp3pim = p4pim[:, 1:] - p3pim0
    # return np.dot(np.dot(dp3pip, covInv), dp3pip.T) +\
    #        np.dot(np.dot(dp3pim, covInv), dp3pim.T) +\
    return np.einsum('...i, ij, ...j -> ...', dp3pip, covInv, dp3pip) +\
           np.einsum('...i, ij, ...j -> ...', dp3pim, covInv, dp3pim) +\
           2.*lam[:, 0]  * gmass(p4pip, p4pim) +\
           2.*np.sum(lam[:, 1:] * gmomentum(p4pip, p4pim, p4k), axis=-1)

def gradient(p4pip, p4pim, p4k, covInv, lam):
    """ 15D Gradient """
    lamm, lame, lamp =\
        lam[:, 0].reshape(-1, 1), lam[:, 1].reshape(-1, 1), lam[:, 2:]
    (epip, p3pip), (epim, p3pim), (ek, p3k) =\
        [(x[:, 0].reshape(-1, 1), x[:, 1:]) for x in [p4pip, p4pim, p4k]]

    ddp1 = np.dot(p3pip, covInv) - lame*p3pip/epip - lamp
    ddp2 = np.dot(p3pim, covInv) - lame*p3pim/epim - lamp
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
    hess[:, 10,     7] = hess[:,    7,  10] = -2*ek.ravel()  # dlamm dek
    hess[:, 10,  8:11] = hess[:, 8:11,  10] =  2*p3k         # dlamm dpk
    hess[:, 11,   0:3] = hess[:, 0:3,   11] = -p3pip / epip  # dp1 dlam_eps
    hess[:, 11,   3:6] = hess[:, 3:6,   11] = -p3pim / epim  # dp2 dlam_eps
    hess[:, 11:, 7:11] = hess[:, 7:11, 11:] =  np.eye(4)     # dek dlam_eps
    hess[:, 12:,  0:3] = hess[:, 0:3,  12:] = -np.eye(3)     # dlam_p dp1
    hess[:, 12:,  3:6] = hess[:, 3:6,  12:] = -np.eye(3)     # dlam_p dp2

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

    logs = {key : [] for key in ['chi2', 'xi', 'grad', 'hess', 'det']}
    def save_log(xi, p4pip, p4pim, p4ks, lam, grad, hess):
        logs['chi2'].append(chi2(p3pip0, p3pim0, p4pip, p4pim, p4ks, covInv, lam))
        logs['xi'].append(xi.copy())
        logs['grad'].append(grad.copy())
        logs['hess'].append(hess.copy())
        logs['det'].append(np.linalg.det(hess))

    def calc(xi):
        p3pip, p3pim, p4ks, lam = xi[:, :3], xi[:, 3:6], xi[:, 6:10], xi[:, 10:]
        p4pip, p4pim = [p3top4(p, mpi) for p in [p3pip, p3pim]]
        grad = gradient(p4pip, p4pim, p4ks, covInv, lam)
        hess = hessian(p4pip, p4pim, p4ks, covInv, lam)
        return (p4pip, p4pim, grad, hess)

    def print_summary(p4pip, p4pim, grad, hess):
        print('p4pip\n{}'.format(p4pip[0]))
        print('p4pim\n{}'.format(p4pim[0]))
        print('grad\n{}'.format(grad[0]))
        print('hess\n{}'.format(hess[0].diagonal()))
        print('invHess\n{}'.format(np.linalg.inv(hess[0]).diagonal()))

    xi = np.column_stack([p3pip, p3pim, p4ks, lam])
    for iter in range(nit):
        print('Iteration {}'.format(iter))
        p4pip, p4pim, grad, hess = calc(xi)
        save_log(xi, p4pip, p4pim, p4ks, lam, grad, hess)
        print_summary(p4pip, p4pim, grad, hess)
        xi -= np.einsum('kij, ki -> kj', np.linalg.inv(hess), grad)

    p4pip, p4pim, grad, hess = calc(xi)
    save_log(xi, p4pip, p4pim, p4ks, lam, grad, hess)
    print('Final')
    print_summary(p4pip, p4pim, grad, hess)

    np.savez('logs/fitres',
        chi2=logs['chi2'],
          xi=logs['xi'],
        grad=logs['grad'],
        hess=logs['hess'],
         det=logs['det'],
    )

    return logs

def generated_mass(p3pip, p3pim):
    import matplotlib.pyplot as plt
    p4pip, p4pim = [p3top4(p, MASS_DICT['pi+']) for p in[p3pip, p3pim]]
    m = mass(p4pip + p4pim)
    print(np.mean(m), np.std(m))
    x, bins, e = make_hist(m)

    plt.figure(figsize=(6,5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    plt.grid()
    plt.tight_layout()
    plt.xlabel(r'$m(\pi^+\pi^-)$ (MeV)', fontsize=16)
    plt.show()

def main():
    from event_generator import generate
    cov = np.diag([3,3,5])
    N = 1
    p3pip, p3pim = generate(N, cov)

    generated_mass(p3pip, p3pim)
    logs = fit_to_ks(p3pip, p3pim, cov, nit=5)
    print(logs['chi2'])

if __name__ == '__main__':
    main()
