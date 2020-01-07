#! /usr/bin/env python3

import numpy as np

from event_generator import UNIT, MASS_DICT, p3top4, mass
from reffit import gmass, gmomentum, chi2
import eintools as et

def predicted_resid_uncert(Vk, Hk, Cl):
    """ Uncertainty of the predicted residual
        $V_k + H_k C_{k-1} H_k^T$ """
    return Vk + et.mtxabat(Hk, Cl)

def gain(Cl, Hk, Rlk):
    """ Gain matrix
        $C_{k-1} H^T_k (R_k^{k-1})^{-1}$ """
    return et.mtxabtc(Cl, Hk, np.linalg.inv(Rlk))

#TODO: covariance_full is probably wrong
def covariance_full(Cl, Kk, Hk, Vk):
    """ Full expression for covariance update """
    return et.mtxabat(1 - et.mtxab(Kk, Hk), Cl) + et.mtxabat(Kk, Vk)

def covariance_fast(Cl, Kk, Hk):
    """ Simple but numerically unstable covariance update """
    return et.mtxab(1 - et.mtxab(Kk, Hk), Cl)

def covariance_normal(Cl, Kk, Hk, Rlk):
    """ Simplified, but numerically stable covariance update """
    return Cl - et.mtxab(Kk, 2*et.mtxab(Hk, Cl) - et.mtxabt(Rlk, Kk))

def covariance_exact(Cl, Kk, Gk):
    """ Covariance update for exact constraints """
    return et.mtxabat(1 - et.mtxab(Kk, Gk), Cl)

def gain_exact(Cl, Gk):
    """ Gain matrix for exact constraints """
    cInv = np.linalg.inv(et.mtxabat(Gk, Cl))
    return (et.mtxabtc(Cl, Gk, cInv), cInv)

def xi_upd(xi, Kk, rk):
    """ Update state vector """
    # print('xi_upd {} {} {}'.format(xi.shape, Kk.shape, rk.shape))
    return xi - np.einsum('kij, kj -> ki', Kk, rk)

#TODO: refactor pfit_to_ks
def pfit_to_ks(p3pip, p3pim, cov, nit=5, gpit=3, gmit=3):
    """ Progressive mass-constrained fit for Ks0 -> pi+ pi- """
    mpi = MASS_DICT['pi+']
    N = p3pip.shape[0]
    covInv = np.linalg.inv(cov)
    p3pip0, p3pim0 = p3pip.copy(), p3pim.copy()
    p4pip, p4pim = [p3top4(p, mpi) for p in [p3pip, p3pim]]
    p4ks = p4pip + p4pim

    ndim = 10
    Ck = np.zeros((N, ndim, ndim))
    Ck[:] = 10**3*np.eye(ndim)
    print('Init Ck\n{}'.format(Ck[0]))
    print('Ck shape {}'.format(Ck.shape))
    xi = np.column_stack([p3pip, p3pim, p4ks])

    def unpack(xi):
        p3pip, p3pim, p4ks = xi[:, :3], xi[:, 3:6], xi[:, 6:10]
        p4pip, p4pim = [p3top4(p, mpi) for p in [p3pip, p3pim]]
        epip = p4pip[:, 0].reshape(-1, 1)
        epim = p4pim[:, 0].reshape(-1, 1)
        return (p3pip, p3pim, p4ks, p4pip, p4pim, epip, epim)

    ncov = np.zeros((N, *cov.shape))
    ncov[:] = cov
    logs = {key: [] for key in ['xi', 'cov', 'chi2', 'mk']}
    xi0 = xi.copy()
    for idx in range(nit):
        print('Iteration {}'.format(idx+1))

        # Apply pi+ momentum measurement constraint #
        Hk = np.zeros((N, 3, ndim))
        Hk[:,:, :3] = np.eye(3)
        Rlk = predicted_resid_uncert(cov, Hk, Ck)
        Kk = gain(Ck, Hk, Rlk)
        xi = xi_upd(xi, Kk, xi[:,:3] - xi0[:,:3])  # trivial at 1st iteration
        # Ck = covariance_full(Ck, Kk, Hk, ncov)
        Ck = covariance_normal(Ck, Kk, Hk, Rlk)

        # Apply pi- momentum measurement constraint #
        Hk = np.zeros((N, 3, ndim))
        Hk[:,:,3:6] = np.eye(3)
        Rlk = predicted_resid_uncert(cov, Hk, Ck)
        Kk = gain(Ck, Hk, Rlk)
        xi = xi_upd(xi, Kk, xi[:,3:6] - xi0[:,3:6])  # trivial at 1st iteration
        # Ck = covariance_full(Ck, Kk, Hk, ncov)
        Ck = covariance_normal(Ck, Kk, Hk, Rlk)

        # momentum conservation constraint #
        for _ in range(gpit):
            p3pip, p3pim, p4ks, p4pip, p4pim, epip, epim = unpack(xi)
            Gk = np.zeros((N, 4, ndim))
            Gk[:, 0, :3] = -p3pip/epip
            Gk[:, 0,3:6] = -p3pim/epim
            Gk[:, 1:, :3] = Gk[:, 1:, 3:6] = -np.eye(3)
            Gk[:, :, 6:] = np.eye(4)
            Kk, _ = gain_exact(Ck, Gk)
            gp = gmomentum(p4pip, p4pim, p4ks)
            xi = xi_upd(xi, Kk, gp)
        Ck = covariance_exact(Ck, Kk, Gk)

        # mass constraint #
        for _ in range(gmit):
            _, _, p4ks, p4pip, p4pim, _, _ = unpack(xi)
            Gk = np.zeros((N, 1, ndim))
            Gk[:, 0, 6:] = 2*np.einsum('ki, i -> ki', p4ks, np.array([-1, 1, 1, 1]))
            Kk, _ = gain_exact(Ck, Gk)
            gm = gmass(p4pip, p4pim).reshape(-1, 1)
            xi = xi_upd(xi, Kk, gm)
        Ck = covariance_exact(Ck, Kk, Gk)

        # write log #
        _, _, p4ks, p4pip, p4pim, _, _ = unpack(xi)
        logs['xi'].append(xi.copy())
        logs['cov'].append(Ck.copy())
        logs['chi2'].append(chi2(p3pip0, p3pim0, p4pip, p4pim, covInv))
        logs['mk'].append(mass(p4ks))
    print('Final Ck\n{}'.format(Ck[0]))
    return logs

def main():
    from event_generator import generate
    # et.VERB = True
    cov = np.diag([3,3,5])**2 * UNIT**2
    N = 1
    (p3pip, p3pim), p3pipGen, p3pimGen = generate(N, cov)

    logs = pfit_to_ks(p3pip, p3pim, cov, nit=3, gpit=3, gmit=3)
    np.savez('logs/pfitres',
        chi2=logs['chi2'],
          Ck=logs['cov'],
          mk=logs['mk'],
          xi=logs['xi'],
      pipgen=p3pipGen,
      pimgen=p3pimGen,
         cov=cov
    )

if __name__ == '__main__':
    main()
