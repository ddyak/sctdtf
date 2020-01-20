#! /usr/bin/env python3

import numpy as np

import event_generator as eg
import reffit as rf
import eintools as et
import pathlib

def predicted_resid_uncert(Vk, Hk, Cl):
    """ Uncertainty of the predicted residual
        $V_k + H_k C_{k-1} H_k^T$ """
    return Vk + et.mtxabat(Hk, Cl)

def gain(Cl, Hk, RlkInv):
    """ Gain matrix
        $C_{k-1} H^T_k (R_k^{k-1})^{-1}$ """
    return et.mtxabtc(Cl, Hk, RlkInv)

def covariance_full(Cl, Kk, Hk, Vk):
    """ Full expression for covariance update """
    return et.mtxabat(np.eye(Cl.shape[-1]) - et.mtxab(Kk, Hk), Cl) + et.mtxabat(Kk, Vk)

def covariance_fast(Cl, Kk, Hk):
    """ Simple but numerically unstable covariance update """
    return et.mtxab(np.eye(Cl.shape[-1]) - et.mtxab(Kk, Hk), Cl)

def covariance_normal(Cl, Kk, Hk, Rlk):
    """ Simplified, but numerically stable covariance update """
    return Cl - et.mtxab(Kk, 2*et.mtxab(Hk, Cl) - et.mtxabt(Rlk, Kk))

def covariance_exact(Cl, Kk, Gk):
    """ Covariance update for exact constraints """
    # Rlk = et.mtxabat(Gk, Cl)
    # return Cl - et.mtxab(Kk, 2*et.mtxab(Gk, Cl) - et.mtxabt(Rlk, Kk))    
    return et.mtxabat(np.eye(Cl.shape[-1]) - et.mtxab(Kk, Gk), Cl)

def gain_exact(Cl, Gk, cInv):
    """ Gain matrix for exact constraints """
    return et.mtxabtc(Cl, Gk, cInv)

def gcgtinv(Gk, Cl):
    """ $(G_kC_{k-1}G_k^T)^{-1}$ """
    return np.linalg.inv(et.mtxabat(Gk, Cl))

def xi_upd(Kk, rk):
    """ Update state vector """
    return -np.einsum('kij, kj -> ki', Kk, rk)

def unpack(xi):
        p3pip, p3pim, p4ks = xi[:, :3], xi[:, 3:6], xi[:, 6:10]
        p4pip, p4pim = [eg.p3top4(p, eg.MASS_DICT['pi+']) for p in [p3pip, p3pim]]
        epip = p4pip[:, 0].reshape(-1, 1)
        epim = p4pim[:, 0].reshape(-1, 1)
        return (p3pip, p3pim, p4ks, p4pip, p4pim, epip, epim)

def cascade_unpack(xi):
        p3_ks_pip, p3_ks_pim, p4ks = xi[:, :3], xi[:, 3:6], xi[:, 6:10]
        p4_ks_pip, p4_ks_pim = [eg.p3top4(p, eg.MASS_DICT['pi+']) for p in [p3_ks_pip, p3_ks_pim]]
        e_ks_pip = p4_ks_pip[:, 0].reshape(-1, 1)
        e_ks_pim = p4_ks_pim[:, 0].reshape(-1, 1)
        
        p3_phi_pip, p3_phi_pim, p4phi = xi[:, 10:13], xi[:, 13:16], xi[:, 16:20]
        p4_phi_pip, p4_phi_pim = [eg.p3top4(p, eg.MASS_DICT['pi+']) for p in [p3_phi_pip, p3_phi_pim]]
        e_phi_pip = p4_phi_pip[:, 0].reshape(-1, 1)
        e_phi_pim = p4_phi_pim[:, 0].reshape(-1, 1)

        p4d0 = xi[:, 20:24]

        return p4_ks_pip, p4_ks_pim, p4ks, e_ks_pip, e_ks_pim, \
            p4_phi_pip, p4_phi_pim, p4phi, e_phi_pip, e_phi_pim, p4d0


def apply_meas(Hk, rk, cov, Ck, full=True):
    """ Apply measurement constraint """
    Rlk = predicted_resid_uncert(cov, Hk, Ck)
    RlkInv = np.linalg.inv(Rlk)
    Kk = gain(Ck, Hk, RlkInv)
    if full:
        return (xi_upd(Kk, rk), covariance_full(Ck, Kk, Hk, cov), et.chi2_item(rk, RlkInv))
    return (xi_upd(Kk, rk), covariance_normal(Ck, Kk, Hk, Rlk), et.chi2_item(rk, RlkInv))

def pfit_to_ks(p3pip, p3pim, cov, nit=5, gpit=3, gmit=3):
    """ Progressive mass-constrained fit for Ks0 -> pi+ pi- """
    mpi = eg.MASS_DICT['pi+']
    N = p3pip.shape[0]
    covInv = np.linalg.inv(cov)
    p3pip0, p3pim0 = p3pip.copy(), p3pim.copy()
    p4pip, p4pim = [eg.p3top4(p, mpi) for p in [p3pip, p3pim]]
    p4ks = p4pip + p4pim

    ndim = 10
    Ck = np.zeros((N, ndim, ndim))
    Ck[:] = 10**3*np.eye(ndim) * eg.UNIT**2
    xi = np.column_stack([p3pip, p3pim, p4ks])

    logs = {key: [] for key in ['xi', 'cov', 'chi2', 'mk', 'chi2v0']}
    xi0 = xi.copy()

    _, _, p4ks, p4pip, p4pim, _, _ = unpack(xi)
    logs['xi'].append(xi.copy())

    for idx in range(nit):
        print('Iteration {}'.format(idx+1))
        Ck = np.zeros((N, ndim, ndim))
        Ck[:] = 10**4*np.eye(ndim) * eg.UNIT**2
        chi2 = np.zeros(N)

        # Apply pi+ momentum measurement constraint #
        Hk = np.zeros((N, 3, ndim))
        Hk[:,:, :3] = np.eye(3)
        rk = xi[:,:3] - xi0[:,:3]
        dxi, Ck, chi2k = apply_meas(Hk, -rk, cov, Ck)
        xi += dxi
        chi2 += chi2k

        # Apply pi- momentum measurement constraint #
        Hk = np.zeros((N, 3, ndim))
        Hk[:,:,3:6] = np.eye(3)
        rk = xi[:,3:6] - xi0[:,3:6]
        dxi, Ck, chi2k = apply_meas(Hk, -rk, cov, Ck)
        xi += dxi
        chi2 += chi2k
      
       # momentum conservation constraint #
        gp_const = rf.gmomentum(p4pip, p4pim, p4ks)
        for _ in range(gpit):
            p3pip, p3pim, p4ks, p4pip, p4pim, epip, epim = unpack(xi)
            gp = rf.gmomentum(p4pip, p4pim, p4ks)
            Gk = np.zeros((N, 4, ndim))
            Gk[:, 0, :3] = -p3pip/epip
            Gk[:, 0,3:6] = -p3pim/epim
            Gk[:, 1:, :3] = Gk[:, 1:, 3:6] = -np.eye(3)
            Gk[:, :, 6:] = np.eye(4)
            GCGTInv = gcgtinv(Gk, Ck)
            Kk = gain_exact(Ck, Gk, GCGTInv)
            xi += xi_upd(Kk, gp)
        # _, _, p4ks, p4pip, p4pim, _, _ = unpack(xi)
        Ck = covariance_exact(Ck, Kk, Gk)
        chi2 += et.chi2_item(gp_const, GCGTInv)
        
        def gmass(p4ks):
            return rf.MASS_DICT['K0_S']**2 - rf.mass_sq(p4ks)

        # mass constraint #
        _, _, p4ks, p4pip, p4pim, _, _ = unpack(xi)
        gm_const = gmass(p4ks).reshape(-1, 1)
        for _ in range(gmit):
            print('Mass iteration {}'.format(_ + 1))
            _, _, p4ks, p4pip, p4pim, _, _ = unpack(xi)
            gm = gmass(p4ks).reshape(-1, 1)
            # gm = rf.gmass(p4pip, p4pim).reshape(-1, 1)
            Gk = np.zeros((N, 1, ndim))
            Gk[:,0,6:] = 2 * np.einsum('ki, i -> ki', p4ks, np.array([-1, 1, 1, 1]))
            GCGTInv = gcgtinv(Gk, Ck)
            Kk = gain_exact(Ck, Gk, GCGTInv)
            xi += xi_upd(Kk, gm)
        # _, _, _, p4pip, p4pim, _, _ = unpack(xi)
        Ck = covariance_exact(Ck, Kk, Gk)
        chi2 += et.chi2_item(gm_const, GCGTInv)

        # write log #
        _, _, p4ks, p4pip, p4pim, _, _ = unpack(xi)
        logs['xi'].append(xi.copy())
        logs['cov'].append(Ck.copy())
        # logs['chi2v0'].append(rf.chi2(p3pip0, p3pim0, p4pip, p4pim, covInv))
        logs['chi2'].append(chi2)
        logs['mk'].append(eg.mass(p4ks))
    # print('Final Ck\n{}'.format(Ck[0]))
    return logs


def pfit_to_d0(p3_ks_pip, p3_ks_pim, p3_phi_pip, p3_phi_pim, cov, nit=5, gpit=3, gmit=3):
    """ Progressive mass-constrained fit for D0 -> Ks Phi """
    md0, mphi, mks, mpi = [eg.MASS_DICT[key]
                           for key in ['D0', 'phi', 'K0_S', 'pi+']]
    N = p3_ks_pip.shape[0]

    covInv = np.linalg.inv(cov)

    p3_ks_pip0, p3_ks_pim0, p3_phi_pip0, p3_phi_pim0 = p3_ks_pip.copy(), p3_ks_pim.copy(), p3_phi_pip.copy(), p3_phi_pim.copy()
    p4_ks_pip, p4_ks_pim, p4_phi_pip, p4_phi_pim = [eg.p3top4(p, mpi) for p in [p3_ks_pip, p3_ks_pim, p3_phi_pip, p3_phi_pim]]

    p4ks = p4_ks_pip + p4_ks_pim
    p4phi = p4_phi_pip + p4_phi_pim
    p4d0 = p4ks + p4phi

    ndim = 24
    Ck = np.zeros((N, ndim, ndim))
    Ck[:] = 10**3*np.eye(ndim) * eg.UNIT**2
    xi = np.column_stack([p3_ks_pip, p3_ks_pim, p4ks, p3_phi_pip, p3_phi_pim, p4phi, p4d0])

    logs = {key: [] for key in ['xi', 'cov', 'chi2', 'mk', 'chi2v0']}
    xi0 = xi.copy()

    logs['xi'].append(xi.copy())

    for idx in range(nit):
        print('Iteration {}'.format(idx+1))
        Ck = np.zeros((N, ndim, ndim))
        Ck[:] = 10**4*np.eye(ndim) * eg.UNIT**2
        chi2 = np.zeros(N)

        # Apply pi+ momentum measurement constraint #
        Hk = np.zeros((N, 3, ndim))
        Hk[:,:, :3] = np.eye(3)
        rk = xi[:,:3] - xi0[:,:3]
        dxi, Ck, chi2k = apply_meas(Hk, -rk, cov, Ck)
        xi += dxi
        chi2 += chi2k

        # Apply pi- momentum measurement constraint #
        Hk = np.zeros((N, 3, ndim))
        Hk[:,:,3:6] = np.eye(3)
        rk = xi[:,3:6] - xi0[:,3:6]
        dxi, Ck, chi2k = apply_meas(Hk, -rk, cov, Ck)
        xi += dxi
        chi2 += chi2k
      
        # Apply pi+ momentum measurement constraint #
        Hk = np.zeros((N, 3, ndim))
        Hk[:,:, 10:13] = np.eye(3)
        rk = xi[:,10:13] - xi0[:,10:13]
        dxi, Ck, chi2k = apply_meas(Hk, -rk, cov, Ck)
        xi += dxi
        chi2 += chi2k

        # Apply pi- momentum measurement constraint #
        Hk = np.zeros((N, 3, ndim))
        Hk[:,:,13:16] = np.eye(3)
        rk = xi[:,13:16] - xi0[:,13:16]
        dxi, Ck, chi2k = apply_meas(Hk, -rk, cov, Ck)
        xi += dxi
        chi2 += chi2k      

       # momentum conservation constraint #
        
        p4_ks_pip, p4_ks_pim, p4ks, _, _, _, _, _, _, _, _ = cascade_unpack(xi)
        gp_const = rf.gmomentum(p4_ks_pip, p4_ks_pim, p4ks)
        for _ in range(gpit):
            p4_ks_pip, p4_ks_pim, p4ks, e_ks_pip, e_ks_pim, _, _, _, _, _, _ = cascade_unpack(xi)
            gp = rf.gmomentum(p4_ks_pip, p4_ks_pim, p4ks)
            Gk = np.zeros((N, 4, ndim))
            Gk[:, 0, :3] = -p4_ks_pip[:, 1:4] / e_ks_pip
            Gk[:, 0,3:6] = -p4_ks_pim[:, 1:4] / e_ks_pim
            Gk[:, 1:, :3] = Gk[:, 1:, 3:6] = -np.eye(3)
            Gk[:, :, 6:10] = np.eye(4)
            GCGTInv = gcgtinv(Gk, Ck)
            Kk = gain_exact(Ck, Gk, GCGTInv)
            xi += xi_upd(Kk, gp)
        Ck = covariance_exact(Ck, Kk, Gk)
        chi2 += et.chi2_item(gp_const, GCGTInv)
        
#=================================================================================
       # momentum conservation constraint #
        _, _, _, _, _, p4_phi_pip, p4_phi_pim, p4phi, e_phi_pip, e_phi_pim, p4d0 = cascade_unpack(xi)
        gp_const = rf.gmomentum(p4_phi_pip, p4_phi_pim, p4phi)
        for _ in range(gpit):
            _, _, _, _, _, p4_phi_pip, p4_phi_pim, p4phi, e_phi_pip, e_phi_pim, p4d0 = cascade_unpack(xi)
            gp = rf.gmomentum(p4_phi_pip, p4_phi_pim, p4phi)
            Gk = np.zeros((N, 4, ndim))
            Gk[:, 0, 10:13] = -p4_phi_pip[:, 1:4] / e_phi_pip
            Gk[:, 0, 13:16] = -p4_phi_pim[:, 1:4] / e_phi_pim
            Gk[:, 1:,10:13] = Gk[:, 1:, 13:16] = -np.eye(3)
            Gk[:, :, 16:20] = np.eye(4)
            GCGTInv = gcgtinv(Gk, Ck)
            Kk = gain_exact(Ck, Gk, GCGTInv)
            xi += xi_upd(Kk, gp)
        Ck = covariance_exact(Ck, Kk, Gk)
        chi2 += et.chi2_item(gp_const, GCGTInv)

       # momentum conservation constraint #
        _, _, p4ks, _, _, _, _, p4phi, _, _, p4d0 = cascade_unpack(xi)
        gp_const = rf.gmomentum(p4ks, p4phi, p4d0)
        for _ in range(gpit):
            _, _, p4ks, _, _, _, _, p4phi, _, _, p4d0 = cascade_unpack(xi)
            gp = rf.gmomentum(p4ks, p4phi, p4d0)
            Gk = np.zeros((N, 4, ndim))
            Gk[:, :, 6:10] = -np.eye(4)
            Gk[:, :, 16:20] = -np.eye(4)
            Gk[:, :, 20:] = np.eye(4)
            GCGTInv = gcgtinv(Gk, Ck)
            Kk = gain_exact(Ck, Gk, GCGTInv)
            xi += xi_upd(Kk, gp)
        Ck = covariance_exact(Ck, Kk, Gk)
        chi2 += et.chi2_item(gp_const, GCGTInv)

        def gmass(mass, p4):
            return mass**2 - rf.mass_sq(p4)

        # mass constraint #
        _, _, p4ks, _, _, _, _, p4phi, _, _, p4d0 = cascade_unpack(xi)
        gm_const = gmass(mks, p4ks).reshape(-1, 1)
        for _ in range(gmit):
            print('Mass iteration {}'.format(_ + 1))
            _, _, p4ks, _, _, _, _, p4phi, _, _, p4d0 = cascade_unpack(xi)
            gm = gmass(mks, p4ks).reshape(-1, 1)
            Gk = np.zeros((N, 1, ndim))
            Gk[:,0,6:10] = 2 * np.einsum('ki, i -> ki', p4ks, np.array([-1, 1, 1, 1]))
            GCGTInv = gcgtinv(Gk, Ck)
            Kk = gain_exact(Ck, Gk, GCGTInv)
            xi += xi_upd(Kk, gm)
        Ck = covariance_exact(Ck, Kk, Gk)
        chi2 += et.chi2_item(gm_const, GCGTInv)

        _, _, p4ks, _, _, _, _, p4phi, _, _, p4d0 = cascade_unpack(xi)
        gm_const = gmass(mphi, p4phi).reshape(-1, 1)
        for _ in range(gmit):
            print('Mass iteration {}'.format(_ + 1))
            _, _, p4ks, _, _, _, _, p4phi, _, _, p4d0 = cascade_unpack(xi)
            gm = gmass(mphi, p4phi).reshape(-1, 1)
            Gk = np.zeros((N, 1, ndim))
            Gk[:,0,16:20] = 2 * np.einsum('ki, i -> ki', p4phi, np.array([-1, 1, 1, 1]))
            GCGTInv = gcgtinv(Gk, Ck)
            Kk = gain_exact(Ck, Gk, GCGTInv)
            xi += xi_upd(Kk, gm)
        Ck = covariance_exact(Ck, Kk, Gk)
        chi2 += et.chi2_item(gm_const, GCGTInv)

        _, _, p4ks, _, _, _, _, p4phi, _, _, p4d0 = cascade_unpack(xi)
        gm_const = gmass(md0, p4d0).reshape(-1, 1)
        for _ in range(gmit):
            print('Mass iteration {}'.format(_ + 1))
            _, _, p4ks, _, _, _, _, p4phi, _, _, p4d0 = cascade_unpack(xi)
            gm = gmass(md0, p4d0).reshape(-1, 1)
            Gk = np.zeros((N, 1, ndim))
            Gk[:,0,20:24] = 2 * np.einsum('ki, i -> ki', p4d0, np.array([-1, 1, 1, 1]))
            GCGTInv = gcgtinv(Gk, Ck)
            Kk = gain_exact(Ck, Gk, GCGTInv)
            xi += xi_upd(Kk, gm)
        Ck = covariance_exact(Ck, Kk, Gk)
        chi2 += et.chi2_item(gm_const, GCGTInv)


        # write log #
        _, _, p4ks, _, _, _, _, p4phi, _, _, p4d0 = cascade_unpack(xi)
        logs['xi'].append(xi.copy())
        logs['cov'].append(Ck.copy())
        # logs['chi2v0'].append(rf.chi2(p3pip0, p3pim0, p4pip, p4pim, covInv))
        logs['chi2'].append(chi2)
        logs['mk'].append(eg.mass(p4ks))

    # print('Final Ck\n{}'.format(Ck[0]))
    return logs


def main():
    from event_generator import generate, generate_cascade
    # et.VERB = True
    cov = np.diag([3,3,5])**2 * eg.UNIT**2
    N = 10**3

    is_cascade_decay = True * 0

    for energy in [0, 250, 1000]:
        ptot = np.array([energy, 0, 0])
        if energy == 0:
            ptot = None
  
        if is_cascade_decay is True:
            (p3_ks_pip, p3_ks_pim, p3_phi_pip, p3_phi_pim), \
            p3_ks_pip_gen, p3_ks_pim_gen, p3_phi_pip_gen, p3_phi_pim_gen \
                = generate_cascade(N, cov, ptot=ptot)

            logs = pfit_to_d0(p3_ks_pip, p3_ks_pim, p3_phi_pip, p3_phi_pim, cov, nit=5, gpit=1, gmit=1)

            pathlib.Path('logs/d_meson/kalman').mkdir(parents=True, exist_ok=True) 
            np.savez('logs/d_meson/kalman/fitres_{:.1f}MeV'.format(energy),
                        chi2=logs['chi2'],
                    chi2v0=logs['chi2v0'],
                        Ck=logs['cov'],
                        mk=logs['mk'],
                        xi=logs['xi'],
                p3_ks_pip_gen=p3_ks_pip_gen,
                p3_ks_pim_gen=p3_ks_pim_gen,
            p3_phi_pip_gen=p3_phi_pip_gen,
            p3_phi_pim_gen=p3_phi_pim_gen,
                        cov=cov
            )
        else:
            (p3pip, p3pim), p3pipGen, p3pimGen = generate(N, cov, ptot=ptot)
            logs = pfit_to_ks(p3pip, p3pim, cov, nit=5, gpit=1, gmit=1)

            pathlib.Path('logs/kaon/kalman').mkdir(parents=True, exist_ok=True) 
            np.savez('logs/kaon/kalman/fitres_{:.1f}MeV'.format(energy),
                chi2=logs['chi2'],
            chi2v0=logs['chi2v0'],
                Ck=logs['cov'],
                mk=logs['mk'],
                xi=logs['xi'],
            pipgen=p3pipGen,
            pimgen=p3pimGen,
                cov=cov
            )

if __name__ == '__main__':
    main()
