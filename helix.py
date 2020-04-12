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

def apply_meas(Hk, rk, cov, Ck, full=True):
    """ Apply measurement constraint """
    Rlk = predicted_resid_uncert(cov, Hk, Ck)
    RlkInv = np.linalg.inv(Rlk)
    Kk = gain(Ck, Hk, RlkInv)
    if full:
        return (xi_upd(Kk, rk), covariance_full(Ck, Kk, Hk, cov), et.chi2_item(rk, RlkInv))
    return (xi_upd(Kk, rk), covariance_normal(Ck, Kk, Hk, Rlk), et.chi2_item(rk, RlkInv))

################################################################################

def passiveMoveBy(byX, byY, byZ, helix):
    arcLength2D, new_d0 = calcArcLength2DAndDrAtXY(byX, byY, helix)

#   // Third the new phi0 and z0 can be calculated from the arc length
    _, phi0, omega, z0, tandip = unpack_helix(helix)

    chi = -arcLength2D * omega
    new_phi0 = phi0 + chi
    new_z0 = z0 - byZ + tandip * arcLength2D

#   /// Update the parameters inplace. Omega and tan lambda are unchanged
    # m_d0 = new_d0
    # m_phi0 = new_phi0
    # m_z0 = new_z0

    new_helix = (new_d0, new_phi0, omega, new_z0, tandip)

    return new_helix
    # return arcLength2D


def calcArcLength2DAndDrAtXY(x, y, helix):
#   // Prepare common variables
    d0, phi0, omega, _, _ = unpack_helix(helix)

    # omega = getOmega()
    cosPhi0 = np.cos(phi0) 
    sinPhi0 = np.sin(phi0)
    # d0 = getD0()

    deltaParallel = x * cosPhi0 + y * sinPhi0
    deltaOrthogonal = y * cosPhi0 - x * sinPhi0 + d0
    deltaCylindricalR = np.hypot(deltaOrthogonal, deltaParallel)
    deltaCylindricalRSquared = deltaCylindricalR * deltaCylindricalR

    A = 2 * deltaOrthogonal + omega * deltaCylindricalRSquared
    U = np.sqrt(1 + omega * A)
    UOrthogonal = 1 + omega * deltaOrthogonal #// called nu in the Karimaki paper.
    UParallel = omega * deltaParallel
#   // Note U is a vector pointing from the middle of the projected circle scaled by a factor omega.

#   // Calculate dr
    dr = A / (1 + U)

    # // Calculate the absolute value of the arc length
    chi = -np.arctan2(UParallel, UOrthogonal)

    if (np.fabs(chi) < np.pi / 8): #// Rough guess where the critical zone for approximations begins
    # // Close side of the circle
        principleArcLength2D = deltaParallel / UOrthogonal

        def calcATanXDividedByX(x):
            if x < 1e-9:
                return 1
            return np.arctan(x) / x

        arcLength2D = principleArcLength2D * calcATanXDividedByX(principleArcLength2D * omega)
    else:
    # // Far side of the circle
    # // If the far side of the circle is a well definied concept meaning that we have big enough omega.
        arcLength2D = -chi / omega
    
    return arcLength2D, dr


def setCartesian(position, momentum, charge, bZ):
    #   assert(abs(charge) <= 1);  // Not prepared for doubly-charged particles.
    alpha = getAlpha(bZ)

    #   // We allow for the case that position, momentum are not given
    #   // exactly in the perigee.  Therefore we have to transform the momentum
    #   // with the position as the reference point and then move the coordinate system
    #   // to the origin.

    x = position[0]
    y = position[1]
    z = position[2]

    px = momentum[0]
    py = momentum[1]
    pz = momentum[2]

    ptinv = 1 / np.hypot(px, py)
    omega = charge * ptinv / alpha
    tanLambda = ptinv * pz
    phi0 = np.arctan2(py, px)
    z0 = z
    d0 = 0

    helix = d0, phi0, omega, z0, tanLambda
    
    return passiveMoveBy(-x, -y, 0, helix)

################################################################################

# generator give us 2 helix. We have also magnetic field B = 1.5 Tesla
# first, we need to reconstruct internal particle via POCA


def phidomain(phi):
    rc = phi
    if phi < -np.pi:
        rc += 2 * np.pi
    elif phi > np.pi:
        rc -= 2 * np.pi
    return rc


def unpack_helix(helix):
    return helix[0], helix[1], helix[2], helix[3], helix[4]


def helixPoca(helix1, helix2):
    d0_1, phi0_1, omega_1, z0_1, tandip_1 = unpack_helix(helix1)
    d0_2, phi0_2, omega_2, z0_2, tandip_2 = unpack_helix(helix2)

    r_1 = 1 / omega_1 - 1
    r_2 = 1 / omega_2 - 1

    x0_1 = -(r_1 + d0_1) * np.sin(phi0_1)
    y0_1 = (r_1 + d0_1) * np.cos(phi0_1)

    x0_2 = -(r_2 + d0_2) * np.sin(phi0_2)
    y0_2 = (r_2 + d0_2) * np.cos(phi0_2)

    deltax = x0_2 - x0_1
    deltay = y0_2 - y0_1

    phi1 = [0, 0] ########################################
    phi2 = [0, 0]
    nsolutions = 1

    phi = -np.arctan2(deltax, deltay)
    phinot = phi - np.pi if phi > 0 else phi + np.pi
    phi1[0] = phi if r_1 < 0 else phinot
    phi2[0] = phi if r_2 > 0 else phinot

    R1 = np.abs(r_1)
    R2 = np.abs(r_2)
    Rmin = R1 if R1 < R2 else R2
    Rmax = R1 if R1 > R2 else R2
    dX = np.sqrt(deltax * deltax + deltay * deltay)

    if dX + Rmin > Rmax and dX < R1 + R2:
    #   there are two solutions
        nsolutions = 2
        ddphi1 = np.arccos((dX * dX - R2 * R2 + R1 * R1) / (2.*dX * R1))
        phi1[1] = phidomain(phi1[0] + ddphi1)
        phi1[0] = phidomain(phi1[0] - ddphi1)

        ddphi2 = np.arccos((dX * dX - R1 * R1 + R2 * R2) / (2.*dX * R2))
        phi2[1] = phidomain(phi2[0] - ddphi2)
        phi2[0] = phidomain(phi2[0] + ddphi2)

    elif dX < Rmax:
        if R1 > R2:
            phi2[0] = phi if r_2 < 0 else phinot
        else:
            phi1[0] = phi if r_1 < 0 else phinot

    z1 = 0
    z2 = 0
    first = True
    ibest = 0
    ncirc = 2
    for i in range(nsolutions): #(int i = 0; i < nsolutions; ++i) {
        dphi1 = phidomain(phi1[i] - phi0_1)
        dphi2 = phidomain(phi2[i] - phi0_2)
        for n1 in range(1 - ncirc, 1 + (1 + ncirc)):
            l1 = (dphi1 + n1 * 2 * np.pi) / omega_1
            tmpz1 = (z0_1 + l1 * tandip_1)
            if (n1 == 0 or np.fabs(tmpz1) < 100):
                for n2 in range(1 - ncirc, 1 + (1 + ncirc)):
                    l2 = (dphi2 + n2 * 2 * np.pi) / omega_2
                    tmpz2 = (z0_2 + l2 * tandip_2)
                    if (n2 == 0 or np.fabs(tmpz2) < 100):
                        if (first or (np.fabs(tmpz1 - tmpz2) < np.fabs(z1 - z2))):
                            ibest = i
                            first = False
                            z1 = tmpz1
                            z2 = tmpz2
                            flt1 = l1 / np.cos(np.arctan(tandip_1))
                            flt2 = l2 / np.cos(np.arctan(tandip_2))

    x1 =  r_1 * np.sin(phi1[ibest]) + x0_1
    y1 = -r_1 * np.cos(phi1[ibest]) + y0_1

    x2 =  r_2 * np.sin(phi2[ibest]) + x0_2
    y2 = -r_2 * np.cos(phi2[ibest]) + y0_2

    x = (0.5 * (x1 + x2))
    y = (0.5 * (y1 + y2))
    z = (0.5 * (z1 + z2))
    vtx = np.array([x, y, z])
    return flt1, flt2, vtx
    

def getAlpha(bZ):   
    speedOfLight = 29.9792458
    return 1.0 / (bZ * speedOfLight) * 1e4


def getTransverseMomentum(bZ, omega):
    return 1 / np.fabs(getAlpha(bZ) * omega)


def getTangentialAtArcLength2D(arcLength2D, helix):
    _, phi0, omega, _, tanLambda = unpack_helix(helix)

    chi = - omega * arcLength2D

    tx = np.cos(chi + phi0)
    ty = np.sin(chi + phi0)
    tz = tanLambda

    tangential = np.array([tx, ty, tz])
    return tangential

def getMomentumAtArcLength2D(helix, flt, bfield):
    momentum = getTangentialAtArcLength2D(flt, helix)
    pr = getTransverseMomentum(bfield, helix[2])
    momentum *= pr

    return momentum


def init_helix_momentum(helix, flt):
    bfield = 1.5
    p = getMomentumAtArcLength2D(helix, flt, bfield)
    return p

#############################################################

def init_decay_chain(helix_pip, helix_pim):
    flt1, flt2, vtx = helixPoca(helix_pip, helix_pim)
    p1 = init_helix_momentum(helix_pip, flt1)
    p2 = init_helix_momentum(helix_pim, flt2)

    p = p1 + p2
    E = 0

    return vtx, p, E, p1, p2


def generate_helix(pip, pim, r):
    bZ = 1.5
    helix_p = setCartesian(r, pip, +1, bZ)
    helix_m = setCartesian(r, pim, -1, bZ)

    return helix_p, helix_m

# What with fitting? All right, use fit as in prev model. 
# Just recalculate Jacobians


################################################################################

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


def main():
    from event_generator import generate, generate_cascade
    from event_generator import UNIT, MASS_DICT, p3top4, mass_sq, mass, make_hist
    N = 500
    cov = np.diag([3,3,5])**2 * UNIT**2
    init_vtx = [10, -10, 0]
    dr = []
    dpp = []
    dpm = []

    (p3pip, p3pim), p3pipGen, p3pimGen = generate(N, cov, ptot=np.array([100, 200, 1]))
    for i in range(N):
        pip, pim = p3pipGen[i], p3pimGen[i]
        helix_p, helix_m = generate_helix(pip, pim, init_vtx)
        vtx, p, E, p_p, p_m = init_decay_chain(helix_p, helix_m)
        
        dr.append((init_vtx - vtx)[2])
        dpp.append((pip - p_p))
        dpm.append((pim - p_m)[2])

    from matplotlib import pyplot as plt
    plt.hist(dr, bins=30, alpha=0.5)
    # np.hist(dr)
    plt.show()

    # helix_p, helix_m = generate_helix([100, 100, 0], [100, -150, 0], [10, 10, 10])
    # vtx, p, E, p_p, p_m = init_decay_chain(helix_p, helix_m)
    # vtx, p, E, p_p, p_m = init_decay_chain(helix_m, helix_p)

    # print(helix_p)
    # print(helix_m)
    # print(p_p)
    # print(p_m)
    # print(vtx)
 
if __name__ == '__main__':
    main()
