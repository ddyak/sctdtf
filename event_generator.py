#! /usr/bin/env python3

""" Simple toy event generator for kine fitter test """

import numpy as np
import unittest
import copy

# UNIT = 10**-3
UNIT = 1

MASS_DICT = {
    'K0_S': 497.611 * UNIT,
    'pi+': 139.57018 * UNIT,
    'D0': 1865.84 * UNIT,
    'phi': 1019.461 * UNIT,
    # 'pi0' : 134.9766,
}


def make_hist(data, range=None, nbins=100, density=False):
    if range is None:
        range = (np.min(data), np.max(data))
    dhist, dbins = np.histogram(data, bins=nbins, density=density, range=range)
    dbins = 0.5 * (dbins[1:] + dbins[:-1])
    norm = np.sum(dhist) / data.shape[0]
    errors = np.array([-0.5 + np.sqrt(dhist / norm + 0.25),
                       0.5 + np.sqrt(dhist / norm + 0.25)]) * norm
    return (dbins, dhist, errors)


def gamma(beta):
    """ Lorentz factor """
    return 1. / np.sqrt(1. - beta**2)


def lorentz_boost(lv, bv):
    """ Relativistic transformation """
    beta = np.linalg.norm(bv)
    gam, n = gamma(beta), bv / beta
    t, r = lv[:, 0].reshape(-1, 1), lv[:, 1:]
    return np.column_stack([gam * t - beta * np.dot(r, n.T),
                            r + (gam - 1) * np.dot(r, n.T) * n - gam * t * beta * n])


def energy(mass, p3):
    """ Energy from mass and 3-momentum """
    return np.sqrt(mass**2 + np.sum(p3**2, axis=-1))


def ks2pipi(N, pk=None):
    """ Generator of the Ks0 -> pi+ pi- decays """
    mks, mpi = [MASS_DICT[key] for key in ['K0_S', 'pi+']]
    epi = 0.5*mks
    ppi = np.sqrt(epi**2 - mpi**2)
    costh = 2.*np.random.rand(N) - 1
    phi = 2.*np.random.rand(N)*np.pi
    sinth = np.sqrt(1. - costh**2)
    sinphi, cosphi = np.sin(phi), np.cos(phi)
    p3pip = ppi*np.array([sinth*cosphi, sinth*sinphi, costh]).T

    if pk is not None:
        p4pip, p4pim = [np.empty((p3pip.shape[0], 4)) for _ in range(2)]
        p4pip[:, 0], p4pim[:, 0] = epi, epi
        p4pip[:, 1:], p4pim[:, 1:] = p3pip, -p3pip
        bv = -(pk.reshape(-1, 1) / energy(mks, pk)).T
        p4pip, p4pim = [lorentz_boost(x, bv) for x in [p4pip, p4pim]]
        return (p4pip[:, 1:], p4pim[:, 1:])

    return (p3pip, -p3pip)


def decay_to_two(mmother, md1, md2, pk=None):
    """ Generator of the decay A -> B C """
    # mmother, mphi, mks, mpi = [MASS_DICT[key] for key in ['D0', 'phi' 'K0_S', 'pi+']]

    e_daughter1 = (mmother**2 + md1**2 - md2**2) / (2 * mmother)
    # mnp.sqrt(pks**2 + mks**2)
    # pks = np.sqrt((md0**2 - mphi**2 - mks**2) / 2)
    p_daughter = np.sqrt(e_daughter1**2 - md1**2)
    # p_daughter = np.sqrt((mmother**2 - md1**2 - md2**2) / 2)
    # e_daughter1 = np.sqrt(p_daughter**2 + md1**2)
    costh = 2.*np.random.rand(pk.shape[0]) - 1
    phi = 2.*np.random.rand(pk.shape[0])*np.pi
    sinth = np.sqrt(1. - costh**2)
    sinphi, cosphi = np.sin(phi), np.cos(phi)
    p3daughter1 = p_daughter*np.array([sinth*cosphi, sinth*sinphi, costh]).T

    if pk is not None:
        p4daughter1, p4daughter2 = [
            np.empty((p3daughter1.shape[0], 4)) for _ in range(2)]
        p4daughter1[:, 0] = e_daughter1
        p4daughter2[:, 0] = mmother - e_daughter1
        p4daughter1[:, 1:], p4daughter2[:, 1:] = p3daughter1, -p3daughter1

        # print(pk.reshape(-1, 3, 1))
        # print('==================')
        # print(energy(mmother, pk)[0])

        bv = -(pk.reshape(-1, 3, 1) / energy(mmother, pk)[0]).T[0].T
        # bv = -(pk.reshape(-1, 3, 1) / energy(mmother, pk)).T[0].T
        # print(bv)
        p4daughter1 = np.array([lorentz_boost(np.array(
            [p4daughter1[idx]]), bv[idx].reshape(1, -1)) for idx in range(bv.shape[0])])
        p4daughter2 = np.array([lorentz_boost(np.array(
            [p4daughter2[idx]]), bv[idx].reshape(1, -1)) for idx in range(bv.shape[0])])
        # p4daughter2 = np.array([lorentz_boost(p4daughter2, b.reshape(1, -1)) for b in bv])
        # p4daughter1 = np.array([lorentz_boost(p4daughter1, b.reshape(1, -1)) for b in bv])
        p4daughter1 = p4daughter1.reshape(-1, 4)
        p4daughter2 = p4daughter2.reshape(-1, 4)
        return p4daughter1[:, 1:], p4daughter2[:, 1:]

    return (p3daughter1, -p3daughter1)


def d0_to_ks_phi(N, pk=None):
    # def d0_to_ks_phi(N, pk=np.array([1,0,0])):
    """ Generator of the D0 -> Ks0 R decays """
    md0, mphi, mks, mpi = [MASS_DICT[key]
                           for key in ['D0', 'phi', 'K0_S', 'pi+']]
    eks = (md0**2 + mks**2 - mphi**2) / (2 * md0)
    # mnp.sqrt(pks**2 + mks**2)
    # pks = np.sqrt((md0**2 - mphi**2 - mks**2) / 2)
    pks = np.sqrt(eks**2 - mks**2)
    costh = 2.*np.random.rand(N) - 1
    phi = 2.*np.random.rand(N)*np.pi
    sinth = np.sqrt(1. - costh**2)
    sinphi, cosphi = np.sin(phi), np.cos(phi)
    p3ks = pks*np.array([sinth*cosphi, sinth*sinphi, costh]).T

    if pk is not None:
        p4ks, p4phi = [np.empty((p3ks.shape[0], 4)) for _ in range(2)]
        p4ks[:, 0] = eks
        p4phi[:, 0] = md0 - eks
        p4ks[:, 1:], p4phi[:, 1:] = p3ks, -p3ks
        bv = -(pk.reshape(-1, 1) / energy(md0, pk)).T

#     (p3_ks_pip, p3_ks_pim), (p3_phi_pip, p3_phi_pim) = d0_to_ks_phi(N, ptot)

        (p3_ks_pip, p3_ks_pim), (p3_phi_pip, p3_phi_pim) = decay_to_two(mks, mpi, mpi, pk=p4ks[:, 1:]), decay_to_two(mphi, mpi, mpi, pk=p4phi[:, 1:])

        p4_ks_pip = p3top4(p3_ks_pip, MASS_DICT['pi+'])
        p4_ks_pim = p3top4(p3_ks_pim, MASS_DICT['pi+'])
        p4_phi_pip = p3top4(p3_phi_pip, MASS_DICT['pi+'])
        p4_phi_pim = p3top4(p3_phi_pim, MASS_DICT['pi+'])

        # print(p4_ks_pip + p4_ks_pim + p4_phi_pip + p4_phi_pim)
        # print(np.sqrt(np.sum((p4_ks_pip + p4_ks_pim + p4_phi_pip + p4_phi_pim)[:, 1:]**2, axis=1)))

        p4_ks_pip, p4_ks_pim, p4_phi_pip, p4_phi_pim = [lorentz_boost(x, bv) for x in [p4_ks_pip, p4_ks_pim, p4_phi_pip, p4_phi_pim]]

        # p4ks, p4phi = [lorentz_boost(x, bv) for x in [p4ks, p4phi]]
        # return decay_to_two(mks, mpi, mpi, pk=p4ks[:, 1:]), decay_to_two(mphi, mpi, mpi, pk=-p4phi[:, 1:])
        # return (p4ks[:, 1:], p4phi[:, 1:])

        # print(p4_ks_pip + p4_ks_pim + p4_phi_pip + p4_phi_pim)
        # print(np.sqrt(np.sum((p4_ks_pip + p4_ks_pim + p4_phi_pip + p4_phi_pim)[:, 1:]**2, axis=1)))

        return (p4_ks_pip[:, 1:], p4_ks_pim[:, 1:]), (p4_phi_pip[:, 1:], p4_phi_pim[:, 1:])

    (a, b), (c, d) = decay_to_two(mks, mpi, mpi, pk=p3ks), decay_to_two(mphi, mpi, mpi, pk=-p3ks)

    # print(np.sqrt(np.sum((a+b+c+d)[:, 1:]**2, axis=1)))

    return decay_to_two(mks, mpi, mpi, pk=p3ks), decay_to_two(mphi, mpi, mpi, pk=-p3ks)


def mass_sq(p4):
    return p4[:, 0]**2 - np.sum(p4[:, 1:]**2, axis=-1)


def mass(p4):
    return np.sqrt(mass_sq(p4))


def p3top4(p3, mass):
    return np.column_stack([energy(mass, p3), p3])


def measurement_sampler(p3pip, p3pim, cov):
    """ Samples measurement error """
    assert cov.shape == (3, 3)
    N = p3pip.shape[0]
    dp = np.random.multivariate_normal([0, 0, 0], cov, 2*N)
    p3pip += dp[:N]
    p3pim += dp[N:]
    return (p3pip, p3pim)


def generate_cascade(N, cov, ptot=None):
    """ Generates N cascade events for a given covariance matrix """
    (p3_ks_pip, p3_ks_pim), (p3_phi_pip, p3_phi_pim) = d0_to_ks_phi(N, ptot)
    return ((*measurement_sampler(copy.deepcopy(p3_ks_pip), copy.deepcopy(p3_ks_pim), cov),
             *measurement_sampler(copy.deepcopy(p3_phi_pip), copy.deepcopy(p3_phi_pim), cov)),
            p3_ks_pip, p3_ks_pim, p3_phi_pip, p3_phi_pim)


def generate(N, cov, ptot=None):
    """ Generates N events for a given covariance matrix """
    p3pip, p3pim = ks2pipi(N, ptot)
    return (measurement_sampler(copy.deepcopy(p3pip), copy.deepcopy(p3pim), cov), p3pip, p3pim)


class TestGenerator(unittest.TestCase):
    N = 10**4
    ptot = 10**3

    @staticmethod
    def check_mass(p3pip, p3pim):
        """ Compares pi+pi- invariant mass and Ks0 mass """
        mks, mpi = [MASS_DICT[key] for key in ['K0_S', 'pi+']]
        p4pip, p4pim = [p3top4(p3, mpi) for p3 in [p3pip, p3pim]]
        return np.allclose(mks**2 * np.ones(p3pip.shape[0]), mass_sq(p4pip + p4pim))

    @staticmethod
    def check_momentum(p3pip, p3pim, ptot):
        """ Checks pi+pi- total momentum """
        return np.allclose(ptot, p3pip + p3pim)

    def test_k0s_frame_mass(self):
        p3pip, p3pim = ks2pipi(TestGenerator.N)
        self.assertTrue(TestGenerator.check_mass(p3pip, p3pim))

    def test_k0s_frame_momentum(self):
        p3pip, p3pim = ks2pipi(TestGenerator.N)
        self.assertTrue(TestGenerator.check_momentum(
            p3pip, p3pim, np.zeros(3)))

    def test_lab_frame_mass(self):
        ptot = TestGenerator.ptot*np.random.rand(3)
        p3pip, p3pim = ks2pipi(TestGenerator.N, ptot)
        self.assertTrue(TestGenerator.check_mass(p3pip, p3pim))

    def test_lab_frame_momentum(self):
        ptot = TestGenerator.ptot*np.random.rand(3)
        p3pip, p3pim = ks2pipi(TestGenerator.N, ptot)
        self.assertTrue(TestGenerator.check_momentum(p3pip, p3pim, ptot))


def resolution_plot():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    mpi = MASS_DICT['pi+']
    cov = np.diag([3, 3, 5])**2
    N = 10**5
    print(generate(N, cov))
    p4pip, p4pim = [p3top4(p, mpi) for p in generate(N, cov)[0]]
    x, bins, e = make_hist(mass(p4pip + p4pim))

    plt.figure(figsize=(6, 5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    plt.minorticks_on()
    plt.grid(which='both')
    plt.xlabel(r'$m(\pi^+\pi^-)$ (MeV)', fontsize=16)
    plt.tight_layout()
    plt.savefig('mpipi.pdf')
    plt.show()


def cascade_resolution_plot():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    mpi = MASS_DICT['pi+']
    cov = np.diag([3, 3, 5])**2
    N = 10**5
    p3_ks_pip, p3_ks_pim, p3_phi_pip, p3_phi_pim = [
        p3top4(p, mpi) for p in generate_cascade(N, cov)[0]]
    # x, bins, e = make_hist(mass(p3_ks_pip + p3_ks_pim + p3_phi_pip + p3_phi_pim))
    # x, bins, e = make_hist(mass(p3_ks_pip + p3_ks_pim))
    x, bins, e = make_hist(mass(p3_phi_pip + p3_phi_pim))
    plt.figure(figsize=(6, 5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    # plt.minorticks_on()
    plt.grid(which='both')
    plt.xlabel(r'$m(\phi)$ (MeV)', fontsize=16)
    plt.tight_layout()
    plt.savefig('mphi.pdf')
    plt.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        unittest.main()
    # resolution_plot()
    cascade_resolution_plot()
    # cov = np.diag([3,3,5])**2
    # print(generate_cascade(1, cov))
    # print(generate(1, cov))
