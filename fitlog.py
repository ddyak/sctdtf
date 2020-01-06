#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2

from event_generator import mass, make_hist

def plot_ks0_mass(xi):
    """ """
    m = mass(xi[:, 6:10])
    m = m[~np.isnan(m)]
    for _ in range(5):
        mean, std = np.mean(m), np.std(m)
        m = m[np.abs(m - mean) < 5.*std]
    print(m.shape, np.mean(m), np.std(m))
    # x, bins, e = make_hist(m, range=[497.601, 498.621])

    # plt.figure(figsize=(6,5))
    # plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    # plt.grid()
    # plt.xlabel(r'$m(\pi^+\pi^-)$ (MeV)', fontsize=16)
    # plt.tight_layout()
    # plt.show()

def plot_chi2(chisq):
    chisq = chisq[~np.isnan(chisq)]
    rng = [0, 10]
    nbins = 100
    for _ in range(5):
        mean, std = np.mean(chisq), np.std(chisq)
        chisq = chisq[np.abs(chisq - mean) < 5.*std]
    print(chisq.shape, chisq.mean(), chisq.std())
    x, bins, e = make_hist(chisq, range=rng, nbins=nbins, density=False)

    plt.figure(figsize=(6,5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    norm = chisq.shape[0]*(rng[1]-rng[0])/nbins
    plt.plot(x, norm*chi2.pdf(x, 1))
    plt.grid()
    plt.xlabel(r'$\chi^2$', fontsize=16)
    plt.tight_layout()

def print_log(data, niter):
    def print_hessian(h):
        for row in h:
            for item in row:
                print('{:9.3f}'.format(item), end=' ')
            print('')

    def print_xi(x):
        print('  p3pip', end=' ')
        for item in x[:3]:
            print('{:9.3f}'.format(item), end=' ')
        print('\n  p3pim', end=' ')
        for item in x[3:6]:
            print('{:9.3f}'.format(item), end=' ')
        print('\n    p4k', end=' ')
        for item in x[6:10]:
            print('{:9.3f}'.format(item), end=' ')
        print('\nlambdas', end=' ')
        for item in x[10:]:
            print('{:9.3f}'.format(item), end=' ')
        print('')

    for iter in range(niter):
        print('\nHessian iter {}'.format(iter))
        print_hessian(data['hess'][iter][0])

        print('\nInverse Hessian iter {}'.format(iter))
        print_hessian(np.linalg.inv(data['hess'][iter][0]))

        print('\nGradient iter {}'.format(iter))
        print_xi(data['grad'][iter][0])

        print('\nXi iter {}'.format(iter))
        print_xi(data['xi'][iter][0])

def main():
    data = np.load('logs/fitres.npz')
    plot_ks0_mass(data['xi'][-1])
    plot_chi2(data['chi2'][-1])
    plt.show()

if __name__ == '__main__':
    main()
    # # print_log(data, 4)
    # for idx in range(4):
    #     xi = data['xi'][idx]
    #     xi = xi[~np.isnan(xi).any(axis=1)]
    #     print(xi.shape)
    #     plot_ks0_mass(xi)
    # plt.show()
