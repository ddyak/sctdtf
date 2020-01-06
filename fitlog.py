#! /usr/bin/env python3

import numpy as np
from event_generator import mass, make_hist

def plot_ks0_mass(xi):
    """ """
    import matplotlib.pyplot as plt
    m = mass(xi[:, 6:10])
    print(np.mean(m), np.std(m))
    x, bins, e = make_hist(m)

    plt.figure(figsize=(6,5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    plt.grid()
    plt.tight_layout()
    plt.xlabel(r'$m(\pi^+\pi^-)$ (MeV)', fontsize=16)
    plt.show()

def read_log(niter):
    data = np.load('logs/fitres.npz')

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

if __name__ == '__main__':
    read_log(4)
