#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.patches as mpl_patches

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

from scipy.stats import chi2
import scipy.stats as stats

import math

from event_generator import UNIT, MASS_DICT, p3top4, mass_sq, mass, make_hist, energy
import eintools as et

import pathlib

def plot_params_d0(xi, xi0, p3_ks_pip_gen, p3_ks_pim_gen, p3_phi_pip_gen, p3_phi_pim_gen, savedir):
    ks3 = p3_ks_pip_gen + p3_ks_pim_gen
    phi3 = p3_phi_pip_gen + p3_phi_pim_gen
    d03 = ks3 + phi3

    eks_gen = energy(MASS_DICT['K0_S'], ks3).reshape(-1, 1)
    ephi_gen = energy(MASS_DICT['phi'], phi3).reshape(-1, 1)
    ed0_gen = energy(MASS_DICT['D0'], d03).reshape(-1, 1)
    
    xi_gen = np.hstack((p3_ks_pip_gen, p3_ks_pim_gen, eks_gen, ks3,
                        p3_phi_pip_gen, p3_phi_pim_gen, ephi_gen, phi3,
                        ed0_gen, d03))

    if (savedir.split('/')[0] == 'reffit'):
        xi = np.hstack((xi[:, 0:10], xi[:,15:25], xi[:,30:34]))
        xi0 = np.hstack((xi0[:, 0:10], xi0[:,15:25], xi0[:,30:34]))

    def augment(xi):
        ks_pip_p = np.sqrt(np.sum(xi[:, 0:3]**2, axis=-1)).reshape(-1, 1)
        ks_pim_p = np.sqrt(np.sum(xi[:, 3:6]**2, axis=-1)).reshape(-1, 1)
        ks_p = np.sqrt(np.sum(xi[:, 7:10]**2, axis=-1)).reshape(-1, 1)
        ks_m = np.sqrt(xi[:, 6]**2 - np.sum(xi[:, 7:10]** 2, axis=-1)).reshape(-1, 1)
        
        phi_pip_p = np.sqrt(np.sum(xi[:, 10:13]**2, axis=-1)).reshape(-1, 1)
        phi_pim_p = np.sqrt(np.sum(xi[:, 13:16]**2, axis=-1)).reshape(-1, 1)
        phi_p = np.sqrt(np.sum(xi[:, 17:20]**2, axis=-1)).reshape(-1, 1)
        phi_m = np.sqrt(xi[:, 16]**2 - np.sum(xi[:, 17:20]** 2, axis=-1)).reshape(-1, 1)
        
        d0_p = np.sqrt(np.sum(xi[:, 21:24]**2, axis=-1)).reshape(-1, 1)
        d0_m = np.sqrt(xi[:, 20]**2 - np.sum(xi[:, 21:24]** 2, axis=-1)).reshape(-1, 1)
        
        ks_dpx = (xi[:, 0] + xi[:, 3] - xi[:, 7]).reshape(-1, 1)
        ks_dpy = (xi[:, 1] + xi[:, 4] - xi[:, 8]).reshape(-1, 1)
        ks_dpz = (xi[:, 2] + xi[:, 5] - xi[:, 9]).reshape(-1, 1)
        ks_dpe = (energy(MASS_DICT['pi+'], xi[:, 0:3]) + energy(MASS_DICT['pi+'], xi[:, 3:6]) - xi[:, 6]).reshape(-1, 1)

        phi_dpx = (xi[:, 10] + xi[:, 13] - xi[:, 17]).reshape(-1, 1)
        phi_dpy = (xi[:, 11] + xi[:, 14] - xi[:, 18]).reshape(-1, 1)
        phi_dpz = (xi[:, 12] + xi[:, 15] - xi[:, 19]).reshape(-1, 1)
        phi_dpe = (energy(MASS_DICT['pi+'], xi[:, 10:13]) + energy(MASS_DICT['pi+'], xi[:, 13:16]) - xi[:, 16]).reshape(-1, 1)

        d0_dpx = (xi[:, 7] + xi[:, 17] - xi[:, 21]).reshape(-1, 1)
        d0_dpy = (xi[:, 8] + xi[:, 18] - xi[:, 22]).reshape(-1, 1)
        d0_dpz = (xi[:, 9] + xi[:, 19] - xi[:, 23]).reshape(-1, 1)
        d0_dpe = (xi[:, 6] + xi[:, 16] - xi[:, 20]).reshape(-1, 1)

        return np.hstack((xi, ks_pip_p, ks_pim_p, ks_p, ks_m, phi_pip_p, phi_pim_p, phi_p, phi_m, d0_p, d0_m, 
        ks_dpx, ks_dpy, ks_dpz, ks_dpe, phi_dpx, phi_dpy, phi_dpz, phi_dpe, d0_dpx, d0_dpy, d0_dpz, d0_dpe))

    xi = augment(xi)
    xi0 = augment(xi0)
    xi_gen = augment(xi_gen)

    filenames = ['ks_pip_px', 'ks_pip_py', 'ks_pip_pz', 'ks_pim_px', 'ks_pim_py', 'ks_pim_pz', 'ks_E', 'ks_px', 'ks_py', 'ks_pz',
                'phi_pip_px', 'phi_pip_py', 'phi_pip_pz', 'phi_pim_px', 'phi_pim_py', 'phi_pim_pz', 'phi_E', 'phi_px', 'phi_py', 'phi_pz',
                'd0_E', 'd0_px', 'd0_py', 'd0_pz', 
                'ks_pip_p', 'ks_pim_p', 'ks_p', 'ks_m',
                'phi_pip_p', 'phi_pim_p', 'phi_p', 'phi_m',
                'd0_p', 'd0_m',
                'ks_dpx', 'ks_dpy', 'ks_dpz', 'ks_dpe',
                'phi_dpx', 'phi_dpy', 'phi_dpz', 'phi_dpe',
                'd0_dpx', 'd0_dpy', 'd0_dpz', 'd0_dpe',
                ]

    labels = {  'ks_pip_px':r'$p_x(\pi^+_{K_S^0})$ (MeV)',
                'ks_pip_py':r'$p_y(\pi^+_{K_S^0})$ (MeV)',
                'ks_pip_pz':r'$p_z(\pi^+_{K_S^0})$ (MeV)',
                'ks_pip_p' :r'$p(\pi^+_{K_S^0})$ (MeV)',
                'ks_pim_px':r'$p_x(\pi^-_{K_S^0})$ (MeV)',
                'ks_pim_py':r'$p_y(\pi^-_{K_S^0})$ (MeV)',
                'ks_pim_pz':r'$p_z(\pi^-_{K_S^0})$ (MeV)',
                'ks_pim_p' :r'$p(\pi^-_{K_S^0})$ (MeV)',
                'ks_E'     :r'$E(K_S^0)$ (MeV)',
                'ks_px'    :r'$p_x(K_S^0)$ (MeV)',
                'ks_py'    :r'$p_y(K_S^0)$ (MeV)',
                'ks_pz'    :r'$p_z(K_S^0)$ (MeV)',
                'ks_p'     :r'$p(K_S^0)$ (MeV)',
                'ks_m'     :r'$m(K_S^0)$ (MeV)',
                'ks_dpx'   :r'Conservation $p_x(K_S^0)$ (MeV)',
                'ks_dpy'   :r'Conservation $p_y(K_S^0)$ (MeV)',
                'ks_dpz'   :r'Conservation $p_z(K_S^0)$ (MeV)',
                'ks_dpe'   :r'Conservation $E(K_S^0)$ (MeV)',
                
                'phi_pip_px':r'$p_x(\pi^+_{\phi})$ (MeV)',
                'phi_pip_py':r'$p_y(\pi^+_{\phi})$ (MeV)',
                'phi_pip_pz':r'$p_z(\pi^+_{\phi})$ (MeV)',
                'phi_pip_p' :r'$p(\pi^+_{\phi})$ (MeV)',
                'phi_pim_px':r'$p_x(\pi^-_{\phi})$ (MeV)',
                'phi_pim_py':r'$p_y(\pi^-_{\phi})$ (MeV)',
                'phi_pim_pz':r'$p_z(\pi^-_{\phi})$ (MeV)',
                'phi_pim_p' :r'$p(\pi^-_{\phi})$ (MeV)',
                'phi_E'     :r'$E(\phi)$ (MeV)',
                'phi_px'    :r'$p_x(\phi)$ (MeV)',
                'phi_py'    :r'$p_y(\phi)$ (MeV)',
                'phi_pz'    :r'$p_z(\phi)$ (MeV)',
                'phi_p'     :r'$p(\phi)$ (MeV)',
                'phi_m'     :r'$m(\phi)$ (MeV)',
                'phi_dpx'   :r'Conservation $p_x(\phi)$ (MeV)',
                'phi_dpy'   :r'Conservation $p_y(\phi)$ (MeV)',
                'phi_dpz'   :r'Conservation $p_z(\phi)$ (MeV)',
                'phi_dpe'   :r'Conservation $E(\phi)$ (MeV)',
                
                'd0_E'     :r'$p_x(D^0)$ (MeV)',
                'd0_px'    :r'$p_x(D^0)$ (MeV)',
                'd0_py'    :r'$p_y(D^0)$ (MeV)',
                'd0_pz'    :r'$p_z(D^0$ (MeV)',
                'd0_p'     :r'$p(D^0)$ (MeV)',
                'd0_m'     :r'$m(D^0)$ (MeV)',
                'd0_dpx'   :r'Conservation $p_x(D^0)$ (MeV)',
                'd0_dpy'   :r'Conservation $p_y(D^0)$ (MeV)',
                'd0_dpz'   :r'Conservation $p_z(D^0)$ (MeV)',
                'd0_dpe'   :r'Conservation $E(D^0)$ (MeV)',} 

    for i in range(xi.shape[1]):
        if filenames[i] not in ['ks_dpe']:
            continue

        fitted = xi[:, i] - xi_gen[:, i]
        unfitted = xi0[:, i] - xi_gen[:, i]

        for _ in range(5):
            fit_mean, fit_std = np.mean(fitted), np.std(fitted)
            fitted = fitted[np.abs(fitted - fit_mean) < 5.*fit_std]
            unfit_mean, unfit_std = np.mean(unfitted), np.std(unfitted)
            unfitted = unfitted[np.abs(unfitted - unfit_mean) < 5.*unfit_std]

        fig, ax = plt.subplots(figsize=(4, 3))
        # plt.figure(figsize=(4,3))
        # ax.figure()

        if len(fitted) != 0:
            plt.errorbar(*make_hist(fitted, density=True),
                        linestyle='none', marker='.', markersize=4, label='after fit')
        if 1e-1 > (fit_std / unfit_std) or (fit_std / unfit_std) > 1e1:
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel(labels[filenames[i]], fontsize=16)
      
        # textstr = '\n'.join((
            # r"$\sigma_{{after}}$ = {:0.3f}".format(fit_std),
            # r"$\mu_{{after}}$ = {:0.3f}".format(fit_mean),
            # ))

        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        # verticalalignment='top', bbox=props)

        # ax.legend(loc='upper right', fontsize=12)
        # ax.grid()
        plt.xlabel(labels[filenames[i]], fontsize=12)
        plt.tight_layout()

        pathlib.Path('fig/d_meson/{}'.format(savedir)).mkdir(parents=True, exist_ok=True) 
        plt.savefig('fig/d_meson/{}/fit_{}.pgf'.format(savedir, filenames[i]))
        # plt.show()


def plot_evlolution_params_d0(xi_full, p3_ks_pip_gen, p3_ks_pim_gen, p3_phi_pip_gen, p3_phi_pim_gen, savedir):
    iter = -1
    for xi in xi_full:
        iter += 1

        ks3 = p3_ks_pip_gen + p3_ks_pim_gen
        phi3 = p3_phi_pip_gen + p3_phi_pim_gen
        d03 = ks3 + phi3

        eks_gen = energy(MASS_DICT['K0_S'], ks3).reshape(-1, 1)
        ephi_gen = energy(MASS_DICT['phi'], phi3).reshape(-1, 1)
        ed0_gen = energy(MASS_DICT['D0'], d03).reshape(-1, 1)
        
        xi_gen = np.hstack((p3_ks_pip_gen, p3_ks_pim_gen, eks_gen, ks3,
                            p3_phi_pip_gen, p3_phi_pim_gen, ephi_gen, phi3,
                            ed0_gen, d03))

        if (savedir.split('/')[0] == 'reffit'):
            xi = np.hstack((xi[:, 0:10], xi[:,15:25], xi[:,30:34]))
            # xi0 = np.hstack((xi0[:, 0:10], xi0[:,15:25], xi0[:,30:34]))

        def augment(xi):
            ks_pip_p = np.sqrt(np.sum(xi[:, 0:3]**2, axis=-1)).reshape(-1, 1)
            ks_pim_p = np.sqrt(np.sum(xi[:, 3:6]**2, axis=-1)).reshape(-1, 1)
            ks_p = np.sqrt(np.sum(xi[:, 7:10]**2, axis=-1)).reshape(-1, 1)
            ks_m = np.sqrt(xi[:, 6]**2 - np.sum(xi[:, 7:10]** 2, axis=-1)).reshape(-1, 1)
            
            phi_pip_p = np.sqrt(np.sum(xi[:, 10:13]**2, axis=-1)).reshape(-1, 1)
            phi_pim_p = np.sqrt(np.sum(xi[:, 13:16]**2, axis=-1)).reshape(-1, 1)
            phi_p = np.sqrt(np.sum(xi[:, 17:20]**2, axis=-1)).reshape(-1, 1)
            phi_m = np.sqrt(xi[:, 16]**2 - np.sum(xi[:, 17:20]** 2, axis=-1)).reshape(-1, 1)
            
            d0_p = np.sqrt(np.sum(xi[:, 21:24]**2, axis=-1)).reshape(-1, 1)
            d0_m = np.sqrt(xi[:, 20]**2 - np.sum(xi[:, 21:24]** 2, axis=-1)).reshape(-1, 1)
            
            ks_dpx = (xi[:, 0] + xi[:, 3] - xi[:, 7]).reshape(-1, 1)
            ks_dpy = (xi[:, 1] + xi[:, 4] - xi[:, 8]).reshape(-1, 1)
            ks_dpz = (xi[:, 2] + xi[:, 5] - xi[:, 9]).reshape(-1, 1)
            ks_dpe = (energy(MASS_DICT['pi+'], xi[:, 0:3]) + energy(MASS_DICT['pi+'], xi[:, 3:6]) - xi[:, 6]).reshape(-1, 1)

            phi_dpx = (xi[:, 10] + xi[:, 13] - xi[:, 17]).reshape(-1, 1)
            phi_dpy = (xi[:, 11] + xi[:, 14] - xi[:, 18]).reshape(-1, 1)
            phi_dpz = (xi[:, 12] + xi[:, 15] - xi[:, 19]).reshape(-1, 1)
            phi_dpe = (energy(MASS_DICT['pi+'], xi[:, 10:13]) + energy(MASS_DICT['pi+'], xi[:, 13:16]) - xi[:, 16]).reshape(-1, 1)

            d0_dpx = (xi[:, 7] + xi[:, 17] - xi[:, 21]).reshape(-1, 1)
            d0_dpy = (xi[:, 8] + xi[:, 18] - xi[:, 22]).reshape(-1, 1)
            d0_dpz = (xi[:, 9] + xi[:, 19] - xi[:, 23]).reshape(-1, 1)
            d0_dpe = (xi[:, 6] + xi[:, 16] - xi[:, 20]).reshape(-1, 1)

            return np.hstack((xi, ks_pip_p, ks_pim_p, ks_p, ks_m, phi_pip_p, phi_pim_p, phi_p, phi_m, d0_p, d0_m, 
            ks_dpx, ks_dpy, ks_dpz, ks_dpe, phi_dpx, phi_dpy, phi_dpz, phi_dpe, d0_dpx, d0_dpy, d0_dpz, d0_dpe))

        xi = augment(xi)
        # xi0 = augment(xi0)
        xi_gen = augment(xi_gen)

        filenames = ['ks_pip_px', 'ks_pip_py', 'ks_pip_pz', 'ks_pim_px', 'ks_pim_py', 'ks_pim_pz', 'ks_E', 'ks_px', 'ks_py', 'ks_pz',
                    'phi_pip_px', 'phi_pip_py', 'phi_pip_pz', 'phi_pim_px', 'phi_pim_py', 'phi_pim_pz', 'phi_E', 'phi_px', 'phi_py', 'phi_pz',
                    'd0_E', 'd0_px', 'd0_py', 'd0_pz', 
                    'ks_pip_p', 'ks_pim_p', 'ks_p', 'ks_m',
                    'phi_pip_p', 'phi_pim_p', 'phi_p', 'phi_m',
                    'd0_p', 'd0_m',
                    'ks_dpx', 'ks_dpy', 'ks_dpz', 'ks_dpe',
                    'phi_dpx', 'phi_dpy', 'phi_dpz', 'phi_dpe',
                    'd0_dpx', 'd0_dpy', 'd0_dpz', 'd0_dpe',
                    ]

        labels = {  'ks_pip_px':r'$p_x(\pi^+_{K_S^0})$ (MeV)',
                    'ks_pip_py':r'$p_y(\pi^+_{K_S^0})$ (MeV)',
                    'ks_pip_pz':r'$p_z(\pi^+_{K_S^0})$ (MeV)',
                    'ks_pip_p' :r'$p(\pi^+_{K_S^0})$ (MeV)',
                    'ks_pim_px':r'$p_x(\pi^-_{K_S^0})$ (MeV)',
                    'ks_pim_py':r'$p_y(\pi^-_{K_S^0})$ (MeV)',
                    'ks_pim_pz':r'$p_z(\pi^-_{K_S^0})$ (MeV)',
                    'ks_pim_p' :r'$p(\pi^-_{K_S^0})$ (MeV)',
                    'ks_E'     :r'$E(K_S^0)$ (MeV)',
                    'ks_px'    :r'$p_x(K_S^0)$ (MeV)',
                    'ks_py'    :r'$p_y(K_S^0)$ (MeV)',
                    'ks_pz'    :r'$p_z(K_S^0)$ (MeV)',
                    'ks_p'     :r'$p(K_S^0)$ (MeV)',
                    'ks_m'     :r'$m(K_S^0)$ (MeV)',
                    'ks_dpx'   :r'Conservation $p_x(K_S^0)$ (MeV)',
                    'ks_dpy'   :r'Conservation $p_y(K_S^0)$ (MeV)',
                    'ks_dpz'   :r'Conservation $p_z(K_S^0)$ (MeV)',
                    'ks_dpe'   :r'Conservation $E(K_S^0)$ (MeV)',
                    
                    'phi_pip_px':r'$p_x(\pi^+_{\phi})$ (MeV)',  
                    'phi_pip_py':r'$p_y(\pi^+_{\phi})$ (MeV)',
                    'phi_pip_pz':r'$p_z(\pi^+_{\phi})$ (MeV)',
                    'phi_pip_p' :r'$p(\pi^+_{\phi})$ (MeV)',
                    'phi_pim_px':r'$p_x(\pi^-_{\phi})$ (MeV)',
                    'phi_pim_py':r'$p_y(\pi^-_{\phi})$ (MeV)',
                    'phi_pim_pz':r'$p_z(\pi^-_{\phi})$ (MeV)',
                    'phi_pim_p' :r'$p(\pi^-_{\phi})$ (MeV)',
                    'phi_E'     :r'$E(\phi)$ (MeV)',
                    'phi_px'    :r'$p_x(\phi)$ (MeV)',
                    'phi_py'    :r'$p_y(\phi)$ (MeV)',
                    'phi_pz'    :r'$p_z(\phi)$ (MeV)',
                    'phi_p'     :r'$p(\phi)$ (MeV)',
                    'phi_m'     :r'$m(\phi)$ (MeV)',
                    'phi_dpx'   :r'Conservation $p_x(\phi)$ (MeV)',
                    'phi_dpy'   :r'Conservation $p_y(\phi)$ (MeV)',
                    'phi_dpz'   :r'Conservation $p_z(\phi)$ (MeV)',
                    'phi_dpe'   :r'Conservation $E(\phi)$ (MeV)',
                    
                    'd0_E'     :r'$p_x(D^0)$ (MeV)',
                    'd0_px'    :r'$p_x(D^0)$ (MeV)',
                    'd0_py'    :r'$p_y(D^0)$ (MeV)',
                    'd0_pz'    :r'$p_z(D^0$ (MeV)',
                    'd0_p'     :r'$p(D^0)$ (MeV)',
                    'd0_m'     :r'$m(D^0)$ (MeV)',
                    'd0_dpx'   :r'Conservation $p_x(D^0)$ (MeV)',
                    'd0_dpy'   :r'Conservation $p_y(D^0)$ (MeV)',
                    'd0_dpz'   :r'Conservation $p_z(D^0)$ (MeV)',
                    'd0_dpe'   :r'Conservation $E(D^0)$ (MeV)',} 


        for i in [33,]:#range(xi.shape[1]):
            
            fitted = xi[:, i] - xi_gen[:, i]
            # unfitted = xi0[:, i] - xi_gen[:, i]

            for _ in range(5):
                fit_mean, fit_std = np.mean(fitted), np.std(fitted)
                fitted = fitted[np.abs(fitted - fit_mean) < 5.*fit_std]
                # unfit_mean, unfit_std = np.mean(unfitted), np.std(unfitted)
                # unfitted = unfitted[np.abs(unfitted - unfit_mean) < 5.*unfit_std]

            # plt.figure(figsize=(8, 6))

            # if 1e-1 > (fit_std / unfit_std) or (fit_std / unfit_std) > 1e1:
                # plt.subplot(1, 2, 1)
            if len(fitted) != 0:
                if fit_std > 1e-3:
                    plt.errorbar(*make_hist(fitted, range=[-0.1, 0.1], density=True),
                            linestyle='none', marker='.', markersize=4, label='fit')
            # if 1e-1 > (fit_std / unfit_std) or (fit_std / unfit_std) > 1e1:
                plt.plot([], [], ' ', label=r"$\sigma_{{fit}}$ {:0.3f}".format(fit_std))
                # plt.plot([], [], ' ', label=r"$\sigma_{{unfit}}$ {:0.3f}".format(unfit_std))
                plt.plot([], [], ' ', label=r"$\mu_{{fit}}$ {:0.3f}".format(fit_mean))
                # plt.plot([], [], ' ', label=r"$\mu_{{unfit}}$ {:0.3f}".format(unfit_mean))
                plt.legend(loc='upper right')
                plt.grid()
                plt.xlabel(labels[filenames[i]], fontsize=16)
                # plt.subplot(1, 2, 2)    
            # if len(unfitted) != 0:
                # plt.errorbar(*make_hist(unfitted, density=True),
                            # linestyle='none', marker='.', markersize=4, label='unfit')
            # plt.plot([], [], ' ', label=r"$\sigma_{{fit}}$ {:0.3f}".format(fit_std))
            # plt.plot([], [], ' ', label=r"$\sigma_{{unfit}}$ {:0.3f}".format(unfit_std))
            # plt.plot([], [], ' ', label=r"$\mu_{{fit}}$ {:0.3f}".format(fit_mean))
            # plt.plot([], [], ' ', label=r"$\mu_{{unfit}}$ {:0.3f}".format(unfit_mean))
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel("Evolution: {} {}, iter: {}".format(labels[filenames[i]], savedir, iter), fontsize=16)
    plt.tight_layout()

            # pathlib.Path('fig/d_meson/{}'.format(savedir)).mkdir(parents=True, exist_ok=True) 
            # plt.savefig('fig/d_meson/{}/evolution_{}.png'.format(savedir, filenames[i]))
    plt.show()


def plot_params(xi, xi0, pimgen, pipgen, savedir):
    ks3 = pipgen + pimgen
    xi_gen = np.hstack((pipgen, pimgen, 
        (energy(MASS_DICT['pi+'], pipgen) + energy(MASS_DICT['pi+'], pimgen)).reshape(-1, 1), 
        ks3))

    def augment(xi):
        ks_m = np.sqrt(xi[:, 6]**2 - np.sum(xi[:, 7:10]
                                            ** 2, axis=-1)).reshape(-1, 1)
        ks_p = np.sqrt(np.sum(xi[:, 7:10]**2, axis=-1)).reshape(-1, 1)
        pip_p = np.sqrt(np.sum(xi[:, :3]**2, axis=-1)).reshape(-1, 1)
        pim_p = np.sqrt(np.sum(xi[:, 3:6]**2, axis=-1)).reshape(-1, 1)
        dpx = (xi[:, 0] + xi[:, 3] - xi[:, 7]).reshape(-1, 1)
        dpy = (xi[:, 1] + xi[:, 4] - xi[:, 8]).reshape(-1, 1)
        dpz = (xi[:, 2] + xi[:, 5] - xi[:, 9]).reshape(-1, 1)
        dpe = (energy(MASS_DICT['pi+'], xi[:, 0:3]) + energy(MASS_DICT['pi+'], xi[:, 3:6]) -
        xi[:, 6]).reshape(-1, 1)

        return np.hstack((xi[:, :10], ks_m, ks_p, pip_p, pim_p, dpx, dpy, dpz, dpe))

    xi = augment(xi)
    xi0 = augment(xi0)
    xi_gen = augment(xi_gen)

    filenames = ['pip_px', 'pip_py', 'pip_pz', 'pim_px', 'pim_py', 'pim_pz',
             'ks_E', 'ks_px', 'ks_py', 'ks_pz', 'ks_m', 'ks_p', 'pip_p', 
             'pim_p', 'dpx', 'dpy', 'dpz', 'dpe']

    labels = {   'pip_px':r'$p_x(\pi+)$ (MeV)',
                 'pip_py':r'$p_y(\pi+)$ (MeV)',
                 'pip_pz':r'$p_z(\pi+)$ (MeV)',
                 'pim_px':r'$p_x(\pi-)$ (MeV)', 
                 'pim_py':r'$p_y(\pi-)$ (MeV)', 
                 'pim_pz':r'$p_z(\pi-)$ (MeV)',
                 'ks_E'  :r'$E(K_S^0)$ (MeV)', 
                 'ks_px' :r'$p_x(K_S^0)$ (MeV)', 
                 'ks_py' :r'$p_y(K_S^0)$ (MeV)', 
                 'ks_pz' :r'$p_z(K_S^0)$ (MeV)', 
                 'ks_m'  :r'$m(K_S^0)$ (MeV)', 
                 'ks_p'  :r'$p(K_S^0)$ (MeV)', 
                 'pip_p' :r'$p(\pi+)$ (MeV)', 
                 'pim_p' :r'$p(\pi-)$ (MeV)', 
                 'dpx'   :r'Conservation $p_x$ (MeV)', 
                 'dpy'   :r'Conservation $p_y$ (MeV)', 
                 'dpz'   :r'Conservation $p_z$ (MeV)',
                 'dpe'   :r'Conservation $E$ (MeV)',} 

    for i in range(xi.shape[1]):
        fitted = xi[:, i] - xi_gen[:, i]
        unfitted = xi0[:, i] - xi_gen[:, i]

        for _ in range(5):
            fit_mean, fit_std = np.mean(fitted), np.std(fitted)
            fitted = fitted[np.abs(fitted - fit_mean) < 5.*fit_std]
            unfit_mean, unfit_std = np.mean(unfitted), np.std(unfitted)
            unfitted = unfitted[np.abs(unfitted - unfit_mean) < 5.*unfit_std]

        plt.figure(figsize=(4, 3))
        if len(fitted) != 0:
            plt.errorbar(*make_hist(fitted, density=True),
                         linestyle='none', marker='.', markersize=4, label='fit')
        if len(unfitted) != 0:
            plt.errorbar(*make_hist(unfitted, density=True),
                         linestyle='none', marker='.', markersize=4, label='unfit')
        plt.plot([], [], ' ', label=r"$\sigma_{{fit}}$ {:0.3f}".format(fit_std))
        plt.plot([], [], ' ', label=r"$\sigma_{{unfit}}$ {:0.3f}".format(unfit_std))
        plt.plot([], [], ' ', label=r"$\mu_{{fit}}$ {:0.3f}".format(fit_mean))
        plt.plot([], [], ' ', label=r"$\mu_{{unfit}}$ {:0.3f}".format(unfit_mean))
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel(labels[filenames[i]], fontsize=14)
        plt.tight_layout()
        pathlib.Path('fig/kaon/{}'.format(savedir)).mkdir(parents=True, exist_ok=True) 
        plt.savefig('fig/kaon/{}/fit_{}.pgf'.format(savedir, filenames[i]))
    # plt.show()

def plot_pull(xi, cov, pimgen, pipgen, savedir):
    ks3 = pipgen + pimgen
    xi0 = np.hstack((pipgen, pimgen, energy(
        MASS_DICT['K0_S'], ks3).reshape(-1, 1), ks3)) 

    filenames = ['pip_px', 'pip_py', 'pip_pz', 'pim_px', 'pim_py', 'pim_pz',
             'ks_E', 'ks_px', 'ks_py', 'ks_pz']

    labels = {'pip_px':r'$p_x(\pi+)$ (MeV)',
                 'pip_py':r'$p_y(\pi+)$ (MeV)',
                 'pip_pz':r'$p_z(\pi+)$ (MeV)',
                 'pim_px':r'$p_x(\pi-)$ (MeV)', 
                 'pim_py':r'$p_y(\pi-)$ (MeV)', 
                 'pim_pz':r'$p_z(\pi-)$ (MeV)',
                 'ks_E'  :r'$E(K_S^0)$ (MeV)', 
                 'ks_px' :r'$p_x(K_S^0)$ (MeV)', 
                 'ks_py' :r'$p_y(K_S^0)$ (MeV)', 
                 'ks_pz' :r'$p_z(K_S^0)$ (MeV)' }

    for i in range(xi0.shape[1]):

        fig, ax = plt.subplots(figsize=(4, 3))
            # ax.figure()

        m = (xi[:, i] - xi0[:, i])[cov[:, i, i] > 0]
        cov2 = cov[cov[:, i, i] > 0]
        m = m / cov2[:, i, i] ** 0.5
       # m = (xi[:, i] - xi0[:, i]) / cov[:, i, i] ** 0.5

        for _ in range(5):
            mean, std = np.mean(m), np.std(m)
            m = m[np.abs(m - mean) < 5.*std]
        print(m.shape, np.mean(m), np.std(m))
        if np.std(m) < 0.000001 or len(m) == 0:
            return
        x, bins, e = make_hist(m)

        ax.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)

        norm = (x[-1] - x[0]) / len(x) * sum(bins)
        mu = 0
        sigma = 1
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax.plot(x, norm * stats.norm.pdf(x, mu, sigma))
       
        # textstr = '\n'.join((
        #     r"$\sigma_{{after}}$ = {:0.3f}".format(fit_std),
        #     r"$\sigma_{{before}}$ = {:0.3f}".format(unfit_std),
        #     r"$\mu_{{after}}$ = {:0.3f}".format(fit_mean),
        #     r"$\mu_{{before}}$ = {:0.3f}".format(unfit_mean)
        #     ))

        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        # verticalalignment='top', bbox=props)

        ax.plot([], [], ' ', label=r"$mean$ {:0.3f}".format(mean))
        ax.plot([], [], ' ', label=r"$\sigma~$ {:0.3f}".format(std))
       
        ax.legend()
        ax.grid()
        ax.xlabel(labels[filenames[i]], fontsize=12)
        ax.tight_layout()
        ax.show()
        # plt.savefig('fig/kaon/{}/pull_{}.pgf'.format(savedir, filenames[i]))
    # plt.show()

def plot_pull_d0(xi, cov, p3_ks_pip_gen, p3_ks_pim_gen, p3_phi_pip_gen, p3_phi_pim_gen, savedir):
    ks3 = p3_ks_pip_gen + p3_ks_pim_gen
    phi3 = p3_phi_pip_gen + p3_phi_pim_gen
    d03 = ks3 + phi3

    eks_gen = energy(MASS_DICT['K0_S'], ks3).reshape(-1, 1)
    ephi_gen = energy(MASS_DICT['phi'], phi3).reshape(-1, 1)
    ed0_gen = energy(MASS_DICT['D0'], d03).reshape(-1, 1)

    xi_gen = np.hstack((p3_ks_pip_gen, p3_ks_pim_gen, eks_gen, ks3,
                        p3_phi_pip_gen, p3_phi_pim_gen, ephi_gen, phi3,
                        ed0_gen, d03))

    if (savedir.split('/')[0] == 'reffit'):
        xi = np.hstack((xi[:, 0:10], xi[:,15:25], xi[:,30:34]))
        cov[:, 10:20, 10:20] = cov[:, 15:25, 15:25]
        cov[:, 20:24, 20:24] = cov[:, 30:34, 30:34]
   
    filenames = ['ks_pip_px', 'ks_pip_py', 'ks_pip_pz', 'ks_pim_px', 'ks_pim_py', 'ks_pim_pz', 'ks_E', 'ks_px', 'ks_py', 'ks_pz',
                 'phi_pip_px', 'phi_pip_py', 'phi_pip_pz', 'phi_pim_px', 'phi_pim_py', 'phi_pim_pz', 'phi_E', 'phi_px', 'phi_py', 'phi_pz',
                 'd0_E', 'd0_px', 'd0_py', 'd0_pz', 
                 ]

    labels = {  'ks_pip_px':r'$p_x(\pi^+_{K_S^0})$ (MeV)',
                'ks_pip_py':r'$p_y(\pi^+_{K_S^0})$ (MeV)',
                'ks_pip_pz':r'$p_z(\pi^+_{K_S^0})$ (MeV)',
                'ks_pim_px':r'$p_x(\pi^-_{K_S^0})$ (MeV)',
                'ks_pim_py':r'$p_y(\pi^-_{K_S^0})$ (MeV)',
                'ks_pim_pz':r'$p_z(\pi^-_{K_S^0})$ (MeV)',
                'ks_E'     :r'$E(K_S^0)$ (MeV)',
                'ks_px'    :r'$p_x(K_S^0)$ (MeV)',
                'ks_py'    :r'$p_y(K_S^0)$ (MeV)',
                'ks_pz'    :r'$p_z(K_S^0)$ (MeV)',
                'ks_p'     :r'$p(K_S^0)$ (MeV)',
                'ks_m'     :r'$m(K_S^0)$ (MeV)',
                'ks_dpx'   :r'Conservation $p_x(K_S^0)$ (MeV)',
                'ks_dpy'   :r'Conservation $p_y(K_S^0)$ (MeV)',
                'ks_dpz'   :r'Conservation $p_z(K_S^0)$ (MeV)',
                'ks_dpe'   :r'Conservation $E(K_S^0)$ (MeV)',
                
                'phi_pip_px':r'$p_x(\pi^+_{\phi})$ (MeV)',
                'phi_pip_py':r'$p_y(\pi^+_{\phi})$ (MeV)',
                'phi_pip_pz':r'$p_z(\pi^+_{\phi})$ (MeV)',
                'phi_pip_p' :r'$p(\pi^+_{\phi})$ (MeV)',
                'phi_pim_px':r'$p_x(\pi^-_{\phi})$ (MeV)',
                'phi_pim_py':r'$p_y(\pi^-_{\phi})$ (MeV)',
                'phi_pim_pz':r'$p_z(\pi^-_{\phi})$ (MeV)',
                'phi_pim_p' :r'$p(\pi^-_{\phi})$ (MeV)',
                'phi_E'     :r'$E(\phi)$ (MeV)',
                'phi_px'    :r'$p_x(\phi)$ (MeV)',
                'phi_py'    :r'$p_y(\phi)$ (MeV)',
                'phi_pz'    :r'$p_z(\phi)$ (MeV)',
                'phi_p'     :r'$p(\phi)$ (MeV)',
                'phi_m'     :r'$m(\phi)$ (MeV)',
                'phi_dpx'   :r'Conservation $p_x(\phi)$ (MeV)',
                'phi_dpy'   :r'Conservation $p_y(\phi)$ (MeV)',
                'phi_dpz'   :r'Conservation $p_z(\phi)$ (MeV)',
                'phi_dpe'   :r'Conservation $E(\phi)$ (MeV)',
                
                'd0_E'     :r'$E(D^0)$ (MeV)',
                'd0_px'    :r'$p_x(D^0)$ (MeV)',
                'd0_py'    :r'$p_y(D^0)$ (MeV)',
                'd0_pz'    :r'$p_z(D^0)$ (MeV)',
                'd0_p'     :r'$p(D^0)$ (MeV)',
                'd0_m'     :r'$m(D^0)$ (MeV)',
                'd0_dpx'   :r'Conservation $p_x(D^0)$ (MeV)',
                'd0_dpy'   :r'Conservation $p_y(D^0)$ (MeV)',
                'd0_dpz'   :r'Conservation $p_z(D^0)$ (MeV)',
                'd0_dpe'   :r'Conservation $E(D^0)$ (MeV)',} 

    for i in range(xi.shape[1]):
        if filenames[i] not in ['ks_E']:
            continue
        
        m = (xi[:, i] - xi_gen[:, i])[cov[:, i, i] > 0]
        cov2 = cov[cov[:, i, i] > 0]
        m = m / cov2[:, i, i] ** 0.5

        for _ in range(5):
            mean, std = np.mean(m), np.std(m)
            m = m[np.abs(m - mean) < 5.*std]
        if np.std(m) < 0.000001:
            return
        x, bins, e = make_hist(m)

        fig, ax = plt.subplots(figsize=(4, 3))
        
        ax.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)

        norm = (x[-1] - x[0]) / len(x) * sum(bins)
        mu = 0
        sigma = 1
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, norm * stats.norm.pdf(x, mu, sigma))

        textstr = '\n'.join((
            r"$\mu$ = {:0.3f}".format(mean),
            r"$\sigma~$ = {:0.3f}".format(std)
            ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

        ax.grid()
        plt.xlabel('pull ' + labels[filenames[i]], fontsize=12)
        plt.tight_layout()
        plt.savefig('fig/d_meson/{}/pull_{}.pgf'.format(savedir, filenames[i]))
    # plt.show()

def plot_conservation(xi):
    """ """
    pi4p = p3top4(xi[:, 0:3], MASS_DICT['pi+'])
    pi4m = p3top4(xi[:, 3:6], MASS_DICT['pi+'])

    m = pi4m + pi4p - xi[:, 6:10]

    plt.figure(figsize=(6, 5))
    for i in range(4):
        x, bins, e = make_hist(m[:, i])
        plt.errorbar(x, bins, e, linestyle='none',
                     marker='.', markersize=4, label=i)
        plt.legend()

    plt.grid()
    plt.xlabel(r'$Conservation$ (MeV)', fontsize=16)
    plt.tight_layout()
    # plt.show()


def plot_pipi_mass(xi):
    """ """
    pi4p = p3top4(xi[:, 0:3], MASS_DICT['pi+'])
    pi4m = p3top4(xi[:, 3:6], MASS_DICT['pi+'])
    m = mass_sq(pi4p + pi4m)**0.5

    if np.std(m) < 0.000001:
        return
    x, bins, e = make_hist(m)  # , range=[497, 498])

    plt.figure(figsize=(6, 5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    plt.grid()
    plt.xlabel(r'$m(\pi^+\pi^-)$ (MeV)', fontsize=16)
    plt.tight_layout()
    # plt.show()


def plot_ks0_mass(xi):
    """ """
    m = mass(xi[:, 6:10])
    m = m[~np.isnan(m)]
    for _ in range(5):
        mean, std = np.mean(m), np.std(m)
        m = m[np.abs(m - mean) < 5.*std]
    print(m.shape, np.mean(m), np.std(m))
    if np.std(m) < 0.000001:
        return
    x, bins, e = make_hist(m)  # , range=[497, 498])

    plt.figure(figsize=(6, 5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    plt.grid()
    plt.xlabel(r'$m(K_S^0)$ (MeV)', fontsize=16)
    plt.tight_layout()
    # plt.show()

def plot_d0_mass(xi):
    """ """
    m = mass(xi[:, 20:24])
    m = m[~np.isnan(m)]
    for _ in range(5):
        mean, std = np.mean(m), np.std(m)
        m = m[np.abs(m - mean) < 5.*std]
    print(m.shape, np.mean(m), np.std(m))
    if np.std(m) < 0.000001:
        return
    x, bins, e = make_hist(m)  # , range=[497, 498])

    plt.figure(figsize=(6, 5))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    plt.grid()
    plt.xlabel(r'$m(K_S^0)$ (MeV)', fontsize=16)
    plt.tight_layout()
    # plt.show()

def plot_chi2(chisq, savedir):
    chisq = chisq[~np.isnan(chisq) & (chisq < 100)]
    rng = [0, 10]
    nbins = 50
    # for _ in range(5):
    #     mean, std = np.mean(chisq), np.std(chisq)
    #     chisq = chisq[np.abs(chisq - mean) < 3.*std]
    print(chisq.shape, chisq.mean(), chisq.std())
    x, bins, e = make_hist(chisq, range=rng, nbins=nbins, density=False)

    plt.figure(figsize=(3, 2))
    plt.errorbar(x, bins, e, linestyle='none', marker='.', markersize=4)
    norm = chisq.shape[0]*(rng[1]-rng[0])/nbins if rng is not None else 1
    plt.plot(x, norm*chi2.pdf(x, 1), label=r'$\chi^2(1)$')
    # plt.plot(x, norm*chi2.pdf(x, 2), label=r'$\chi^2(2)$')
    # plt.plot(x, norm*chi2.pdf(x, 3), label=r'$\chi^2(3)$')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    # plt.xlabel(r'$\chi^2$', fontsize=16)
    plt.tight_layout()

    pathlib.Path('fig/{}'.format(savedir)).mkdir(parents=True, exist_ok=True) 
    plt.savefig('fig/{}/chi.pdf'.format(savedir))
    # plt.show()


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
    # decays = ['kaon'] #, 'd_meson']
    decays = ['d_meson']
    methods = ['reffit', 'kalman']
    # methods = ['kalman']
    # energies = [0, 250, 1000]
    energies = [250]
    
    for decay in decays:
        for method in methods:
            for energy in energies:
                data = np.load('logs/{}/{}/fitres_{:.1f}MeV.npz'.
                format(decay, method, energy))
        
                # plot_chi2(data['chi2'][-1], '{}/{}/{}'.format(decay, method, energy))

                if decay == 'd_meson':
                    # plot_evlolution_params_d0(data['xi'],  data['p3_ks_pip_gen'], data['p3_ks_pim_gen'],
                        # data['p3_phi_pip_gen'], data['p3_phi_pim_gen'], '{}/{}'.format(method, energy))

                    plot_params_d0(data['xi'][-1], data['xi'][0], data['p3_ks_pip_gen'], data['p3_ks_pim_gen'],
                        data['p3_phi_pip_gen'], data['p3_phi_pim_gen'], '{}/{}'.format(method, energy))
                    plot_pull_d0(data['xi'][-1], data['Ck'][-1], data['p3_ks_pip_gen'], data['p3_ks_pim_gen'],
                        data['p3_phi_pip_gen'], data['p3_phi_pim_gen'], '{}/{}'.format(method, energy))
                else:
                    pass
                    # plot_params(data['xi'][-1], data['xi'][0], data['pimgen'], 
                        # data['pipgen'], '{}/{}'.format(method, energy))
                    # plot_pull(data['xi'][-1], data['Ck'][-1], data['pimgen'], 
                        # data['pipgen'], '{}/{}'.format(method, energy))


        plt.show()
             
if __name__ == '__main__':
    main()

    # for idx in range(4):
    #     xi = data['xi'][idx]
    #     xi = xi[~np.isnan(xi).any(axis=1)]
    #     print(xi.shape)
    #     plot_ks0_mass(xi)
    # plt.show()
