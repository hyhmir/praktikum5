import matplotlib as mpl
from matplotlib import figure, animation
mpl.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'axes.labelsize': 'large',
    'figure.dpi': 100,
})
from matplotlib.ticker import AutoMinorLocator

import numpy as np
import numba
import scipy as sci
from scipy.optimize import curve_fit

import re
from measio import read_measdict

def L2_gauss_diff(x, μ, σ):
    C2 = 1
    return (
        -np.sqrt(C2)
        * np.sqrt(2/(σ**3*np.sqrt(np.pi)))
        * (x-μ)
        * np.exp(-1/2 * (x-μ)**2 / σ**2)
    )

def integral_L2_gauss_diff(x, μ, σ):
    C2 = 1
    return (
        np.sqrt(C2)
        * np.sqrt(2*σ/np.sqrt(np.pi))
        * np.exp(-1/2 * (x-μ)**2 / σ**2)
    )

def L2_sech_diff(x, μ, a):
    C2 = 1
    x = x-μ
    return -np.sqrt(3/(2*a)) * np.sinh(x/a) / np.cosh(x/a)**2

def integral_L2_sech_diff(x, μ, a):
    C2 = 1
    x = x-μ
    return np.sqrt(3*a/2) * 1 / np.cosh(x/a)

ν_B = []

def fig_absorb(savefig=None):

    fig = mpl.figure.Figure([5, 7])
    ax1, ax2 = fig.subplots(2, 1)
    axs = [ax1, ax2]

    μ0 = 1.256637062e-6
    N = 1557
    a, Δa = 13.5e-2, 0.3e-2
    b, Δb = 12.3e-2, 0.1e-2
    d = np.sqrt(a**2 + b**2)
    Δd = np.sqrt( (Δa*a/d)**2 + (Δb*b/d)**2 )
    
    measdict = read_measdict('measurements/*.csv')

    arr = [
        ['79MHz', '85MHz', '90MHz'],
        [79.2e6, 85.03e6, 90.25e6]
    ]
    Δν = 0.2e6

    for i, (measname, ν) in enumerate(zip(*arr)):

        I_, U_ = measdict[measname]
        I = 1e-3 * I_
        U = 1e-3 * U_

        U -= (U[0] + U[-1])/2.0
        I_mid = (I[1:] + I[:-1])/2.0
        U_mid = (U[1:] + U[:-1])/2.0

        L2_U = np.sqrt( np.sum(U_mid**2 * np.diff(I)) )
        U_L2_normed = U/L2_U

        I_lin = np.linspace(100e-3, 500e-3, num=501)

        B = N*μ0*I/d
        B_mid = N*μ0*I_mid/d
        B_lin = N*μ0*I_lin/d

        if i == 0: p0 = [270e-3, 7e-3]
        elif i == 1: p0 = [290e-3, 7e-3]
        elif i == 2: p0 = [310e-3, 7e-3]
        par, cov = curve_fit(L2_gauss_diff, I, U_L2_normed, p0=p0)
        μ, σ = par
        Δμ, Δσ = np.sqrt(np.diag(cov))

        plt, = ax1.plot(
            1e3 * B, 1e3 * U,
            marker='.', markersize=8, ls='', zorder=10,
            color=mpl.colormaps['viridis'](i/4),
            label=f'${1e-6 * ν:.1f}\\,\\mathrm{{MHz}}$'
        )

        if i == 0: p0 = [270e-3, 5e-3]
        elif i == 1: p0 = [290e-3, 7e-3]
        elif i == 2: p0 = [310e-3, 7e-3]
        par, cov = curve_fit(L2_sech_diff, I, U_L2_normed, p0=p0)
        mean, a = par
        Δmean, Δa = np.sqrt(np.diag(cov))

        U_fit = L2_U * L2_gauss_diff(I_lin, μ, σ)
        U_altfit = L2_U * L2_sech_diff(I_lin, mean, a)

        # In I-space
        peakdist_minmax = I[np.argmin(U)] - I[np.argmax(U)]
        peakdist_fit = 2*σ
        Δpeakdist_fit = 2*Δσ
        peakdist_altfit = np.log(3 + 2*np.sqrt(2)) * a
        Δpeakdist_altfit = np.log(3 + 2*np.sqrt(2)) * Δa

        var_unexplained_fit = np.sum((U - L2_U*L2_gauss_diff(I, μ, σ))**2) / np.sum(U**2)
        var_unexplained_altfit = np.sum((U - L2_U*L2_sech_diff(I, mean, a))**2) / np.sum(U**2)
        # print(
        #     f'${1e-6 * ν:.1f} \\pm {1e-6 * Δν:.1f}$ & ' +
        #     f'{1e3 * peakdist_minmax:.1f} & ' +
        #     f'${1e3 * peakdist_fit:.1f} \\pm {1e3 * Δpeakdist_fit:.1f}$ & ' +
        #     f'{100 * var_unexplained_fit:.1f}\\% & ' +
        #     f'${1e3 * peakdist_altfit:.1f} \\pm {1e3 * Δpeakdist_altfit:.1f}$ & ' +
        #     f'{100 * var_unexplained_altfit:.1f}\\% ' +
        #     '\\\\'
        # )
        # In B-space
        B_peakdist = peakdist_altfit * N*μ0/d
        ΔB_peakdist = np.sqrt(
            (Δpeakdist_altfit * N*μ0/d)**2
            + (peakdist_altfit * N*μ0*Δd/d**2)**2
        )
        B_μ = mean * N*μ0/d
        ΔB_μ = np.sqrt(
            (Δmean * N*μ0/d)**2
            + (mean * N*μ0*Δd/d**2)**2
        )
        # print(
        #     f'${1e-6 * ν:.1f} \\pm {1e-6 * Δν:.1f}$ & ' +
        #     f'${1e3 * B_μ:.1f} \\pm {1e3 * ΔB_μ:.1f}$ & ' +
        #     f'${1e3 * B_peakdist:.3f} \\pm {1e3 * ΔB_peakdist:.3f}$ ' +
        #     '\\\\'
        # )

        list.append(ν_B, (ν, Δν, B_μ, ΔB_μ))

        p1, = ax1.plot(
            1e3 * B_lin, 1e3 * U_fit,
            marker='', ls='-', lw=1,
            color=mpl.colormaps['magma'](i/5+.25),
        )
        p2, = ax1.plot(
            1e3 * B_lin, 1e3 * U_altfit,
            marker='', ls='--', lw=1,
            color=mpl.colormaps['magma'](i/5+.25),
        )

        integral_U = np.cumsum(U_mid*np.diff(I))
        integral_U_fit = L2_U * integral_L2_gauss_diff(I_lin, μ, σ)
        integral_U_altfit = L2_U * integral_L2_sech_diff(I_lin, mean, a)

        # ax2.plot(
        #     B_lin, 1e3 * integral_U_fit,
        #     marker='', ls='-', lw=1,
        #     color=mpl.colormaps['magma'](i/5+.25),
        # )

        f = lambda x, c: L2_U*integral_L2_sech_diff(x, mean, a) + c
        par, cov = curve_fit(f, I_mid, integral_U, p0=[2e-3])
        c, = par
        Δc, = np.sqrt(np.diag(cov))

        ax2.plot(
            1e3 * B_mid, 1e3 * (integral_U - c),
            marker='^', markersize=5, ls='', zorder=10,
            color=mpl.colormaps['viridis'](i/4),
            label=f'${1e-6 * ν:.1f}\\,\\mathrm{{MHz}}$'
        )
        ax2.plot(
            1e3 * B_lin, 1e3 * integral_U_altfit,
            marker='', ls='--', lw=1,
            color=mpl.colormaps['magma'](i/5+.25),
            label='fit $\mathrm{sech}(x) \pm C$' if i == 2 else None
        )

    ax1.set(
        title='Napetost, sorazmerna odvodu absorpcije',
        xlabel=r'gostota $B_0\,[\mathrm{mT}]$',
        ylabel=r'$\Delta U\,[\mathrm{mV}]$',
        xlim=(2,3.8),
        # ylim=(1e-5,0.4),
        #xscale='log',
    )
    ax2.set(
        title='Absorpcija',
        xlabel=r'gostota $B_0\,[\mathrm{mT}]$',
        ylabel=r'enote absorpcije',
        xlim=(2,3.8),
        ylim=(-2, 13),
        yticks=[],
        #xscale='log',
    )

    ax2.legend(frameon=False)
    l1 = ax1.legend(frameon=False)
    l2 = ax1.legend(
        [p1,p2],
        ['odvod Gaussovke', 'odvod $\mathrm{sech}(x)$'],
        frameon=False,
        loc='lower left'
    )
    ax1.add_artist(l1)
    
    fig.subplots_adjust(
        left=0.15,
        hspace=0.4
    )
    fig.savefig(savefig, dpi=300)

def linear(x, k, c):
    return k*x + c

def fig_by_freq(savefig=None):

    fig = mpl.figure.Figure([4, 3])
    ax = fig.subplots(1, 1)

    ν, Δν, B, ΔB = np.array(ν_B).T
    μ_B = 9.27e-24
    h = 6.6261e-34

    par, cov = curve_fit(
        linear, ν, B,
        sigma=ΔB, absolute_sigma=True
    )
    k, c = par
    Δk, Δc = np.sqrt(np.diag(cov))

    by_k, Δby_k = 1/k, Δk/k**2
    print(f'({1e-9 * by_k:.0f} \\pm {1e-9 * Δby_k:.0f})\\,\\mathrm{{GHz/T}}')

    g, Δg = by_k * h/μ_B, Δby_k * h/μ_B
    print(f'({g:.1f} \\pm {Δg:.1f})\\,\\mathrm{{GHz/T}}')

    ax.errorbar(
        1e-6 * ν, 1e3 * B, yerr=1e3 * ΔB,
        fmt='s', color='tab:red',
        ecolor='dimgray', capsize=3,
    )
    ν_lin = np.array([70e6, 100e6])
    ax.plot(
        1e-6 * ν_lin, 1e3 * linear(ν_lin, *par),
        ls='--', color='darkslategrey',
        label=f'$k = ({1e9 * k:.2f} \\pm {1e9 * Δk:.2f})\\,\\mathrm{{T/GHz}}$'
    )

    ax.set(
        title='Napetost, sorazmerna odvodu absorpcije',
        xlabel=r'frekvenca $\nu\,[\mathrm{MHz}]$',
        ylabel=r'gostota $B\,[\mathrm{mT}]$',
        xlim=(77,93),
        ylim=(2.7,3.5),
        #xscale='log',
    )
    ax.legend(frameon=False)

    fig.subplots_adjust(
        left=0.16, right=0.92,
        bottom=0.18
    )
    fig.savefig(savefig, dpi=300)

fig_absorb(savefig='spectra.pdf')
fig_by_freq(savefig='by-freq.pdf')
