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

def linear(x, k, c):
    return k*x + c

def debye_re(ω, δP0, τ):
    return δP0 / (1 + (ω*τ)**2)

def debye_im(ω, δP0, τ):
    return δP0 * (-1)*ω*τ / (1 + (ω*τ)**2)

def fig_modulation(savefig=None):

    fig = mpl.figure.Figure([4, 3])
    ax = fig.subplots(1, 1)

    ν_ref = 30
    φ, Δφ = np.deg2rad(144), np.deg2rad(1)

    measdict = read_measdict('measurements/*.csv')

    U0, R, ΔR = measdict['deltaP-by-modulation']

    ax.errorbar(
        U0, R, yerr=ΔR,
        fmt='s', color='tab:red',
        ecolor='dimgray', capsize=3,
    )

    par, cov = curve_fit(
        linear, U0, R,
        sigma=ΔR, absolute_sigma=True
    )
    k, c = par
    Δk, Δc = np.sqrt(np.diag(cov))

    U0_lin = np.array([-0.5, 1])
    ax.plot(
        U0_lin, linear(U0_lin, *par),
        ls='--', color='darkslategrey',
        label=f'$k = ({k:.1f} \\pm {Δk:.1f})$'
    )

    ax.set(
        title=r'Napetost, sorazmerna $|\delta P|$ pri ' f'$\\nu = {ν_ref}\\,\\mathrm{{Hz}}$',
        xlabel=r'modulacijski $U_0\,[\mathrm{V}]$',
        ylabel=r'$(U \propto |\delta P|)\,[\mathrm{V}]$',
        xlim=(0,0.7),
        ylim=(-10,200),
        # xscale='log',
    )

    ax.legend(frameon=False)
    fig.subplots_adjust(
        left=0.16, right=0.92,
        bottom=0.16,
    )
    fig.savefig(savefig, dpi=300)

def fig_XY_by_freq(savefig=None):

    fig = mpl.figure.Figure([6.2, 3.8])
    ax1, ax2 = fig.subplots(1, 2)
    axs = [ax1, ax2]
    fig.suptitle(r'Kompleksna napetost, sorazmerna $\delta P$')

    U0 = 30
    φ, Δφ = np.deg2rad(144), np.deg2rad(1)

    measdict = read_measdict('measurements/*.csv')

    ν, X_, ΔX_, Y_, ΔY_ = measdict['XY-by-freq']
    ω = 2*np.pi * ν
    X, ΔX = 1e-3 * X_, 1e-3 * 50*ΔX_ # Error a posteriori adjusted according to fit
    Y, ΔY = 1e-3 * Y_, 1e-3 * 60*ΔY_ # Error a posteriori adjusted to fit

    ax1.errorbar(
        ν, 1e3 * X, yerr=1e3 * ΔX,
        fmt='^', color=mpl.colormaps['magma'](0.3),
        ecolor='dimgray', capsize=3,
    )
    ax2.errorbar(
        ν, 1e3 * Y, yerr=1e3 * ΔY,
        fmt='^', color=mpl.colormaps['magma'](0.4),
        ecolor='dimgray', capsize=3,
    )

    ν_interval = [np.min(ν)-200, np.max(ν)+100]
    ν_lin = np.linspace(*ν_interval, num=501)
    ω_lin = 2*np.pi * ν_lin

    # Real part fit

    p0 = [0.1, 5e-3]
    par, cov = curve_fit(
        debye_re, ω, X, p0=p0,
        sigma=ΔX, absolute_sigma=True
    )
    δP0X, τX = par
    Δ_δP0X, ΔτX = np.sqrt(np.diag(cov))

    ax1.plot(
        ν_lin, 1e3 * debye_re(ω_lin, *par),
        ls='-', lw=1.2, color=mpl.colormaps['magma'](0.65),
    )

    label=(
        f'$\\tau = ({1e3 * τX:.1f} \\pm {1e3 * ΔτX:.1f})\\,\\mathrm{{ms}}$' +
        '\n' +
        f'$\\delta P_0 = ({1e3 * δP0X:.0f} \\pm {1e3 * Δ_δP0X:.0f})\\,\\mathrm{{mV}}$'
    )
    ax1.text(
        0.95, 0.95, label,
        transform=ax1.transAxes,
        verticalalignment='top', horizontalalignment='right'
    )

    # Imag part fit

    par, cov = curve_fit(
        debye_im, ω, Y, p0=par,
        sigma=ΔY, absolute_sigma=True
    )
    δP0Y, τY = par
    Δ_δP0Y, ΔτY = np.sqrt(np.diag(cov))
    ax2.plot(
        ν_lin, 1e3 * debye_im(ω_lin, *par),
        ls='-', lw=1.2, color=mpl.colormaps['magma'](0.75),
    )

    label=(
        f'$\\tau = ({1e3 * τY:.1f} \\pm {1e3 * ΔτY:.1f})\\,\\mathrm{{ms}}$' +
        '\n' +
        f'$\\delta P_0 = ({1e3 * δP0Y:.0f} \\pm {1e3 * Δ_δP0Y:.0f})\\,\\mathrm{{mV}}$'
    )
    ax2.text(
        0.95, 0.95, label,
        transform=ax2.transAxes,
        verticalalignment='top', horizontalalignment='right'
    )

    for i, ax in enumerate(axs):
        ax.set(
            title=r'Realna komponenta' if i == 0 else 'Imaginarna komponenta',
            xlabel=r'frekvenca $\nu\,[\mathrm{Hz}]$',
            ylabel=(
                r'$(X \propto \Re(\delta P))\,[\mathrm{mV}]$'
                if i == 0 else
                r'$(Y \propto \Im(\delta P))\,[\mathrm{mV}]$'
            ),
            xlim=ν_interval,
            ylim=(10,-150) if i == 0 else (0, 80),
            # xscale='log',
        )

    fig.subplots_adjust(
        # left=0.16, right=0.92,
        top=0.84,
        wspace=0.3
    )
    fig.savefig(savefig, dpi=300)

def fig_linearized(savefig=None):

    fig = mpl.figure.Figure([4, 3])
    ax = fig.subplots(1, 1)

    ν_ref = 30
    φ, Δφ = np.deg2rad(144), np.deg2rad(1)

    measdict = read_measdict('measurements/*.csv')

    ν, X_, ΔX_, Y_, ΔY_ = measdict['XY-by-freq']
    ω = 2*np.pi * ν
    X, ΔX = 1e-3 * X_, 1e-3 * 50*ΔX_
    Y, ΔY = 1e-3 * Y_, 1e-3 * 60*ΔY_

    r = -Y/X
    Δr = np.sqrt( (ΔY/X)**2 + (Y*ΔX/X**2)**2 )

    num_ignore=6
    ax.errorbar(
        ν[:-num_ignore], r[:-num_ignore], yerr=Δr[:-num_ignore],
        #ν, r, yerr=Δr,
        fmt='o', color=mpl.colormaps['gnuplot'](0.6),
        ecolor='dimgray', capsize=3,
    )

    par, cov = curve_fit(
        linear, ω, r,
        sigma=Δr, absolute_sigma=True
    )
    k, c = par
    Δk, Δc = np.sqrt(np.diag(cov))

    ν_interval = [np.min(ν)-200, np.max(ν)+100]
    ν_lin = np.linspace(*ν_interval, num=501)
    ω_lin = 2*np.pi * ν_lin

    ax.plot(
        ν_lin, linear(ω_lin, *par),
        ls='--', color='darkslategrey',
        label=(
            r'naklon razmerja po $\omega$,' + '\n' +
            f'$\\tau = ({1e3 * k:.1f} \\pm {1e3 * Δk:.1f})\\,\\mathrm{{ms}}$'
        )
    )

    ax.set(
        title=r'Razmerje komponent predstavlja $(-\tan\alpha)$',
        xlabel=r'frekvenca $\nu\,[\mathrm{Hz}]$',
        ylabel=r'razmerje $(-Y/X)$',
        xlim=(0,420),
        ylim=(0,8),
        # xscale='log',
    )

    ax.legend(frameon=False)
    fig.subplots_adjust(
        left=0.16, right=0.92,
        bottom=0.16,
    )
    fig.savefig(savefig, dpi=300)

fig_modulation(savefig='modulation.pdf')
fig_XY_by_freq(savefig='XY-by-freq.pdf')
fig_linearized(savefig='linearized.pdf')
