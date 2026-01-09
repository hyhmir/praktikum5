import matplotlib as mpl
from matplotlib import figure, animation
import scienceplots
mpl.pyplot.style.use('science')
from matplotlib.ticker import AutoMinorLocator

import numpy as np
import numba
import scipy as sci
from scipy.optimize import curve_fit

import re
from measio import read_measdict

def linear(x, k, c):
    return k*x + c

def fig_T2_star_ions(savefig=None):

    fig = mpl.figure.Figure([4, 3])
    ax = fig.subplots(1, 1)

    measdict = read_measdict('zz_martin-hvala/nmr/measurements/*.csv')
    print(measdict)

    t_, U_ = measdict['single-halfpi-fallof-ions']
    t = 50e-6 * t_
    U = 2 * U_
    ΔU = 0.2

    ax.errorbar(
        1e6 * t, np.log(U), yerr=ΔU/U,
        fmt='.', color=mpl.colormaps['magma'](0.55),
        ecolor='dimgray', capsize=3,
        label=r'meritve'
    )

    par, cov = curve_fit(
        linear, t, np.log(U),
        sigma=ΔU/U, absolute_sigma=True
    )
    m, c = par
    Δm, Δc = np.sqrt(np.diag(cov))

    τ = -1/m
    Δτ = Δm/m**2

    t_interval = [np.min(t) - 20e-6, np.max(t) + 20e-6]
    t_lin = np.linspace(*t_interval)
    ax.plot(
        1e6 * t_lin, linear(t_lin, *par),
        ls='--', color='tab:red',
        label=f'$T_2^* = ({1e3 * τ:.2f} \\pm {1e3 * Δτ:.2f})\\,\\mathrm{{ms}}$'
    )

    ax.set(
        title=r'Pojemanje po enem $\pi/2$ sunku (ioni)',
        xlabel=r'čas $t\,[\mathrm{\mu s}]$',
        ylabel=r'$\mathop{\rm ln}(M) + {\rm konst.}$',
        xlim=1e6*np.array(t_interval)
    )
    ax.legend(frameon=False, loc='lower left')

    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.18)
    fig.savefig(savefig, dpi=300)

def exp_saturate(x, M0, τ):
    return M0*(1 - np.exp(-x/τ))

def fig_T1_ions(savefig=None):

    fig = mpl.figure.Figure([4, 3])
    ax = fig.subplots(1, 1)

    measdict = read_measdict('zz_martin-hvala/nmr/measurements/*.csv')

    t_, U_ = measdict['double-halfpi-ions']
    t = 1e-3 * t_
    U = U_
    ΔU = 0.2

    ax.errorbar(
        1e3 * t, U, yerr=ΔU,
        fmt='.', color=mpl.colormaps['magma'](0.55),
        ecolor='dimgray', capsize=3,
        label=r'meritve'
    )

    p0 = [3, 1e-3]
    par, cov = curve_fit(
        exp_saturate, t, U, p0=p0,
        sigma=ΔU+0*U, absolute_sigma=True
    )
    M0, τ = par
    ΔM0, Δτ = np.sqrt(np.diag(cov))

    t_interval = [np.min(t) - 1e-3, np.max(t) + 1e-3]
    t_lin = np.linspace(*t_interval)
    ax.plot(
        1e3 * t_lin, exp_saturate(t_lin, *par),
        ls='--', color='tab:red',
        label=f'$T_1 = ({1e3 * τ:.0f} \\pm {1e3 * Δτ:.0f})\\,\\mathrm{{ms}}$'
    )

    ax.set(
        title=r'Signal po drugem $\pi/2$ sunku (ioni)',
        xlabel=r'čas med sunkoma $\tau\,[\mathrm{ms}]$',
        ylabel=r'$\propto M\,[\mathrm{V}]$',
    )
    ax.legend(frameon=False)

    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.18)
    fig.savefig(savefig, dpi=300)

def fig_T1_noions(savefig=None):

    fig = mpl.figure.Figure([4, 3])
    ax = fig.subplots(1, 1)

    measdict = read_measdict('zz_martin-hvala/nmr/measurements/*.csv')
    print(measdict)

    t_, U_ = measdict['double-halfpi-noions']
    t = 1e-3 * t_
    U = U_
    ΔU = 0.2

    ax.errorbar(
        1e3 * t, U, yerr=ΔU,
        fmt='.', color=mpl.colormaps['viridis'](0.4),
        ecolor='dimgray', capsize=3,
        label=r'meritve'
    )

    p0 = [3, 1e-3]
    par, cov = curve_fit(
        exp_saturate, t, U, p0=p0,
        sigma=ΔU+0*U, absolute_sigma=True
    )
    M0, τ = par
    ΔM0, Δτ = np.sqrt(np.diag(cov))

    t_interval = [np.min(t) - 1e-4, np.max(t) + 1e-4]
    t_lin = np.linspace(*t_interval)
    ax.plot(
        1e3 * t_lin, exp_saturate(t_lin, *par),
        ls='--', color=mpl.colormaps['plasma'](0.7),
        label=f'$T_1 = ({1e3 * τ:.2f} \\pm {1e3 * Δτ:.2f})\\,\\mathrm{{ms}}$'
    )

    ax.set(
        title=r'Signal po drugem $\pi/2$ sunku (voda)',
        xlabel=r'čas med sunkoma $\tau\,[\mathrm{ms}]$',
        ylabel=r'$\propto M\,[\mathrm{V}]$',
    )
    ax.legend(frameon=False)

    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.18)
    fig.savefig(savefig, dpi=300)

def fig_T2_ions(savefig=None):

    fig = mpl.figure.Figure([4, 3])
    ax = fig.subplots(1, 1)

    measdict = read_measdict('zz_martin-hvala/nmr/measurements/*.csv')

    t_, U_ = measdict['halfpi-pi-ions']
    t = 1e-3 * t_
    U = U_
    ΔU = 0.2

    ax.errorbar(
        1e3 * t, np.log(U), yerr=ΔU/U,
        fmt='.', color=mpl.colormaps['magma'](0.55),
        ecolor='dimgray', capsize=3,
        label=r'meritve'
    )

    par, cov = curve_fit(
        linear, t, np.log(U),
        sigma=ΔU+0*U, absolute_sigma=True
    )
    m, c = par
    Δm, Δc = np.sqrt(np.diag(cov))

    τ = -1/m
    Δτ = Δm/m**2

    t_interval = [np.min(t) - 1e-4, np.max(t) + 1e-4]
    t_lin = np.linspace(*t_interval)
    ax.plot(
        1e3 * t_lin, linear(t_lin, *par),
        ls='--', color='tab:red',
        label=f'$T_2 = ({1e3 * τ:.1f} \\pm {1e3 * Δτ:.1f})\\,\\mathrm{{ms}}$'
    )

    ax.set(
        title=r'Amplituda spinskega odmeva (ioni)',
        xlabel=r'čas med $\pi/2$ in $\pi$ sunkom $\tau\,[\mathrm{ms}]$',
        ylabel=r'$\mathop{\rm ln}(M) + {\rm konst.}$',
    )
    ax.legend(frameon=False)

    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.18)
    fig.savefig(savefig, dpi=300)

T2_star, ΔT2_star = 0.12e-3, 0.02e-3
γ = 2.675e8
δB = 1/(γ*T2_star)
Δ_δB = ΔT2_star/(γ*T2_star**2)

# print(δB)
# print(Δ_δB)

fig_T2_star_ions(savefig='06_NMR/porocilo/T2-star-ions.pdf')
fig_T1_ions(savefig='06_NMR/porocilo/T1-ions.pdf')
fig_T1_noions(savefig='06_NMR/porocilo/T1-noions.pdf')
fig_T2_ions(savefig='06_NMR/porocilo/T2-ions.pdf')
