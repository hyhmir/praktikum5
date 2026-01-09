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


measdict = read_measdict('zz_martin-hvala/diff/measurements/*.csv')

a, Δa = 48.8e-2, 1e-2
b, Δb = 100.1e-2, 1e-2
d, Δd = 15e-3, 1e-3

h_max = 165.5e-3
t_, h_diff_ = measdict['peak-by-t']
t = 60 * t_
h_diff = 1e-3 * h_diff_

S0 = 10e-2*22e-2
S0_u = 3614.66

k = (a+b)/a
Δk = np.sqrt(
    ((1/a - (a+b)/a**2) * Δa)**2 +
    (Δb/a)**2
)

Δn = 0.029
S_predicted = k*b*d*Δn
ΔS_predicted = np.sqrt(
    (Δk*b*d*Δn)**2 +
    (k*Δb*d*Δn)**2 +
    (k*b*Δd*Δn)**2
)
# print(f'S_predicted = ({1e6 * S_predicted:.0f} \\pm {1e6 * ΔS_predicted:.0f})\\,\\mathrm{{mm^2}}')

S_tri_u, S1_u, S2_u, S3_u = measdict['areas']

S1, = (S1_u - S_tri_u) * S0/S0_u
S2, = (S2_u - S_tri_u) * S0/S0_u
S3, = (S2_u - S_tri_u) * S0/S0_u

S = np.mean([S1, S2, S3])
ΔS = np.sqrt(np.var([S1, S2, S3])/3)

# print(f'S1 = {1e6 * S1:.0f}\\,\\mathrm{{mm^2}}')
# print(f'S2 = {1e6 * S2:.0f}\\,\\mathrm{{mm^2}}')
# print(f'S3 = {1e6 * S3:.0f}\\,\\mathrm{{mm^2}}')
# print(f'S = ({1e6 * S:.0f} \\pm {1e6 * ΔS:.0f})\\,\\mathrm{{mm^2}}')

h = h_max - np.cumsum(h_diff)
Δh = 6*0.5e-3

sortindices = np.argsort(t)
t = t[sortindices]
h = h[sortindices]

s = 1/(4*np.pi*k**2) * (S/h)**2
Δs = np.sqrt(
    (2*Δk/(4*np.pi*k**3) * (S/h)**2)**2 +
    (1/(4*np.pi*k**2) * 2*(S/h) * ΔS/h)**2 +
    (1/(4*np.pi*k**2) * 2*(S/h) * S*Δh/h**2)**2
)


def linear(x, k, c):
    return k*x + c

def fig_by_t(savefig=None):

    fig = mpl.figure.Figure([4.5, 3])
    ax = fig.subplots(1, 1)

    ax.errorbar(
        1/60 * t, 1e3 * h, yerr=1e3 * Δh,
        fmt='.', color='purple',
        ecolor='dimgray', capsize=3,
    )

    ax.set(
        title=r'Višina vrha nad diagonalo',
        xlabel=r'čas $t\,[\mathrm{min}]$',
        ylabel=r'$h\,[\mathrm{mm}]$',
        xlim=(-5,158),
        ylim=(70,170)
    )

    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.18)
    fig.savefig(savefig, dpi=300)

def fig_linearized(savefig=None):

    fig = mpl.figure.Figure([5, 4])
    ax = fig.subplots(1, 1)

    ax.errorbar(
        1/60 * t, 1e6 * s, yerr=1e6 * Δs,
        fmt='^', color=mpl.colormaps['viridis'](0.6),
        ecolor='dimgray', capsize=3,
    )

    par, cov = curve_fit(linear, t, s, sigma=Δs, absolute_sigma=True)
    D, _ = par
    ΔD, _ = np.sqrt(np.diag(cov))

    t_interval = [np.min(t)-5*60, np.max(t)+5*60]
    t_lin = np.linspace(*t_interval)
    ax.plot(
        1/60 * t_lin, 1e6 * linear(t_lin, *par),
        ls='--', color='tab:red',
        label=(
            f'$D = ({1e10 * D:.2f} \\pm {1e10 * ΔD:.2f})\\,\cdot 10^{{-10}}' +
            '\\,\\mathrm{{m^2/s}}$'
        )
    )

    ax.set(
        title=r'Linearizirana oblika',
        xlabel=r'čas $t\,[\mathrm{min}]$',
        ylabel=r'$\frac{1}{4\pi k^2} \left( \frac{S}{h} \right)^2 \,[\mathrm{mm^2}]$',
        xlim=1/60 * np.array(t_interval),
        #ylim=(70,170)
    )
    ax.legend(frameon=False)

    fig.subplots_adjust(left=0.2, right=0.92, bottom=0.18)
    fig.savefig(savefig, dpi=300)

fig_by_t(savefig='10_DIF/porocilo/by-t.pdf')
fig_linearized(savefig='10_DIF/porocilo/linearized.pdf')
