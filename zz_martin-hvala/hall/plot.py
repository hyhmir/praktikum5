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


measdict = read_measdict('measurements/*.csv')

B = 0.173
c = 0.95e-3
k = 1.3806503e-23
e0 = 1.602167e-19

direction, T12, U12, I_ = measdict['U-by-T']
I12 = 1e-3 * I_

T12 += 273.15

T1 = T12[direction == 0]
U1_true = U12[direction == 0]
I1_true = I12[direction == 0]

T2 = T12[direction == 1]
U2 = U12[direction == 1]
I2 = I12[direction == 1]
ΔU2 = 0.001
ΔI2 = 0.05e-3

T = T2
ΔT = 0.5

U1 = np.interp(T2, T1, U1_true)
I1 = np.interp(T2, T1, I1_true)
ΔU1 = 0.001 + 0*U1
ΔU1[(T > 65.0) * (T < 79.0)] *= 3
ΔI1 = 0.05e-3 + 0*I1
ΔI1[(T > 65.0) * (T < 79.0)] *= 4

U = (U1-U2)/2
ΔU = (ΔU1+ΔU2)/2

I = (I1+I2)/2
ΔI = (ΔI1+ΔI2)/2

R_H = U*c/(I*B)
ΔR_H = np.sqrt(
    (ΔU*c/(I*B))**2
    + (U*c*ΔI/(I**2*B))**2
)
by_kT = 1/(k*T)

def fig_U_by_T(savefig=None):

    fig = mpl.figure.Figure([4.5, 5])
    gs = mpl.gridspec.GridSpec(2, 9, wspace=1, hspace=0.5, height_ratios=[3,5])
    ax1, ax2, ax3 = axs = [
        fig.add_subplot(gs[0,:4]),
        fig.add_subplot(gs[0,5:]),
        fig.add_subplot(gs[1,1:-1])
    ]

    R = U/I
    ΔR = np.sqrt( (ΔU/I)**2 + (U*ΔI/I)**2 )

    ax1.errorbar(
        T, 1e3 * U, yerr=1e3 * ΔU,
        fmt='.', color='tab:red',
        ecolor='dimgray', capsize=3,
    )
    ax2.errorbar(
        T, 1e3 * I, yerr=1e3 * ΔI,
        fmt='.', color='tab:red',
        ecolor='dimgray', capsize=3,
    )

    ax1.set(
        title=r'Hallova napetost',
        xlabel=r'temperatura $T\,[\mathrm{K}]$',
        ylabel=r'$U_H\,[\mathrm{mV}]$',
        ylim=(0,25),
    )
    ax2.set(
        title=r'Tok',
        xlabel=r'temperatura $T\,[\mathrm{K}]$',
        ylabel=r'$I\,[\mathrm{mA}]$',
        ylim=(0,10),
    )

    ax3.errorbar(
        T, R_H, yerr=ΔR_H,
        fmt='.', color=mpl.colormaps['gnuplot'](0.7),
        ecolor='dimgray', capsize=3,
    )

    ax3.set(
        title=r'Hallova konstanta $R_H = cU_H/IB$',
        xlabel=r'temperatura $T\,[\mathrm{K}]$',
        ylabel=r'$R_H\,[\mathrm{m^3/As}]$',
    )

    for ax in axs:
        ax.set(xlim=(305,358))

    fig.subplots_adjust(
        left=0.12, right=0.94,
        bottom=0.1,
    )
    fig.savefig(savefig, dpi=300)

def fig_both_U(savefig=None):

    fig = mpl.figure.Figure([4, 3])
    ax = fig.subplots(1, 1)

    ax.errorbar(
        T, U1/I1, yerr=np.sqrt( (ΔU1/I1)**2 + (U1*ΔI1/I1**2) ),
        fmt='.', color='purple',
        ecolor='dimgray', capsize=3,
        label=r'$U_1/I_1$'
    )
    ax.errorbar(
        T, U2/I2, yerr=np.sqrt( (ΔU2/I2)**2 + (U2*ΔI2/I2**2) ),
        fmt='.', color='tab:blue',
        ecolor='dimgray', capsize=3,
        label=r'$U_2/I_2$'
    )

    ax.set(
        title=r'Posamezna upora v obeh orientacijah',
        xlabel=r'temperatura $T\,[\mathrm{K}]$',
        ylabel=r'$R\,[\mathrm{\Omega}]$',
        xlim=(305,358)
    )
    ax.legend(frameon=False)

    fig.subplots_adjust(left=0.2, right=0.92, bottom=0.18)
    fig.savefig(savefig, dpi=300)

def linear(x, k, c):
    return k*x + c

def fig_lnn_by_1_by_kT(savefig=None):

    fig = mpl.figure.Figure([4, 3])
    ax = fig.subplots(1, 1)

    by_kT = 1/(k*T)
    n = e0/R_H
    lnn = np.log(n)
    Δlnn = e0*ΔR_H/R_H**2 / n

    split=10
    ax.errorbar(
        e0 * by_kT[:split], lnn[:split], yerr=Δlnn[:split],
        fmt='d', color=mpl.colormaps['viridis'](0.9),
        ecolor='dimgray', capsize=3,
    )
    ax.errorbar(
        e0 *  by_kT[split:], lnn[split:], yerr=Δlnn[split:],
        fmt='d', color=mpl.colormaps['viridis'](0.5),
        ecolor='dimgray', capsize=3,
    )

    par, cov = curve_fit(
        linear, by_kT[split:], lnn[split:],
        sigma=Δlnn[split:], absolute_sigma=True
    )
    m, c = par
    Δm, Δc = np.sqrt(np.diag(cov))

    by_kT_lin = np.linspace(32/e0, 39/e0)
    ax.plot(
        e0 * by_kT_lin, linear(by_kT_lin, *par),
        ls='--', color='tab:red',
        label=f'$m = ({1/e0 * m:.2f} \\pm {1/e0 * Δm:.2f})\\,\\mathrm{{eV}}$'
    )
    print(f'E_g = ({1/e0 * -2*m:.2f} \\pm {1/e0 * 2*Δm:.3f})\\,\\mathrm{{eV}}')
    var_unexplained = np.sum((lnn - linear(by_kT, *par))**2) / np.sum(lnn**2)
    print(f'{100*var_unexplained:.5f}% of Var unexplained')

    ax.set(
        title=r'Hallova konstanta $R_H = cU_H/IB$',
        xlabel=r'$1/kT\,[\mathrm{(eV)^{-1}}]$',
        ylabel=r'$\mathrm{ln}(n_p)\,[\mathrm{ln(m^{-3})}]$',
        xlim=(32,38),
        ylim=(-37.75,-39.6),
    )
    ax.legend(frameon=False)

    fig.subplots_adjust(left=0.2, right=0.92, bottom=0.18)
    fig.savefig(savefig, dpi=300)


fig_U_by_T(savefig='U-by-T.pdf')
fig_both_U(savefig='both.pdf')
fig_lnn_by_1_by_kT(savefig='lnn-by-1-by-kT.pdf')
