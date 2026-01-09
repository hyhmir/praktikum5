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


@numba.njit
def exponent(t, R):
    return R*np.exp(-R*t)

def fig_single(savefig=None):

    fig = mpl.figure.Figure([5, 4])
    ax = fig.subplots(1, 1)
    
    measdict = read_measdict('measurements/*.txt', sep='\t')

    t_, N, PDF_t_ = measdict['1-channel-radioactive']
    t = 1e-12 * t_
    PDF_t = 1e12 * PDF_t_

    int_PDF_t = np.sum(PDF_t * t[1])
    print(f'integral PDF(t) dt = {int_PDF_t:.6f}')
    print(f'{100 * np.sum(PDF_t > 0) / len(PDF_t)}% of the bins have zero counts')

    t_elapsed = 60.1
    Δt_elapsed_guess = 0.1
    counts = 18771
    R_mean = counts / t_elapsed
    ΔR_mean = counts * Δt_elapsed_guess / t_elapsed**2
    print(f'R_mean = ({R_mean:.1f} +/- {ΔR_mean:.1f}) 1/s, if elapsed time is measured to +/- {Δt_elapsed_guess:.1f} s and no counts are missed')

    plt, = ax.plot(
        1e3 * t, 1e-3 * PDF_t,
        lw=0.8,
        label=(
            'meritve, '
            f'{counts} sunkov v ${t_elapsed}\\,\\mathrm{{s}}$'
        )
    )
    ax.fill_between(
        1e3 * t, 1e-3 * PDF_t,
        color=plt.get_color(), alpha=0.2,
    )

    par, cov = curve_fit(exponent, t, PDF_t, p0=[R_mean])
    R_fit, = par
    σ_R, = np.sqrt(np.diag(cov))
    print(f'R_fit = ({R_fit:.1f} +/- {σ_R:.1f}) 1/s')

    t_lin = np.linspace(np.min(t), np.max(t), num=501)
    plt, = ax.plot(
        1e3 * t, 1e-3 * exponent(t, R_fit),
        ls=':', color='r',
        label=(
            r'fit porazdelitve z '
            f'$R = ({R_fit:.1f} \\pm {σ_R:.1f})\\,\\mathrm{{s}}^{{-1}}$'
        )
    )

    ax.set(
        title='Spekter po časih med razpadi',
        xlabel=r'$t\,[\mathrm{ms}]$',
        ylabel=r'gostota $\frac{\mathrm{d}p}{\mathrm{d}t}\,[\mathrm{ms}^{-1}]$',
        xlim=(0,30),
        ylim=(1e-5,0.4),
        #xscale='log',
        yscale='log',
    )

    fig.subplots_adjust(left=0.15, right=0.92, bottom=0.15)
    ax.legend(frameon=False, loc='upper right')
    
    fig.savefig(savefig, dpi=300)

@numba.njit
def gauss(x, σ):
    μ = 0
    C = 1
    return (
        C/(σ*np.sqrt(2*np.pi))
        * np.exp(-1/2 * (x-μ)**2 / σ**2)
    )

def fig_correlation(savefig=None):

    fig = mpl.figure.Figure([6, 4.5])
    axs = fig.subplots(2, 2).flat
    fig.suptitle('Spekter po časih med razpadi')
    
    measdict = read_measdict('measurements/reserve/*.txt', sep='\t')

    meas_by_degs = [
        (float(m.group(1)), measname)
        for measname in measdict.keys()
        if (m := re.match(r'peak-(\d+)deg.*', measname))
    ]

    for i, (deg, measname) in enumerate(sorted(meas_by_degs)):

        t_, N, PDF_t_ = measdict[measname]
        t = 1e-9 * t_
        PDF_t = 1e9 * PDF_t_
        angle = np.deg2rad(deg)

        ax = axs[i//2]
        cmap = mpl.colormaps['magma' if i % 2 == 0 else 'viridis']

        # int_PDF_t = np.sum(PDF_t * np.mean(np.diff(t)))
        # print(f'integral PDF(t) dt = {int_PDF_t:.6f}')
        # print(f'{100 * np.sum(PDF_t > 0) / len(PDF_t)}% of the bins have zero counts')
        
        n = 3*(len(t)//3)
        t = t[0:n:3]
        PDF_t = PDF_t[0:n:3] + PDF_t[1:n:3] + PDF_t[2:n:3]

        par, cov = curve_fit(gauss, t, PDF_t, p0=[50e-9])
        σ, = par
        Δσ, = np.sqrt(np.diag(cov))

        t_lin = np.linspace(np.min(t), np.max(t), num=401)
        ax.plot(
            1e9 * t_lin, 1e-9 * gauss(t_lin, σ),
            color='0.1', lw=0.8, ls='--',
        )

        ax.bar(
            1e9 * t, 1e-9 * PDF_t, width=1.05e9 * np.mean(np.diff(t)),
            color=cmap((deg/180)**0.5+0.2),
            alpha=0.8,
            label=f'${int(180 - deg)}^\\circ$'
        )

        ax.set(
            xlabel=(
                r'$t\,[\mathrm{ns}]$'
                if i > 4 and (i % 2 == 1) else None
            ),
            ylabel=(
                'gostota'
                r'$\frac{\mathrm{d}p}{\mathrm{d}t}\,[\mathrm{ns}^{-1}]$'
                if i % 4 == 1 else None
            ),
            xlim=(-350,350),
            ylim=(0,0.045),
        )

        fig.subplots_adjust(right=0.95, bottom=0.14)
        ax.legend(frameon=False)
    
    fig.savefig(savefig, dpi=300)

def fig_correlation_by_angle(savefig=None):

    fig = mpl.figure.Figure([3.5, 2.8])
    ax = fig.subplots(1, 1)
    
    measdict = read_measdict('measurements/reserve/*.txt', sep='\t')

    def process_meas(angle, meas):

        t_, N, PDF_t_ = meas
        t = 1e-9 * t_
        PDF_t = 1e9 * PDF_t_
        par, cov = curve_fit(gauss, t, PDF_t, p0=[50e-9])
        σ, = par

        count = np.sum(N)
        A = np.sum(N * np.mean(np.diff(t)))
        Δcount = np.sqrt(np.sum( (A*gauss(t, σ) - N)**2 ))
        return [angle, count, Δcount]

    angles, counts, Δcounts = np.array(sorted([
        process_meas(np.deg2rad(float(m.group(1))), measdict[measname])
        for measname in measdict.keys()
        if (m := re.match(r'peak-(\d+)deg.*', measname))
    ])).T

    ax.errorbar(
        np.rad2deg(np.pi - angles), counts, yerr=Δcounts,
        color='tab:red', fmt='^--',
        ecolor='dimgray', capsize=3,
    )

    ax.set(
        title='Število sunkov v vrhu',
        xlabel='kot med scintilatorjema $\phi$',
        ylabel='število sunkov $N$',
        xlim=(85,190),
        xticks=np.rad2deg(np.pi - angles),
        xticklabels=[90, 120, 135, 155, '', 170, '', 180][::-1],
        ylim=(0,900),
    )

    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.15)
    # ax.legend(frameon=False)

    fig.savefig(savefig, dpi=300)

def fig_random_coincidence(savefig=None):

    fig = mpl.figure.Figure([6, 5.5])
    gs = mpl.gridspec.GridSpec(2, 6, wspace=1.5, hspace=0.5, height_ratios=[4,5])
    axs = [
        fig.add_subplot(gs[0,:3]),
        fig.add_subplot(gs[0,3:]),
        fig.add_subplot(gs[1,1:-1])
    ]
    
    measdict = read_measdict('measurements/reserve/*.txt', sep='\t')

    arr = [
        axs,
        ['random-TDC0', 'random-TDC1', 'random-coincidences'],
        [r'\verb|TDC0|', r'\verb|TDC1|', 'Naključne koincidence']
    ]

    for i, (ax, measname, label) in enumerate(zip(*arr)):

        t_, N, PDF_t_ = measdict[measname]
        t = 1e-12 * t_
        PDF_t = 1e12 * PDF_t_

        int_PDF_t = np.sum(PDF_t * np.mean(np.diff(t)))
        print(f'integral PDF(t) dt = {int_PDF_t:.6f}')
        print(f'{100 * np.sum(PDF_t > 0) / len(PDF_t)}% of the bins have zero counts')

        if i < 2:

            plt, = ax.plot(
                1e3 * t, 1e-3 * PDF_t,
                lw=0.8,
            )
            ax.fill_between(
                1e3 * t, 1e-3 * PDF_t,
                color=plt.get_color(), alpha=0.2,
            )

            par, cov = curve_fit(exponent, t, PDF_t, p0=[1000])
            R_fit, = par
            σ_R, = np.sqrt(np.diag(cov))
            print(f'R_fit = ({R_fit:.1f} +/- {σ_R:.1f}) 1/s')

            t_elapsed = 60.0 if i == 0 else 30
            counts = np.sum(N)
            R_mean = counts / t_elapsed
            print(f'R_mean = {R_mean:.2f} 1/s ({int(counts)} counts in {t_elapsed} s)')
            δ = abs(R_mean/R_fit - 1)
            print(f'δ = {δ:.2f}')

            R_mid = (R_mean + R_fit)/2
            print(f'R_mid = ({R_mid:.0f} +/- {δ*R_mid:.0f}) 1/s')
            print()

            t_lin = np.linspace(np.min(t), np.max(t), num=501)
            plt, = ax.plot(
                1e3 * t, 1e-3 * exponent(t, R_fit),
                ls=':', color='r',
                label=f'$R_{i+1} = ({R_mid:.0f} \\pm {δ*R_mid:.0f})\\,\\mathrm{{s}}^{{-1}}$'
            )

            ax.set(
                title=f'{label}, časi med razpadi',
                xlabel=r'$t\,[\mathrm{ms}]$',
                ylabel=(
                    r'gostota $\frac{\mathrm{d}p}{\mathrm{d}t}\,[\mathrm{ms}^{-1}]$'
                    if i == 0 else None
                ),
                xlim=(0,6),
                yscale='log',
            )
            ax.legend(frameon=False, loc='upper right')
        else:

            n = 3*(len(t)//3)
            t = t[0:n:3]
            PDF_t = PDF_t[0:n:3] + PDF_t[1:n:3] + PDF_t[2:n:3]

            n = 3*(len(t)//3)
            t = t[0:n:3]
            PDF_t = PDF_t[0:n:3] + PDF_t[1:n:3] + PDF_t[2:n:3]

            τ = 1e-6
            R1, ΔR1 = 1189, 12
            R2, ΔR2 = 1395, 24
            R_predicted = τ * R1*R2
            ΔR_predicted = np.sqrt(
                (τ * ΔR1*R2)**2
                + (τ * R1*ΔR2)**2
            )
            print(f'R_mean_predicted = ({R_predicted:.2f} +/- {ΔR_predicted:.2f}) 1/s')

            t_elapsed = 30.0
            counts = np.sum(N)
            R_mean = counts / t_elapsed
            print(f'R_mean = {R_mean:.2f} 1/s ({int(counts)} counts in {t_elapsed} s)')

            ax.bar(
                1e9 * t, 1e-9 * PDF_t, width=1.05e9 * np.mean(np.diff(t)),
                color='tab:blue', alpha=0.8,
                label=f'$R_{{12}} = {R_mean:.2f}\\,\\mathrm{{s^{{-1}}}}$'
            )
            ax.set(
                title=f'{label}, medkanalni časi',
                xlabel=r'$t\,[\mathrm{ms}]$',
                ylabel=(
                    r'gostota $\frac{\mathrm{d}p}{\mathrm{d}t}\,[\mathrm{ms}^{-1}]$'
                ),
                xlim=(-1,1),
            )
            ax.legend(frameon=False)


    fig.subplots_adjust(right=0.95)
    
    fig.savefig(savefig, dpi=300)

# fig_single(savefig='single.pdf')
fig_correlation(savefig='correlation.pdf')
fig_correlation_by_angle(savefig='correlation-by-angle.pdf')
fig_random_coincidence(savefig='random-coincidence.pdf')
