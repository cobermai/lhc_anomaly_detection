import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gamma, chi2

def plot_NMF_components(freqency, H, fig_dir):
    plt.figure(figsize=(12, 5))
    plt.plot(freqency, H.T)
    plt.legend([f"component {i}" for i in range(len(H))])
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Voltage / V')
    plt.savefig(fig_dir)

def plot_ts_circle(ds, da_processed, da_fft_amp, na_fft_nmf_amp, da_ifft_nmf, da_ifft, ds_fft_nmf_rec, ds_fft_rec,
                   figpath):
    event = 2
    magnet = 150
    fig, ax = plt.subplots(2, 3, figsize=(17, 7))

    # x(t)
    ax[0, 0].plot(ds.time, ds.data[event, magnet].T)
    ax[0, 0].set_xlabel('Time / s')
    ax[0, 0].set_ylabel('Voltage / V')
    ax[0, 0].legend(['$x[n]]$'])

    # x*(t)
    ax[0, 1].plot(ds.time, da_processed.data[event, magnet].T)
    ax[0, 1].set_xlabel('Time / s')
    ax[0, 1].set_ylabel('Voltage / V')
    ax[0, 1].legend(['$x^*[n]$'])

    # X(k)
    ax[0, 2].plot(da_fft_amp.frequency, da_fft_amp[event, magnet].T, c="g")
    ax[0, 2].set_xlabel('Frequency / Hz')
    ax[0, 2].set_ylabel('Voltage / V')
    ax[0, 2].legend(['$|X[k]|$'])

    # WH
    ax[1, 2].plot(da_fft_amp.frequency, na_fft_nmf_amp[event, magnet].T, c="r")
    ax[1, 2].set_xlabel('Frequency / Hz')
    ax[1, 2].set_ylabel('Voltage / V')
    ax[1, 2].legend(['$|\hat{X}[n]|$'])

    # \hat{x}*(t)
    ax[1, 1].plot(ds.time, da_ifft_nmf[event, magnet].T, c="r")
    ax[1, 1].plot(ds.time, da_ifft[event, magnet].T, c="g")
    ax[1, 1].set_xlabel('Time / s')
    ax[1, 1].set_ylabel('Voltage / V')
    ax[1, 1].legend(['$\hat{x}^*_{NMF}[n]$', '$\hat{x}^*_{FFT}[n]$'])

    # \hat{x}(t)
    ax[1, 0].plot(ds.time, ds_fft_nmf_rec.data[event, magnet].T, c="r")
    ax[1, 0].plot(ds.time, ds_fft_rec.data[event, magnet].T, c="g")
    ax[1, 0].set_xlabel('Time / s')
    ax[1, 0].set_ylabel('Voltage / V')
    ax[1, 0].legend(['$\hat{x}_{NMF}[n]$', '$\hat{x}_{FFT}[n]$'])
    ax[1, 0].set_ylim(ax[0, 0].get_ylim())

    plt.tight_layout()
    plt.savefig(figpath)

def plot_loss_hist(loss, fpa_identifiers, output_path):
    outlier_events = ["RB_RB.A78_1619330143440000000",
                      "RB_RB.A12_1621014819920000000",
                      "RB_RB.A45_1620797547820000000"]
    bool_outlier = np.isin(fpa_identifiers, outlier_events)

    plt.figure(figsize=(7, 5))
    #for line in loss[bool_outlier]:
    #    plt.axvline(line, c='orange')

    plt.hist(loss, bins=200, density=True)
    plt.ylabel("$||x^*[n] - \hat{x}_{NMF}^*[n]||$", fontsize=15)

    # Fit a gamma distribution to the data
    #params_fit = gamma.fit(fft_nmf_loss[~bool_test])
    params_fit = chi2.fit(loss)

    # Plot the pdf of the fitted gamma distribution
    x = np.linspace(0, loss.max(), 300)
    pdf = chi2.pdf(x, *params_fit)
    plt.plot(x, pdf, 'k--', lw=2)

    upper = chi2.ppf(0.99, *params_fit)
    plt.axvline(upper, c='red')

    plt.legend(['chi2 pdf', '99% conf. interval', 'data'])
    plt.savefig(output_path / 'loss_hist.png')