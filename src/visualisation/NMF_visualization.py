import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gamma, chi2
from sklearn.cluster import KMeans

from src.visualisation.fft_visualisation import plot_circuit_frequencies


def plot_NMF_components(freqency, H, fig_dir):
    plt.figure(figsize=(12, 5))
    plt.plot(freqency, H.T)
    plt.legend([f"component {i}" for i in range(len(H))])
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Voltage / V')
    plt.savefig(fig_dir)

def plot_ts_circle(ds, da_processed, da_fft_amp, na_fft_nmf_amp, da_ifft_nmf, da_ifft, ds_fft_nmf_rec, ds_fft_rec,
                   figpath, event=2, magnet=150):
    fig, ax = plt.subplots(3, 3, figsize=(17, 11))

    # x(t)
    ax[0, 0].plot(ds.time, ds.data[event, magnet].T, c="b")
    ax[0, 0].set_xlabel('Time / s')
    ax[0, 0].set_ylabel('Voltage / V')
    ax[0, 0].legend(['$x[n]]$'])

    # x*(t)
    ax[0, 1].plot(ds.time, da_processed.data[event, magnet].T, c="b")
    ax[0, 1].set_xlabel('Time / s')
    ax[0, 1].set_ylabel('Voltage / V')
    ax[0, 1].legend(['$x^*[n]$'])

    # X(k)
    ax[0, 2].plot(da_fft_amp.frequency, da_fft_amp[event, magnet].T, "-o", c="g")
    ax[0, 2].set_xlabel('Frequency / Hz')
    ax[0, 2].set_ylabel('Voltage / V')
    ax[0, 2].set_xlim((0, 100))
    ax[0, 2].legend(['$|X[k]|$'])

    # WH
    ax[1, 2].plot(da_fft_amp.frequency, na_fft_nmf_amp[event, magnet].T, "-o", c="r")
    ax[1, 2].set_xlabel('Frequency / Hz')
    ax[1, 2].set_ylabel('Voltage / V')
    ax[1, 2].set_xlim((0, 100))
    ax[1, 2].legend(['$|\hat{X}[n]|$'])

    # \hat{x}*(t)
    ax[1, 1].plot(ds.time, da_processed.data[event, magnet].T, c="b")
    ax[1, 1].plot(ds.time, da_ifft_nmf[event, magnet].T, c="r")
    ax[1, 1].plot(ds.time, da_ifft[event, magnet].T, c="g")
    ax[1, 1].set_xlabel('Time / s')
    ax[1, 1].set_ylabel('Voltage / V')
    ax[1, 1].legend(['$x^*[n]$', '$\hat{x}^*_{NMF}[n]$', '$\hat{x}^*_{FFT}[n]$'])
    #ax[1, 1].legend(['$x^*[n]$',  '$\hat{x}^*_{FFT}[n]$'])

    # \hat{x}(t)
    ax[1, 0].plot(ds.time, ds.data[event, magnet].T, c="b")
    ax[1, 0].plot(ds.time, ds_fft_nmf_rec.data[event, magnet].T, c="r")
    ax[1, 0].plot(ds.time, ds_fft_rec.data[event, magnet].T, c="g")
    ax[1, 0].set_xlabel('Time / s')
    ax[1, 0].set_ylabel('Voltage / V')
    ax[1, 0].legend(['$x[n]]$', '$\hat{x}_{NMF}[n]$', '$\hat{x}_{FFT}[n]$'])
    #ax[1, 0].legend(['$x[n]]$', '$\hat{x}_{FFT}[n]$'])
    ax[1, 0].set_ylim(ax[0, 0].get_ylim())

    # NMF+FFT+Preprocessing Loss
    pp_fft_nmf_loss = ds.data[event, magnet].T.values - ds_fft_nmf_rec.data[event, magnet].T.values
    ax[2, 0].plot(ds.time, pp_fft_nmf_loss, c="r")
    ax[2, 0].plot(ds.time, ds.data[event, magnet].T.values - ds_fft_rec.data[event, magnet].T.values , c="g")
    ax[2, 0].set_title(f'Preprocessing + FFT + NMF Loss {np.linalg.norm(pp_fft_nmf_loss):.2f}')
    ax[2, 0].set_xlabel('Time / s')
    ax[2, 0].set_ylabel('Voltage / V')
    ax[2, 0].legend(['$x^[n] - \hat{x}_{NMF}[n]$', '$x^[n] - \hat{x}_{FFT}[n]$'])

    # NMF+FFT Loss
    fft_nmf_loss = da_processed.data[event, magnet].T - da_ifft_nmf.data[event, magnet].T
    ax[2, 1].plot(ds.time, fft_nmf_loss, c="r")
    ax[2, 1].set_title(f'FFT + NMF Loss {np.linalg.norm(fft_nmf_loss):.2f}')
    ax[2, 1].plot(ds.time, da_processed.data[event, magnet].T  - da_ifft.data[event, magnet].T, c="g")
    ax[2, 1].set_xlabel('Time / s')
    ax[2, 1].set_ylabel('Voltage / V')
    ax[2, 1].legend(['$x^*[n] - \hat{x}^*_{NMF}[n]$', '$x^*[n] - \hat{x}^*_{FFT}[n]$'])

    # NMF loss
    nmf_loss = da_fft_amp[event, magnet].T.values - na_fft_nmf_amp[event, magnet].T
    ax[2, 2].plot(da_fft_amp.frequency, nmf_loss, c="r")
    ax[2, 2].set_title(f'NMF Loss: {np.linalg.norm(nmf_loss):.2f}')
    ax[2, 2].set_xlabel('Frequency / Hz')
    ax[2, 2].set_ylabel('Voltage / V')
    ax[2, 2].legend(['$|X[k]| - |\hat{X}[k]|$'])

    plt.tight_layout()
    plt.savefig(figpath)

def plot_loss_hist(loss, output_path, params_fit=None):

    plt.figure(figsize=(7, 5))
    #for line in loss[bool_outlier]:
    #    plt.axvline(line, c='orange')

    plt.hist(loss, bins=200, density=True)
    plt.ylabel("# Events", fontsize=15)
    plt.xlabel("$|||X[k]| - |\hat{X}[k]|||$", fontsize=15)

    # Fit a gamma distribution to the data
    #params_fit = gamma.fit(fft_nmf_loss[~bool_test])
    if params_fit is None:
        params_fit = chi2.fit(loss)

    # Plot the pdf of the fitted gamma distribution
    x = np.linspace(0, loss.max(), 300)
    pdf = chi2.pdf(x, *params_fit)
    plt.plot(x, pdf, 'k--', lw=2)

    upper = chi2.ppf(0.99, *params_fit)
    plt.axvline(upper, c='red')

    plt.legend(['chi2 pdf', '99% conf. interval', 'data'])
    plt.tight_layout()
    plt.savefig(output_path / 'loss_hist.png')

def plot_outliers(ds, df_p_values, loss, out_path, n_outliers, mp3_fpa_df, da_fft_amp=None):
    outlier_path = out_path / 'outliers'
    outlier_path.mkdir(parents=True, exist_ok=True)

    j = 0
    for i, row in df_p_values.head(n_outliers).iterrows():
        outlier_event_index = np.isin(ds.event.values, row['fpa_identifier'])
        event_loss = np.nanmean(loss, axis=-1)[outlier_event_index].reshape(-1)
        
        outlier_magnet_index = np.nanargmax(event_loss)
        quenched_magnet_index = mp3_fpa_df[mp3_fpa_df.fpa_identifier == row['fpa_identifier']]['#Electric_circuit'].values[0]

        plt.figure()
        plt.plot(ds.time, ds.data.loc[{'event': row['fpa_identifier']}].T, alpha=0.5)
        plt.plot(ds.time, ds.data.loc[{'event': row['fpa_identifier']}].values[outlier_magnet_index].T)
        #plt.title(f"p-value: {row['median']:.4} +/-{row['std']:.4}")
        plt.xlabel('Time / s')
        plt.ylabel('Voltage / V')
        plt.savefig(outlier_path / f"{j}_{row['fpa_identifier']}.png")

        plt.figure()
        plt.title(f"{row['fpa_identifier']}\nqench@{quenched_magnet_index} maxloss@{outlier_magnet_index+1}")
        mean = np.nanmean(loss[outlier_event_index], axis=-1).reshape(-1)
        std = np.nanstd(loss[outlier_event_index], axis=-1).reshape(-1)
        plt.plot(np.arange(1, 155), mean)
        plt.xlabel('El. Position')
        plt.ylabel('Loss')
        plt.axvline(quenched_magnet_index, c="r")
        plt.fill_between(np.arange(1, 155), mean-std, mean+std, alpha=0.2)
        plt.savefig(outlier_path / f"{j}_{row['fpa_identifier']}_loss.png")

        #fig, ax = plt.subplots(figsize=(10,5))
        #im = plot_circuit_frequencies(ax, da_fft_amp.values[outlier_event_index], da_fft_amp.frequency, vmin=1e-5, vmax=1)
        #ax.set_xlabel(f'El. Position')
        #ax.set_ylabel(f'Frequency / Hz')
        #cbar = fig.colorbar(im, ax=ax)
        #cbar.set_label('Voltage / V')
        #plt.savefig(outlier_path / f"{i}_{row['fpa_identifier']}_pfm.png")
        j+=1


    plt.figure()
    df_p_values.drop(columns=['fpa_identifier', 'median', 'std']).head(n_outliers).T.boxplot()
    plt.xlabel('Outlier number')
    plt.ylabel('p-value')
    plt.axhline(y=0.01, color='r', linestyle='-', label='99% confidence interval')
    plt.legend()
    plt.savefig(out_path / 'outlier_boxplot.png')


def plot_component_examples(H_norm, W_norm, da_fft_amp, da_processed, experiment_path, n_examples = 5):
    # cluster component weight
    # TODO indexing hoes not work with n_components =1
    fig, ax = plt.subplots(len(H_norm), 2, figsize=(8, len(H_norm) * 3))
    W_ratio = np.nan_to_num(W_norm / np.nansum(W_norm, axis=1, keepdims=True))
    idx_W_sorted_flat = np.argsort(W_ratio, axis=0)[::-1][:n_examples]
    idx_W_examples = np.array(np.unravel_index(idx_W_sorted_flat, da_fft_amp.data.shape[:-1])).T
    c_frequencies = da_fft_amp.frequency.values[np.argmax(H_norm, axis=1)]
    idx_W_sorted_flat = np.argsort(W_norm, axis=0)[::-1][:n_examples]
    for i in range(len(idx_W_examples)):
        for j in range(idx_W_examples.shape[1]):
            ax[i, 1].plot(da_processed.time,
                          da_processed.values[idx_W_examples[i, j, 0], idx_W_examples[i, j, 1]].T)  # ,
            # label=f"{da_processed.event.values[idx_W_examples[i, j, 0]]}, " +
            #      f"El. Position: {idx_W_examples[i, j, 1]}")
        ax[i, 1].set_xlabel('Time / s')
        ax[i, 1].set_ylabel('Voltage / V')
        # ax[i, 1].legend()
        # ax[i, 1].grid()
        # ax[i, 1].set_title(f"{i}. component - {int(c_frequencies[i])} Hz")
        # ax[i, 0].set_title(f"FFT amplitude examples of {i}. component - {int(c_frequencies[i])} Hz")
        for j in range(idx_W_examples.shape[1]):
            ax[i, 0].plot(da_fft_amp.frequency, da_fft_amp.values[idx_W_examples[i, j, 0], idx_W_examples[i, j, 1]].T,
                          label=f"{da_processed.event.values[idx_W_examples[i, j, 0]]}, " +
                                f"El. Position: {idx_W_examples[i, j, 1]}")
        # ax[i, 0].plot(ds_detrend.time, W_sorted_flat[i] * np.sin(2 * np.pi * c_frequencies[i] * ds_detrend.time),
        #              label=f"{int(c_frequencies[i])} Hz")
        # ax[i, 0].legend()
        ax[i, 0].set_xlabel('Frequency / Hz')
        ax[i, 0].set_ylabel('Voltage / V')
        # ax[i, 0].grid()
        #ax[i, 0].set_xlim((0, 300))
        # ax[i, 0].set_ylim(ax[i, 1].get_ylim())
    plt.tight_layout()
    plt.savefig(experiment_path / 'component_examples.png')


def plot_kmeans_centers(H_norm, W_norm, ds_detrend, da_fft_amp, experiment_path):
    # cluster component weight
    n_examples = 1
    n_clusters = len(H_norm)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(W_norm)
    centers = kmeans.cluster_centers_
    fig, ax = plt.subplots(n_clusters, 2, figsize=(15, len(H_norm) * 5))
    custom_ylim = (0, 0)
    for i, center in enumerate(centers):
        distance = np.linalg.norm(W_norm - centers[i], axis=-1)
        distance_min_index_flat = np.argsort(distance)[:n_examples]
        distance_min_index = np.array(np.unravel_index(distance_min_index_flat, da_fft_amp.data.shape[:-1])).T

        signal = ds_detrend.data.values[distance_min_index[0, 0], distance_min_index[0, 1]]
        custom_ylim = (min(custom_ylim[0], signal.min()), max(custom_ylim[1], signal.max()))

        ax[i, 1].plot(ds_detrend.time, signal)
        ax[i, 1].set_title('Time Series of Event')
        ax[i, 1].set_xlabel('Time / s')
        ax[i, 1].set_ylabel('Voltage / V')
        ax[i, 1].grid()

        ax[i, 0].plot(da_fft_amp.frequency, W_norm[distance_min_index_flat] * H_norm.T,
                      label=[f"Component {i}" for i in range(len(H_norm))])
        ax[i, 0].set_title('Weighted Components of Event')
        ax[i, 0].set_xlabel('Frequency / Hz')
        ax[i, 0].set_ylabel('Voltage / V')
        ax[i, 0].legend()
        ax[i, 0].grid()
        ax[i, 0].set_ylim(ax[i, 1].get_ylim())
    plt.setp(ax, ylim=custom_ylim)
    plt.tight_layout()
    plt.savefig(experiment_path / 'component_examples.png')