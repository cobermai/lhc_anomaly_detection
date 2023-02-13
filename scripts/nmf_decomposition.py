from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.stats import gamma, chi2

from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau
from src.models.nmf import NMF
from src.utils.frequency_utils import get_fft_of_DataArray, scale_fft_amplitude, get_ifft_of_DataArray, \
    complex_to_polar, polar_to_complex

def plot_loss(figpath):
    outlier_events = ["RB_RB.A78_1619330143440000000",
                      "RB_RB.A12_1621014819920000000",
                      "RB_RB.A45_1620797547820000000"]
    bool_outlier = np.isin(ds.event.values, outlier_events)

    plt.figure(figsize=(7, 5))
    for line in fft_nmf_loss[bool_outlier]:
        plt.axvline(line, c='orange')

    plt.hist(fft_nmf_loss[~bool_test], bins=200, density=True)
    plt.ylabel("$||x^*[n] - \hat{x}_{NMF}^*[n]||$", fontsize=15)

    # Fit a gamma distribution to the data
    #params_fit = gamma.fit(fft_nmf_loss[~bool_test])
    params_fit = chi2.fit(fft_nmf_loss[~bool_test])

    # Plot the pdf of the fitted gamma distribution
    x = np.linspace(0, fft_nmf_loss[~bool_test].max(), 300)
    pdf = chi2.pdf(x, *params_fit)
    plt.plot(x, pdf, 'k--', lw=2)

    upper = chi2.ppf(0.99, *params_fit)
    plt.axvline(upper, c='red')

    plt.savefig(figpath)

    p_values = 1 - chi2.cdf(fft_nmf_loss[~bool_test], *params_fit)
    df_p_values = pd.DataFrame({'fpa_identifier': ds.event.values[~bool_test],
                                'loss': fft_nmf_loss[~bool_test],
                                'p_values': p_values})
    df_p_values.to_csv(output_path / 'p_values.csv')

def plot_event(figpath):
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

if __name__ == "__main__":
    # define paths to read
    context_path = Path("../data/RB_TC_extract_2022_07_07_processed_filled.csv")

    # define paths to read + write
    dataset_path = Path('D:\\datasets\\20220707_prim_ee_plateau_dataset')
    output_path = Path(f"../output/{os.path.basename(__file__)}/{datetime.now().strftime('%Y-%m-%dT%H.%M.%S.%f')}")
    output_path.mkdir(parents=True, exist_ok=True)

    # load desired fpa_identifiers
    mp3_fpa_df = pd.read_csv(context_path)
    dataset_creator = RBFPAPrimQuenchEEPlateau()
    ds = dataset_creator.load_dataset(fpa_identifiers=mp3_fpa_df.fpa_identifier.unique(),
                                      dataset_path=dataset_path,
                                      drop_data_vars=['simulation', 'el_position_feature', 'event_feature'])

    # model is not trained on data before 2021 and events with fast secondary quenches
    test_conditions = ((mp3_fpa_df['Delta_t(iQPS-PIC)'] / 1000 < 5) &
                       (mp3_fpa_df['Nr in Q event'].astype(str) != '1')) | \
                      (mp3_fpa_df['timestamp_fgc'] < 1611836512820000000)
    bool_test = np.isin(ds.event.values, mp3_fpa_df[test_conditions].fpa_identifier.unique())
    # add dims for indexing flattened data
    bool_train_flat = np.stack([~bool_test for l in range(len(ds.el_position))]).T.reshape(-1)

    # Preprocessing
    f_window = np.hamming
    cutoff_freq = 300
    filt_order = 5
    ds_detrend = dataset_creator.detrend_dim(ds)
    da_win = ds_detrend.data * f_window(len(ds.time))
    da_processed = dataset_creator.lowpass_filter_DataArray(da=da_win, cutoff=cutoff_freq, order=filt_order)

    # calculate fft
    f_lim = (0, 500)
    da_fft = get_fft_of_DataArray(data=da_processed, f_lim=f_lim)
    da_fft_amp = scale_fft_amplitude(data=da_fft, f_window=f_window)
    da_fft_amp = da_fft_amp[:, :, da_fft_amp.frequency < f_lim[1]]
    _, da_fft_phase = complex_to_polar(da_fft)

    # scale fft data
    na_fft_flat = np.nan_to_num(da_fft_amp.data.reshape(-1, np.shape(da_fft_amp.data)[2]))

    # fit and transform NMF, fit both W and H
    hyperparameter = {
        "n_components": 3,
        "solver": "cd",
        "beta_loss": 'frobenius',
        "init": "nndsvd",
        "tol": 1e-5,
        "max_iter": 200,
        "l1_ratio": 0.5,
        "alpha": 0,
        "shuffle": "False",
        "ortho_reg": 0
    }
    nmf_model = NMF(**hyperparameter)
    nmf_model.fit(X=na_fft_flat[bool_train_flat])
    W = nmf_model.transform(X=na_fft_flat)
    H = nmf_model.components_
    H_norm, W_norm = nmf_model.normalize_H(H=H, W=W)

    # reconstruct fft without NMF
    da_ifft = get_ifft_of_DataArray(data=da_fft, start_time=ds.time.values[0])

    # Postprocessing
    da_ifft_win = da_ifft / f_window(len(ds.time))
    ds_fft_rec = xr.apply_ufunc(np.real, da_ifft_win).to_dataset()
    ds_fft_rec["polyfit_coefficients"] = ds_detrend.polyfit_coefficients
    ds_fft_rec = dataset_creator.trend_dim(ds_fft_rec)

    # reconstruct fft with NMF
    na_fft_nmf_amp = (W_norm @ H_norm).reshape(da_fft_amp.data.shape)
    da_fft_nmf_amp = xr.zeros_like(da_fft, dtype=float)
    da_fft_nmf_amp[:, :, :na_fft_nmf_amp.shape[-1]] = na_fft_nmf_amp

    # assemble reconstructed fft
    da_fft_nmf_amp_unscaled = scale_fft_amplitude(data=da_fft_nmf_amp, f_window=f_window, is_polar=True)
    da_fft_nmf = polar_to_complex(da_fft_nmf_amp_unscaled, da_fft_phase)

    # ifft of reconstruction
    da_ifft_nmf = get_ifft_of_DataArray(data=da_fft_nmf, start_time=ds.time.values[0])

    # Postprocessing
    da_ifft_nmf_win = da_ifft_nmf / f_window(len(ds.time))
    ds_fft_nmf_rec = xr.apply_ufunc(np.real, da_ifft_nmf_win).to_dataset()
    ds_fft_nmf_rec["polyfit_coefficients"] = ds_detrend.polyfit_coefficients
    ds_fft_nmf_rec = dataset_creator.trend_dim(ds_fft_nmf_rec)

    # Loss
    fft_loss_sample = da_processed.values - np.real(da_ifft).values
    fft_nmf_loss_sample = da_processed.values - np.real(da_ifft_nmf).values
    nmf_loss_sample = da_fft_amp.data - na_fft_nmf_amp
    fft_loss = np.linalg.norm(np.nan_to_num(fft_loss_sample), axis=2).max(axis=1).reshape(-1)
    fft_nmf_loss = np.linalg.norm(np.nan_to_num(fft_nmf_loss_sample), axis=2).max(axis=1).reshape(-1)
    nmf_loss = np.linalg.norm(np.nan_to_num(nmf_loss_sample), axis=2).max(axis=1).reshape(-1)

    # plot ts circle
    plot_event(output_path / 'event.png')

    # plot loss
    plot_loss(output_path / 'loss.png')

    plt.figure()
    plt.plot(da_fft_amp.frequency, H.T)
    plt.legend([f"component {i}" for i in range(len(H))])
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Voltage / V')
    plt.savefig(output_path / 'components.png')





