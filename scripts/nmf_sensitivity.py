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
from src.utils.utils import dict_to_df_meshgrid
from src.visualisation.NMF_visualization import plot_NMF_components, plot_ts_circle, plot_loss_hist


def NMF_sensitivity_analysis(hyperparameter, out_path):
    df_meshgrid = dict_to_df_meshgrid(hyperparameter)

    df_loss = pd.DataFrame({'fpa_identifier': ds.event.values})
    df_p_values = pd.DataFrame({'fpa_identifier': ds.event.values[~bool_test]})
    loss = []
    for index, row in df_meshgrid.iterrows():
        experiment_name = '_'.join(row.astype(str).values)
        experiment_path = out_path / experiment_name
        experiment_path.mkdir(parents=True, exist_ok=True)

        print(row)

        nmf_model = NMF(**row.to_dict())
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
        fft_nmf_loss_sample = da_processed.values - np.real(da_ifft_nmf).values
        fft_nmf_loss_max = np.linalg.norm(np.nan_to_num(fft_nmf_loss_sample), axis=2).max(axis=1).reshape(-1)
        fft_nmf_loss = np.linalg.norm(np.nan_to_num(fft_nmf_loss_sample))
        df_loss[experiment_name] = fft_nmf_loss_max

        nmf_loss_sample = da_fft_amp.data - na_fft_nmf_amp
        nmf_loss = np.linalg.norm(np.nan_to_num(nmf_loss_sample))
        loss.append([fft_nmf_loss, nmf_loss])

        #calculate p values
        params_fit = chi2.fit(fft_nmf_loss_max[~bool_test])
        p_values = 1 - chi2.cdf(fft_nmf_loss_max[~bool_test], *params_fit)
        df_p_values[experiment_name] = p_values

        # plot ts circle
        plot_ts_circle(ds, da_processed, da_fft_amp, na_fft_nmf_amp, da_ifft_nmf, da_ifft, ds_fft_nmf_rec, ds_fft_rec,
                       experiment_path / 'event.png')

        # plot loss histogram
        plot_loss_hist(fft_nmf_loss_max[~bool_test], ds.event.values[~bool_test], experiment_path)

        # plot components
        plot_NMF_components(da_fft_amp.frequency, H, experiment_path / 'components.png')

    # plot loss
    loss = np.array(loss)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(loss[:, 0], 'g-')
    ax2.plot(loss[:, 1], 'b-')
    ax1.set_ylabel('fft+nmf loss', color='g')
    ax2.set_ylabel('nmf loss', color='b')
    plt.savefig(output_path / 'loss.png')

    df_p_values['median'] = df_p_values.drop(columns=['fpa_identifier']).median(axis=1)
    df_p_values['std'] = df_p_values.drop(columns=['fpa_identifier']).std(axis=1)
    df_p_values = df_p_values.sort_values(by='median').reset_index(drop=True)
    df_loss.to_csv(output_path / 'loss.csv', index=False)
    df_p_values.to_csv(output_path / 'p_values.csv', index=False)

    outlier_path = out_path / 'outliers'
    outlier_path.mkdir(parents=True, exist_ok=True)
    for i, row in df_p_values.head(5).iterrows():
        plt.figure()
        plt.plot(ds.time, ds.data.loc[{'event': row['fpa_identifier']}].T)
        plt.title(f"p-value: {row['median']:.4} +/-{row['std']:.4}")
        plt.xlabel('Time / s')
        plt.ylabel('Voltage / V')
        plt.savefig(outlier_path / f"{i}_{row['fpa_identifier']}.png")

    plt.figure()
    df_p_values.drop(columns=['fpa_identifier', 'median', 'std']).head(10).T.boxplot()
    plt.xlabel('Outlier number')
    plt.ylabel('p-value')
    plt.axhline(y=0.01, color='r', linestyle='-', label='99% confidence interval')
    plt.legend()
    plt.savefig(output_path / 'outlier_boxplot.png')


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
    da_processed = ds_detrend.data * f_window(len(ds.time))
    #da_processed = dataset_creator.lowpass_filter_DataArray(da=da_win, cutoff=cutoff_freq, order=filt_order)

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
        "n_components": [2, 3, 4, 5, 6],
        "solver": ["cd"],
        "beta_loss": ['frobenius'],
        "init": ["nndsvd"],
        "tol": [1e-5],
        "max_iter": [200],
        "l1_ratio": [0.5],
        "alpha": [0],
        "shuffle": ["False"],
        "ortho_reg": [0]
    }
    NMF_sensitivity_analysis(hyperparameter, output_path)
