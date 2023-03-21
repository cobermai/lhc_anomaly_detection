from datetime import datetime
import os
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.stats import chi2
from scipy import signal

from src.datasets.rb_fpa_prim_quench_ee_plateau2_V2 import RBFPAPrimQuenchEEPlateau2_V2
from src.datasets.rb_fpa_prim_quench_ee_plateau_V2 import RBFPAPrimQuenchEEPlateau_V2
from src.models.nmf import NMF
from src.utils.frequency_utils import get_fft_of_DataArray, scale_fft_amplitude, get_ifft_of_DataArray, \
    complex_to_polar, polar_to_complex
from src.utils.utils import dict_to_df_meshgrid, interp
from src.visualisation.NMF_visualization import plot_NMF_components, plot_ts_circle, plot_loss_hist, plot_outliers, \
    plot_component_examples, plot_event_componet_weigts


def NMF_sensitivity_analysis(hyperparameter, outlier_loss, out_path):
    df_meshgrid = dict_to_df_meshgrid(hyperparameter)
    print(f"n_iter: {len(df_meshgrid)}")
    df_loss = pd.DataFrame({'fpa_identifier': ds.event.values})
    df_p_values = pd.DataFrame({'fpa_identifier': ds.event.values})
    loss = []
    df_result = df_meshgrid.copy()
    for index, row in df_meshgrid.iterrows():
        experiment_name = '_'.join(row.astype(str).values)
        experiment_path = out_path / experiment_name
        experiment_path.mkdir(parents=True, exist_ok=True)

        print(row)

        # Preprocessing
        ds_detrend = dataset_creator.detrend_dim(ds, deg=row["trend_deg"])
        f_window = window_functions[row["f_window"]]
        da_processed = ds_detrend.data * f_window(len(ds.time))
        row = row.drop(["trend_deg", "f_window"])

        # Calculate fft
        f_lim = (0, 534)
        da_fft = get_fft_of_DataArray(data=da_processed, f_lim=f_lim)
        da_fft_amp = scale_fft_amplitude(data=da_fft, f_window=f_window)
        da_fft_amp = da_fft_amp[:, :, da_fft_amp.frequency < f_lim[1]]
        _, da_fft_phase = complex_to_polar(da_fft)

        # Flatten data for nmf
        na_fft_flat = np.nan_to_num(da_fft_amp.data.reshape(-1, np.shape(da_fft_amp.data)[2]))

        # Train NMF
        init_components = True
        if init_components:
            df_components = pd.read_csv("../data/1EE_final_components_V2/8_components_fitted.csv",
                                        index_col="Unnamed: 0")
            row["n_components"] = len(df_components.columns)
            H_init = df_components.values.T
            nmf_model = NMF(**row.to_dict())
            nmf_model.components_ = H_init
            nmf_model.n_components_ = len(df_components.columns)
            #W = nmf_model.transform(X=na_fft_flat, H= H_init)
            #nmf_model.fit(X=na_fft_flat, H=H_init, not_init_H_idx=True)
        else:
            nmf_model = NMF(**row.to_dict())
            nmf_model.fit(X=na_fft_flat)
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
        if outlier_loss == "nmf":
            loss_sample = da_fft_amp.data - na_fft_nmf_amp
        elif outlier_loss == "fft+nmf":
            loss_sample = da_processed.values - np.real(da_ifft_nmf).values
        masked_arr = np.ma.masked_invalid(loss_sample)
        loss_magnet = np.linalg.norm(masked_arr, axis=2)
        #loss_max = np.nanmax(loss_magnet, axis=1)
        outlier_radius = 3
        loss_max = pd.DataFrame(loss_magnet.T).rolling(outlier_radius).mean().max().values

        total_loss = np.linalg.norm(masked_arr.compressed())
        df_loss[experiment_name] = loss_max
        component_overlap = np.sum(np.linalg.norm(H, axis=0)) #/ len(H)
        component_overlap_norm = np.sum(np.linalg.norm(H_norm, axis=0)) #/ len(H_norm)

        loss.append([total_loss, loss_magnet, component_overlap, component_overlap_norm])
        df_result.loc[index, "Loss"] = total_loss
        df_result.loc[index, "L2_C"] = component_overlap
        df_result.loc[index, "L2_C_norm"] = component_overlap_norm

        # calculate p values
        params_fit = chi2.fit(loss_max, floc=0)  # fit df, and fshape
        p_values = 1 - chi2.cdf(loss_max, *params_fit)
        df_p_values[experiment_name] = p_values

        # plot ts circle
        plot_ts_circle(ds, da_processed, da_fft_amp, na_fft_nmf_amp, da_ifft_nmf, da_ifft, ds_fft_nmf_rec, ds_fft_rec,
                       experiment_path / 'event.svg', event=2, magnet=150)
        plot_ts_circle(ds, da_processed, da_fft_amp, na_fft_nmf_amp, da_ifft_nmf, da_ifft, ds_fft_nmf_rec, ds_fft_rec,
                       experiment_path / 'event1.svg', event=110, magnet=59)

        # plot loss histogram
        plot_loss_hist(loss_max, experiment_path, params_fit)

        # plot components
        plot_NMF_components(da_fft_amp.frequency, H, experiment_path / 'components.png')

        # plot event signals where each component is the biggest
        plot_component_examples(H_norm, W_norm, da_fft_amp, ds.data, experiment_path, n_examples=3)

        plt.tight_layout()
        plt.savefig(experiment_path / 'component_examples.png')

        pd.DataFrame(H_norm.T,
                     index=da_fft_amp.frequency.values,
                     columns=[f"component_{i}" for i in range(len(H_norm))]).to_csv(experiment_path / "components.csv")


        plot_event_componet_weigts(W_norm, H_norm, da_fft_amp, ds, mp3_fpa_df, experiment_path)

        plt.close('all')
        del ds_detrend, da_processed, da_fft, da_fft_amp, da_fft_phase, na_fft_flat, H_norm, W_norm, H, W, da_ifft, \
            da_ifft_win, ds_fft_rec, na_fft_nmf_amp, da_fft_nmf_amp_unscaled, da_fft_nmf, da_ifft_nmf, \
            da_ifft_nmf_win, ds_fft_nmf_rec
        gc.collect()

    df_result.to_csv(output_path / 'result.csv')
    df_loss.to_csv(output_path / 'loss.csv', index=False)
    df_p_values['median'] = df_p_values.drop(columns=['fpa_identifier']).median(axis=1)
    df_p_values['std'] = df_p_values.drop(columns=['fpa_identifier']).std(axis=1)

    df_p_values = df_p_values.sort_values(by='median').reset_index(drop=True)
    magnets = mp3_fpa_df_unique.set_index('fpa_identifier', drop=True).loc[df_p_values.fpa_identifier, 'Position']
    df_p_values = df_p_values.set_index(magnets, drop=True)
    df_p_values.to_csv(output_path / 'p_values.csv')

    # plot loss
    loss = np.array(loss)
    fig, ax1 = plt.subplots()
    ax1.plot(loss[:, 0], 'g-', label=outlier_loss)
    ax1.set_ylabel(f'{outlier_loss} loss', color='g')
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(loss[:, 2], 'b', label="L2 C")
    ax2.plot(loss[:, 3], 'r', label="L2 C norm")
    ax2.legend()
    plt.savefig(output_path / 'loss.png')


    n_outliers = 10
    loss_hist = np.stack(loss[:, 1], axis=-1)
    outlier_event_index = np.arange(len(ds.event.values))[np.isin(ds.event.values,
                                                                  df_p_values.head(n_outliers).fpa_identifier.values)]
    plot_outliers(ds, df_p_values, loss_hist, out_path, da_fft_amp=None, n_outliers=10, mp3_fpa_df=mp3_fpa_df)

    #for i, row in df_p_values.head(n_outliers).iterrows():
    #    event_loss = np.nanmean(loss_hist, axis=-1)[outlier_event_index[i]].reshape(-1)
    #    outlier_magnet_index = np.nanargmax(event_loss)
    #    plot_ts_circle(ds, da_processed, da_fft_amp, na_fft_nmf_amp, da_ifft_nmf, da_ifft, ds_fft_nmf_rec, ds_fft_rec,
    #                   experiment_path / f'event_outlier_{i}.png', event=outlier_event_index[i],
    #                   magnet=outlier_magnet_index)

    var_loss = np.nanmax(loss_hist[outlier_event_index], axis=1)
    plt.figure()
    plt.plot(var_loss.T)
    plt.ylabel("Max Loss")
    plt.xlabel("# Sensitivity Analysis")
    plt.legend([f"Outlier {i}" for i in range(len(var_loss))])
    plt.savefig(output_path / 'outliers/outlier_loss.png')


if __name__ == "__main__":
    # define paths to read
    context_path = Path("../data/MP3_context_data/20230313_RB_processed.csv")
    snapshot_context_path = Path("../data/RB_snapshot_context.csv")

    # define paths to read + write
    dataset_path = Path('D:\\datasets\\20230313_RBFPAPrimQuenchEEPlateau2_V2')
    output_path = Path(f"../output/{os.path.basename(__file__)}/{datetime.now().strftime('%Y-%m-%dT%H.%M.%S.%f')}")
    output_path.mkdir(parents=True, exist_ok=True)

    # load desired fpa_identifiers
    mp3_fpa_df = pd.read_csv(context_path)
    snapshot_context_df = pd.read_csv(snapshot_context_path)
    mp3_fpa_df = pd.concat([mp3_fpa_df[mp3_fpa_df['timestamp_fgc'] >= 1526582397220000000], snapshot_context_df])
    drop_events = ["RB_RB.A78_1619330143440000000", "RB_RB.A45_1544355147780000000", "RB_RB.A45_1544300287820000000"] # known outliers to drop
    mp3_fpa_df = mp3_fpa_df[~mp3_fpa_df.fpa_identifier.isin(drop_events)]
    mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['fpa_identifier'])

    # add snapshot data
    dataset_creator = RBFPAPrimQuenchEEPlateau2_V2()
    ds = dataset_creator.load_dataset(fpa_identifiers=mp3_fpa_df_unique.fpa_identifier.values,
                                      dataset_path=dataset_path)

    ds = ds.isel(time=slice(0, 400))
    # model is not trained on data before 2021 and events with fast secondary quenches
    window_functions = {"hamming": signal.windows.hamming}

    outlier_loss = "nmf"
    hyperparameter = {
        "trend_deg": [1],
        "f_window": list(window_functions.keys()),
        "n_components": [6],
        "solver": ["mu"],
        "beta_loss": ['frobenius'],
        "init": ["nndsvda"],
        "shuffle": ["False"]
    }
    NMF_sensitivity_analysis(hyperparameter, outlier_loss, output_path)


    """ 
    window_functions = {"ones": np.ones,
                        "hanning": np.hanning,
                        "bartlett": signal.windows.bartlett,
                        "blackman": signal.windows.blackman,
                        "flattop": signal.windows.flattop,
                        "hamming": signal.windows.hamming,
                        "tukey": signal.windows.tukey}
   
    outlier_loss = "nmf"
    hyperparameter = {
        "trend_deg": [0,1],
        "f_window": list(window_functions.keys()),
        "n_components": [2,3,4,5,6,8,9,10,11,12], #2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,19,20
        "solver": ["mu"],
        "beta_loss": ['frobenius', 'kullback-leibler'], #'frobenius', 'kullback-leibler', 'itakura-saito'
        "init": ["nndsvda"],
        #"l1_ratio": [0.5],
        #"alpha": [0.1],
        #"max_iter": [1000],
        "shuffle": ["False"],
        #"ortho_reg": [0]
    }
    NMF_sensitivity_analysis(hyperparameter, outlier_loss, output_path)
    """


