import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.fft import ifft

from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau
from src.models.nmf_complex import ComplexEUCNMF
from src.utils.frequency_utils import get_fft_of_DataArray, get_ifft_of_DataArray

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
    ds_detrend = dataset_creator.detrend_dim(ds)
    da_processed = ds_detrend.data * f_window(len(ds.time))

    # calculate fft
    f_lim = (0, 500)
    da_fft = get_fft_of_DataArray(data=da_processed, f_lim=f_lim)

    X = np.nan_to_num(da_fft.values.reshape(-1, da_fft.values.shape[-1]))[bool_train_flat]
    n_basis = 4
    regularizer, p = 1e-5, 0.2

    nmf = ComplexEUCNMF(n_basis=n_basis, regularizer=regularizer, p=p)
    basis, activation, phase = nmf(X, iteration=2)

    plt.figure()
    plt.plot(nmf.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(output_path / 'loss.png')

    na_components = basis[:, :, np.newaxis] * activation[np.newaxis, :, :] * phase
    na_fft_nmf = np.sum(na_components, axis=1).reshape(da_fft[~bool_test].shape)
    na_ifft = ifft(na_fft_nmf)
    #na_ifft = na_ifft / np.abs(na_ifft).max()

    plt.figure()
    plt.plot(da_fft.frequency, activation.T)
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Voltage / V')
    plt.xlim(f_lim)
    plt.savefig(output_path / 'fcomponents.png')

    plt.figure()
    plt.plot(ds.time, np.real(na_ifft[2,150]), color='red', label='NMF')
    plt.plot(ds.time, np.real(ifft(da_fft.data[2, 150])), color='green', label='FFT')
    plt.plot(ds.time, da_processed.data[2, 150], color='blue', label='Original')
    plt.xlabel('Time / s')
    plt.ylabel('Voltage / V')
    plt.legend()
    plt.savefig(output_path / 'reconstruction.png')

    plt.figure()
    plt.plot(ds.time, np.abs(na_ifft[2,150]), color='blue')
    plt.plot(ds.time, da_processed.data[2, 150], color='black')
    plt.xlabel('Time / s')
    plt.ylabel('Voltage / V')
    plt.savefig(output_path / 'reconstruction_abs.png')

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(n_basis, 2, figsize=(10, 3 * n_basis))
    for i in range(n_basis):
        Z = basis[:, i: i + 1] * activation[i: i + 1, :] * phase[:, i, :]
        Z_reshaped = Z.reshape(da_fft[~bool_test].shape)
        z = ifft(Z_reshaped[2, 150])
        #z = z / np.abs(z).max()

        ax[i, 0].plot(da_fft.frequency, Z_reshaped[2, 150], c=default_colors[i])
        ax[i, 0].set_xlabel('Frequency / Hz')
        ax[i, 0].set_ylabel('Voltage / V')
        ax[i, 0].set_xlim(f_lim)

        ax[i, 1].plot(ds.time, np.real(z), label=f'Component {i}', c=default_colors[i])
        ax[i, 1].set_xlabel('Time / s')
        ax[i, 1].set_ylabel('Voltage / V')

    plt.savefig(output_path / f'component_{i}.png')


    # ifft of reconstruction
    #da_fft_nmf = xr.zeros_like(da_fft, dtype=float)
    #da_fft_nmf[:, :, :na_fft_nmf.shape[-1]] = na_fft_nmf
    #da_ifft_nmf = get_ifft_of_DataArray(data=da_fft_nmf, start_time=ds.time.values[0])


