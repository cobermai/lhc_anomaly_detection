import os
from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau
from src.datasets.rb_fpa_prim_quench_ee_plateau2 import RBFPAPrimQuenchEEPlateau2
from src.utils.frequency_utils import get_fft_of_DataArray, scale_fft_amplitude
from src.visualisation.fft_visualisation import plot_position_frequency_map_ee_plateau

if __name__ == "__main__":

    # define paths to read
    context_path = Path("../data/RB_TC_extract_2022_07_07_processed_filled.csv")
    metadata_path = Path("../data/RB_metadata.csv")
    dataset_path_1EE = Path("D:\\datasets\\20220707_prim_ee_plateau_dataset")
    dataset_path_2EE = Path("D:\\datasets\\20220707_RBFPAPrimQuenchEEPlateau2")

    # define paths to read + write
    output_path = Path(f"D:\\datasets\\FFT_analysis\\U_diode_EE_plateau_detrend")
    output_path.mkdir(parents=True, exist_ok=True)

    # load context and magnet metadata
    mp3_fpa_df = pd.read_csv(context_path, index_col=False)
    rb_magnet_metadata = pd.read_csv(metadata_path, index_col=False)

    # load desired fpa_identifiers
    all_fpa_identifiers = mp3_fpa_df.fpa_identifier.unique()
    dataset_creator_1EE = RBFPAPrimQuenchEEPlateau()
    dataset_creator_2EE = RBFPAPrimQuenchEEPlateau2()
    dataset_1EE = dataset_creator_1EE.load_dataset(fpa_identifiers=all_fpa_identifiers,
                                                   dataset_path=dataset_path_1EE,
                                                   drop_data_vars=['simulation', 'el_position_feature',
                                                                   'event_feature'])
    dataset_2EE = dataset_creator_2EE.load_dataset(fpa_identifiers=all_fpa_identifiers,
                                                   dataset_path=dataset_path_2EE,
                                                   drop_data_vars=['simulation', 'el_position_feature',
                                                                   'event_feature'])

    # Preprocessing
    f_window = np.hamming
    ds_detrend_1EE = dataset_creator_1EE.detrend_dim(dataset_1EE)
    da_win_1EE = ds_detrend_1EE.data * f_window(len(dataset_1EE.time))
    ds_detrend_2EE = dataset_creator_2EE.detrend_dim(dataset_2EE)
    da_win_2EE = ds_detrend_2EE.data * f_window(len(dataset_2EE.time))

    # calculate fft
    f_lim = (0, 534)
    da_fft_1EE = get_fft_of_DataArray(data=da_win_1EE, f_lim=f_lim)
    da_fft_amp_1EE = scale_fft_amplitude(data=da_fft_1EE, f_window=f_window)
    da_fft_amp_1EE = da_fft_amp_1EE[:, :, da_fft_amp_1EE.frequency < f_lim[1]]

    da_fft_2EE = get_fft_of_DataArray(data=da_win_2EE, f_lim=f_lim)
    da_fft_amp_2EE = scale_fft_amplitude(data=da_fft_2EE, f_window=f_window)
    da_fft_amp_2EE = da_fft_amp_2EE[:, :, da_fft_amp_2EE.frequency < f_lim[1]]

    # plot_position_frequency_map of ee_plateau
    circuit_imgs = {"el_pos_odd": plt.imread('../documentation/1_el_pos.png'),
                    "el_pos_even": plt.imread('../documentation/2_el_pos.png'),
                    "phys_pos_odd": plt.imread('../documentation/1_phys_pos.png'),
                    "phys_pos_even": plt.imread('../documentation/2_phys_pos.png')}
    fpa_identifiers = [fi for fi in all_fpa_identifiers if fi in dataset_1EE.event]  # only if data is available
    mp3_fpa_df['date'] = pd.to_datetime(mp3_fpa_df['Date (FGC)']).dt.strftime('%Y-%m-%d')
    for fpa_identifier in fpa_identifiers:
        date = mp3_fpa_df[mp3_fpa_df['fpa_identifier'] == fpa_identifier]['date'].values[0]
        filename = output_path / f"{fpa_identifier}_{date}.png"
        if not os.path.isfile(filename):
            plot_position_frequency_map_ee_plateau(fpa_identifier=fpa_identifier,
                                                   dataset_1EE=dataset_1EE,
                                                   dataset_1EE_fft=da_fft_amp_1EE,
                                                   dataset_2EE_fft=da_fft_amp_2EE,
                                                   dataset_2EE=dataset_2EE,
                                                   mp3_fpa_df=mp3_fpa_df,
                                                   rb_magnet_metadata=rb_magnet_metadata,
                                                   circuit_imgs=circuit_imgs,
                                                   filename=filename,
                                                   vmin=1e-5,
                                                   vmax=1)

            plt.close('all')
            gc.collect()