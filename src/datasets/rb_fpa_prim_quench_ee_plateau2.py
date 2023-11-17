import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.dataset import Dataset
from src.utils.dataset_utils import align_u_diode_data, drop_quenched_magnets, u_diode_data_to_df, data_to_xarray, \
    add_exp_trend_coeff_to_xr, get_df_time_window
from src.utils.frequency_utils import exponential_func
from src.utils.hdf_tools import load_from_hdf_with_regex


class RBFPAPrimQuenchEEPlateau2(Dataset):
    """
    Subclass of Dataset to specify dataset selection. This dataset contains downloaded and simulated u diode data
    during a primary quench.
    Paths must be given to regenerate dataset.
    """

    @staticmethod
    def process_fpa_event(fpa_df: pd.DataFrame, data_path: Path, metadata_path: Path) -> pd.DataFrame:
        """
        load and process data
        :param fpa_df: DataFrame with mp3 data of quenched magnets
        :param data_path: path to hdf5 data
        :param metadata_path: path to file "RB_position_context.csv"
        :return: list of dataframes with data
        """
        fpa_identifier = fpa_df.fpa_identifier.values[0]
        timestamp_fgc = int(fpa_identifier.split("_")[-1])

        if 'Position' in fpa_df.columns:  # check if and where quench occurred
            all_quenched_magnets = fpa_df.Position.values
            quench_times = fpa_df["Delta_t(iQPS-PIC)"].values / 1e3

        # load data
        data_dir = data_path / (fpa_identifier + ".hdf5")
        data = load_from_hdf_with_regex(file_path=data_dir, regex_list=['VoltageNQPS.*U_DIODE'])
        df_data = u_diode_data_to_df(data,
                                     len_data=len(data[0]),
                                     rb_position_context_path=metadata_path,
                                     sort_circuit=fpa_identifier.split("_")[1])
        magnet_list = df_data.columns.values

        if 'Position' in fpa_df.columns: # check if and where quench occurred
            # drop quenched magnet
            max_time = df_data.index.max()
            df_data_noq = drop_quenched_magnets(df_data, all_quenched_magnets, quench_times, max_time)
        else:
            df_data_noq = df_data

        # sometimes only noise is stored, std must be > 3, mean must be in window -1, -10
        mean_range = [-1.5, -10]
        min_std = 1
        drop_columns = df_data_noq.columns[(df_data_noq.std() < min_std) |
                                           (df_data_noq.mean() > mean_range[0]) |
                                           (df_data_noq.mean() < mean_range[1])]
        df_data_noq = df_data_noq.drop(drop_columns, axis=1)

        # cut out time frame to analyze
        if timestamp_fgc < 1526582397220000000:  # data before 2018 has smaller plateau
            # align with energy extraction timestamp
            ee_margins = [-0.25, 0.55]
            t_first_extraction = 0.2
            df_data_aligned, _ = align_u_diode_data(df_data=df_data_noq.copy(),
                                                    method="timestamp_EE",
                                                    t_first_extraction=t_first_extraction,
                                                    ee_margins=ee_margins)

            time_frame = [0.6, 0.95]
            n_samples = 400
            df_data_cut = get_df_time_window(df=df_data_aligned,
                                             timestamp=t_first_extraction,
                                             time_frame=time_frame,
                                             n_samples=n_samples)

        else:
            # align with energy extraction timestamp
            ee_margins = [-0.25, 0.45]
            t_first_extraction = 0.2
            df_data_aligned, _ = align_u_diode_data(df_data=df_data_noq.copy(),
                                                    method="timestamp_EE",
                                                    t_first_extraction=t_first_extraction,
                                                    ee_margins=ee_margins)

            time_frame = [0.6, 0.95]
            n_samples = 800
            df_data_cut = get_df_time_window(df=df_data_aligned,
                                             timestamp=t_first_extraction,
                                             time_frame=time_frame,
                                             n_samples=n_samples)

        # add quenched magnets again for continuity
        dropped_columns_data = magnet_list[~np.isin(magnet_list, df_data_cut.columns)]
        df_data_cut[dropped_columns_data] = np.nan
        # bring into electrical order again
        df_data_cut = df_data_cut[magnet_list]

        return df_data_cut
