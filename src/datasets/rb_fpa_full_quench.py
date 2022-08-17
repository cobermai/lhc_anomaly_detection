import os
from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from src.dataset import Dataset
from src.modeling.sec_quench import get_df_time_window
from src.utils.dataset_utils import align_u_diode_data, drop_quenched_magnets, u_diode_simulation_to_df, \
    u_diode_data_to_df, data_to_xarray, get_u_diode_data_alignment_timestamps
from src.utils.hdf_tools import load_from_hdf_with_regex
from src.utils.utils import interp

data = namedtuple("data", ["X", "y", "idx"])


class RBFPAFullQuench(Dataset):
    """
    Subclass of Dataset to specify dataset selection. This dataset contains downloaded and simulated u diode data
    during a primary quench.
    """

    @staticmethod
    def select_events(context_path: Path, acquisition_summary_path: Optional[Path] = None) -> list:
        """
        generates list of events to load
        :param context_path: path to mp3 Excel file, must be .csv
        :param acquisition_summary_path: optional file path if data is manually analyzed, must be .xlsx
        :return: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        """
        # load mp3 fpa excel
        mp3_fpa_df = pd.read_csv(context_path)

        mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['timestamp_fgc', 'Circuit Name'])
        # only events > 2014 (1388530800000000000), string to unix timestamp with lhcsmapi.Time.to_unix_timestamp()
        # only events = 2021 (1608530800000000000), string to unix timestamp with lhcsmapi.Time.to_unix_timestamp()
        # test 1636530800000000000
        lower_limit = 1388530800000000000
        mp3_fpa_df_period = mp3_fpa_df_unique[mp3_fpa_df_unique['timestamp_fgc'] >= lower_limit].reset_index(drop=True)

        if acquisition_summary_path:
            df_acquisition = pd.read_excel(acquisition_summary_path)
            df_to_analyze = mp3_fpa_df_period.merge(df_acquisition,
                                                    left_on=['Circuit Name', 'timestamp_fgc'],
                                                    right_on=['Circuit Name', 'timestamp_fgc'],
                                                    how="left")
            mp3_fpa_df_period = df_to_analyze[(df_to_analyze['VoltageNQPS.*U_DIODE'] == 1) &
                                              (df_to_analyze["VoltageNXCALS.*U_DIODE"] == 1) &
                                              (df_to_analyze['simulation_data'] == 1)]

        fpa_identifiers = [f"{row['Circuit Family']}_{row['Circuit Name']}_{int(row['timestamp_fgc'])}" for i, row in
                           mp3_fpa_df_period.iterrows()]
        return fpa_identifiers

    @staticmethod
    def generate_dataset(fpa_identifiers: list,
                         dataset_path: Path,
                         context_path: Path,
                         data_path: Path,
                         simulation_path: Path,
                         plot_dataset_path: Optional[Path]):
        """
        generates xarray.DataArray for each fpa identifier. Dataset includes u diode pm data and simulation
        :param fpa_identifiers: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        :param dataset_path: path where to store datasets
        :param context_path: path to mp3 Excel file, must be .csv
        :param data_path: path to hdf5 data
        :param simulation_path: path to hdf5 simulations
        :param plot_dataset_path: optional path to plot dataset events
        """
        dataset_path.mkdir(parents=True, exist_ok=True)
        mp3_fpa_df = pd.read_csv(context_path)

        reference_index = [] #only works if ds is calculated in one go, TODO: load reference index from folder
        for fpa_identifier in fpa_identifiers:
            # if dataset already exists
            if not os.path.isfile(plot_dataset_path / f"{fpa_identifier}.png"): # os.path.isfile(dataset_path / f"{fpa_identifier}.nc"):
                print(fpa_identifier)
                circuit_name = fpa_identifier.split("_")[1]
                timestamp_fgc = int(fpa_identifier.split("_")[2])

                mp3_fpa_df_subset = mp3_fpa_df[(mp3_fpa_df.timestamp_fgc == timestamp_fgc) &
                                               (mp3_fpa_df['Circuit Name'] == circuit_name)]
                all_quenched_magnets = mp3_fpa_df_subset.Position.values
                quench_times = mp3_fpa_df_subset["Delta_t(iQPS-PIC)"].values / 1e3

                # load data pm
                data_dir = data_path / (fpa_identifier + ".hdf5")
                data_pm = load_from_hdf_with_regex(file_path=data_dir, regex_list=['VoltageNQPS.*U_DIODE'])
                df_data_pm = u_diode_data_to_df(data_pm, len_data=len(data_pm[0]))

                # load data nxcals
                data_nxcals = load_from_hdf_with_regex(file_path=data_dir, regex_list=["VoltageNXCALS.*U_DIODE"])
                df_data_nxcals = u_diode_data_to_df(data_nxcals, len_data=len(data_nxcals[0]))

                # load simulation
                simulation_dir = simulation_path / (fpa_identifier + ".hdf")
                data_sim = load_from_hdf_with_regex(file_path=simulation_dir, regex_list=["0v_magf"])
                df_sim = u_diode_simulation_to_df(data_sim, circuit_name=circuit_name)

                # save magnet order for later usage
                magnet_list = df_sim.columns

                # drop quenched magnet
                max_time = np.inf
                df_data_pm_noq = drop_quenched_magnets(df_data_pm, all_quenched_magnets, quench_times, max_time)
                df_data_nxcals_noq = drop_quenched_magnets(df_data_nxcals, all_quenched_magnets, quench_times, max_time)
                df_sim_noq = drop_quenched_magnets(df_sim, all_quenched_magnets, quench_times, max_time)

                # sometimes only noise is stored, mean must be in window -1, -10
                mean_range = [-1.5, -10]
                drop_columns = df_data_pm_noq.columns[~(mean_range[0] > df_data_pm_noq.mean()) |
                                                      (mean_range[1] > df_data_pm_noq.mean())]
                df_data_pm_noq = df_data_pm_noq.drop(drop_columns, axis=1)
                df_data_nxcals_noq = df_data_nxcals_noq.drop(drop_columns, axis=1)

                # align with simulation data
                # align with energy extraction timestamp
                ee_margins = [-0.25, 0.25]  # first ee must be within this interval
                # also integer from mp3 excel can be used as t_first_extraction
                t_first_extraction = get_u_diode_data_alignment_timestamps(df_sim_noq,
                                                                           ee_margins=ee_margins)
                df_data_pm_aligned = align_u_diode_data(df_data=df_data_pm_noq.copy(),
                                                        t_first_extraction=t_first_extraction,
                                                        ee_margins=ee_margins)

                # cut out time frame to analyze
                # All events need to have same index and len, reference index is taken from first event
                # Values from other events are interpolated to reference index
                if len(reference_index) == 0:
                    time_frame_pm = [-0.25, 1.4]
                    df_data_pm_cut = get_df_time_window(df=df_data_pm_aligned, timestamp=0, time_frame=time_frame_pm)
                    time_frame_nxcals = [time_frame_pm[1], np.inf]
                    df_data_nxcals_cut = get_df_time_window(df=df_data_nxcals_noq, timestamp=0,
                                                            time_frame=time_frame_nxcals)
                    df_data_cut = pd.concat([df_data_pm_cut, df_data_nxcals_cut.dropna()])
                    reference_index = df_data_cut.index
                else:
                    df_data = pd.concat([df_data_pm_aligned, df_data_nxcals_noq.dropna()])
                    df_data_cut = interp(df_data, reference_index)

                # adjust simulation length to data
                df_sim_noq_resampled = interp(df_sim_noq, reference_index)

                # add quenched magnets again for continuity
                dropped_columns_data = magnet_list[~magnet_list.isin(df_data_cut.columns)]
                dropped_columns_simulation = magnet_list[~magnet_list.isin(df_sim_noq_resampled.columns)]
                df_data_cut[dropped_columns_data] = np.nan
                df_sim_noq_resampled[dropped_columns_simulation] = np.nan
                # bring into electrical order again
                df_data_cut = df_data_cut[magnet_list]
                df_sim_noq_resampled = df_sim_noq_resampled[magnet_list]

                # add data and simulation
                xr_array = data_to_xarray(df_data=df_data_cut, df_simulation=df_sim_noq_resampled,
                                          event_identifier=fpa_identifier)
                xr_array.to_netcdf(dataset_path / f"{fpa_identifier}.nc")

                if plot_dataset_path:
                    plot_dataset_path.mkdir(parents=True, exist_ok=True)
                    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
                    ax[0].plot(df_data_cut.values)
                    ax[0].set_title("Data")
                    ax[0].set_ylabel("Voltage / V")
                    ax[1].plot(df_sim_noq_resampled.values)
                    ax[1].set_title("Simulation")
                    ax[1].set_xlabel("Samples")
                    ax[1].set_ylabel("Voltage / V")
                    plt.setp(ax, ylim=ax[0].get_ylim(), xlim=ax[0].get_xlim())
                    plt.tight_layout()
                    plt.savefig(plot_dataset_path / f"{fpa_identifier}.png")
                    plt.close(fig)
