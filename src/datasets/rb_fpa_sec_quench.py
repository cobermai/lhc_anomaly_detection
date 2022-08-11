import os
from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from src.dataset import Dataset
from src.modeling.sec_quench import get_sec_quench_frame_exclude_quench
from src.utils.dataset_utils import align_u_diode_data, drop_quenched_magnets, u_diode_simulation_to_df, \
    u_diode_data_to_df, data_to_xarray, get_u_diode_data_alignment_timestamps
from src.utils.hdf_tools import load_from_hdf_with_regex
from src.utils.utils import interp

data = namedtuple("data", ["X", "y", "idx"])


class RBFPASecQuench(Dataset):
    """
    Subclass of Dataset to specify dataset selection. This dataset contains downloaded and simulated u diode data
    during a secondary quench.
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
        # only events > 2021 (1608530800000000000), string to unix timestamp with lhcsmapi.Time.to_unix_timestamp()
        lower_limit = 1388530800000000000
        mp3_fpa_df_period = mp3_fpa_df_unique[mp3_fpa_df_unique['timestamp_fgc'] >= lower_limit].reset_index(drop=True)

        if acquisition_summary_path:
            df_acquisition = pd.read_excel(acquisition_summary_path)
            df_to_analyze = mp3_fpa_df_period.merge(df_acquisition,
                                                    left_on=['Circuit Name', 'timestamp_fgc'],
                                                    right_on=['Circuit Name', 'timestamp_fgc'],
                                                    how="left")
            mp3_fpa_df_period = df_to_analyze[(df_to_analyze["VoltageNXCALS.*U_DIODE"] == 1) &
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

        for k, fpa_identifier in enumerate(fpa_identifiers):
            # if dataset already exists
            if not os.path.isfile(dataset_path / f"{fpa_identifier}.nc"):

                circuit_name = fpa_identifier.split("_")[1]
                timestamp_fgc = int(fpa_identifier.split("_")[2])

                # get df with all quenches from event
                mp3_fpa_df_subset = mp3_fpa_df[(mp3_fpa_df.timestamp_fgc == timestamp_fgc) &
                                               (mp3_fpa_df['Circuit Name'] == circuit_name)]
                all_quenched_magnets = mp3_fpa_df_subset.Position.values
                quench_times = mp3_fpa_df_subset["Delta_t(iQPS-PIC)"].values / 1e3

                # load data
                data_dir = data_path / (fpa_identifier + ".hdf5")
                data = load_from_hdf_with_regex(file_path=data_dir, regex_list=["VoltageNXCALS.*U_DIODE"])
                df_data = u_diode_data_to_df(data, len_data=len(data[0]))

                # load simulation
                simulation_dir = simulation_path / (fpa_identifier + ".hdf")
                data_sim = load_from_hdf_with_regex(file_path=simulation_dir, regex_list=["0v_magf"])
                df_sim = u_diode_simulation_to_df(data_sim, circuit_name=circuit_name)

                # save magnet order for later usage
                magnet_list = df_sim.columns

                # define period to analyze, 0 is quench timestamp
                time_frame_after_quench = [0.3, 2]
                n_samples = 17  # if not none, n samples will be taken after time_frame[0] instead of timeframe[1]
                sec_quenches = get_sec_quench_frame_exclude_quench(df_data=df_data,
                                                                   all_quenched_magnets=mp3_fpa_df_subset.Position.values,
                                                                   quench_times=quench_times,
                                                                   time_frame=time_frame_after_quench,
                                                                   n_samples=n_samples)
                xr_array_list = []
                for sec_quench_number, df_quench_frame in enumerate(sec_quenches):
                    if not df_quench_frame.empty:
                        # drop quenched magnet
                        max_time = df_quench_frame.index.max()
                        df_sim_noq = drop_quenched_magnets(df_sim, all_quenched_magnets, quench_times, max_time)

                        # adjust simulation length to data
                        df_sim_noq_resampled = interp(df_sim_noq, df_quench_frame.index)


                        # add quenched magnets again for continuity
                        dropped_columns_data = magnet_list[~magnet_list.isin(df_quench_frame.columns)]
                        dropped_columns_simulation = magnet_list[~magnet_list.isin(df_sim_noq_resampled.columns)]
                        df_quench_frame[dropped_columns_data] = np.nan
                        df_sim_noq_resampled[dropped_columns_simulation] = np.nan
                        # bring into electrical order again
                        df_data_cut = df_quench_frame[magnet_list]
                        df_sim_noq_resampled = df_sim_noq_resampled[magnet_list]

                        if plot_dataset_path:
                            plot_dataset_path.mkdir(parents=True, exist_ok=True)
                            fig, ax = plt.subplots(2, 1, figsize=(15, 10))
                            df_data_cut.plot(legend=False, ax=ax[0])
                            ax[0].set_title("data")
                            df_sim_noq_resampled.plot(legend=False, ax=ax[1])
                            ax[1].set_title("simulation")
                            #plt.setp(ax, ylim=ax[0].get_ylim(), xlim=ax[0].get_xlim())
                            plt.tight_layout()
                            plt.savefig(plot_dataset_path / f"{fpa_identifier}_{sec_quench_number+2}.png")
                            plt.close(fig)

                        # add data and simulation
                        xr_array = data_to_xarray(df_data=df_data_cut, df_simulation=df_sim_noq_resampled,
                                                  event_identifier=fpa_identifier+f"_{sec_quench_number+2}")
                        xr_array_list.append(xr_array)

                xr_array_event = xr.concat(xr_array_list, dim="event")
                xr_array_event.to_netcdf(dataset_path / f"{fpa_identifier}.nc")
                print(f"{k}/{len(fpa_identifiers)}: {fpa_identifier}")
