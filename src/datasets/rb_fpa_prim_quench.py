import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from src.dataset import Dataset
from src.utils.dataset_utils import align_u_diode_data, drop_quenched_magnets, u_diode_simulation_to_df, \
    u_diode_data_to_df, data_to_xarray
from src.utils.hdf_tools import load_from_hdf_with_regex
from src.utils.utils import interp

data = namedtuple("data", ["X", "y", "idx"])


class RBFPAPrimQuench(Dataset):
    """
    Subclass of Dataset to specify dataset selection.
    """

    @staticmethod
    def select_events(context_path: Path) -> list:
        """ generates list of events to load"""
        # load mp3 fpa excel
        mp3_fpa_df = pd.read_csv(context_path)

        mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['timestamp_fgc', 'Circuit Name'])
        # only events > 2014 (1388530800000000000), string to unix timestamp with lhcsmapi.Time.to_unix_timestamp()
        # only events = 2021 (1608530800000000000), string to unix timestamp with lhcsmapi.Time.to_unix_timestamp()
        lower_limit = 1608530800000000000
        mp3_fpa_df_period = mp3_fpa_df_unique[mp3_fpa_df_unique['timestamp_fgc'] >= lower_limit].reset_index(drop=True)

        df_to_analyze = mp3_fpa_df_period[mp3_fpa_df_period['VoltageNQPS.*U_DIODE'] == 1]

        fpa_identifiers = [f"{row['Circuit Family']}_{row['Circuit Name']}_{int(row['timestamp_fgc'])}" for i, row in
                           df_to_analyze.iterrows()]
        return ["RB_RB.A34_1618378572280000000"] #fpa_identifiers

    @staticmethod
    def generate_dataset(fpa_identifiers: list, dataset_path: Path, context_path: Path, data_path: Path,
                         simulation_path: Path):

        dataset_path.mkdir(parents=True, exist_ok=True)
        mp3_fpa_df = pd.read_csv(context_path)

        for fpa_identifier in fpa_identifiers:
            circuit_name = fpa_identifier.split("_")[1]
            timestamp_fgc = int(fpa_identifier.split("_")[2])

            mp3_fpa_df_subset = mp3_fpa_df[(mp3_fpa_df.timestamp_fgc == timestamp_fgc) &
                                           (mp3_fpa_df['Circuit Name'] == circuit_name)]
            all_quenched_magnets = mp3_fpa_df_subset.Position.values
            quench_times = mp3_fpa_df_subset["Delta_t(iQPS-PIC)"].values / 1e3

            # load simulation
            simulation_dir = simulation_path / (fpa_identifier + ".hdf")
            plot_dir = data_path.parent / f"20220707_udiode_aligned_plots_nosim/{fpa_identifier}.png"

            if os.path.isfile(simulation_dir) and not os.path.isfile(plot_dir):
                # load data
                data_dir = data_path / (fpa_identifier + ".hdf5")
                data = load_from_hdf_with_regex(file_path=data_dir, regex_list=['VoltageNQPS.*U_DIODE'])
                df_data = u_diode_data_to_df(data, len_data=len(data[0]))

                # load simulation
                simulation_dir = simulation_path / (fpa_identifier + ".hdf")
                data_sim = load_from_hdf_with_regex(file_path=simulation_dir, regex_list=["0v_magf"])
                df_sim = u_diode_simulation_to_df(data_sim, circuit_name=circuit_name)

                # save magnet order for later usage
                magnet_list = df_sim.columns

                # drop quenched magnet
                max_time = df_data.index.max()
                df_data_noq = drop_quenched_magnets(df_data, all_quenched_magnets, quench_times, max_time)
                df_sim_noq = drop_quenched_magnets(df_sim, all_quenched_magnets, quench_times, max_time)

                # sometimes only noise is stored, mean must be in window -1, -10
                df_data_noq = df_data_noq.drop(
                    df_data_noq.columns[~(-1 > df_data_noq.mean()) & (-10 < df_data_noq.mean())], axis=1)

                # align
                t_first_extraction = min(float(mp3_fpa_df_subset['Delta_t(EE_odd-PIC)'].values[0]) / 1000,
                                         float(mp3_fpa_df_subset['Delta_t(EE_even-PIC)'].values[0]) / 1000)
                df_data_aligned = align_u_diode_data(df_data_noq.copy(), t_first_extraction)

                # cut out time frame to analyze, [-0.25, 1] is 1336 samples
                time_frame = [-0.25, 1]
                df_data_cut = df_data_aligned[
                    (time_frame[0] <= df_data_aligned.index) & (time_frame[1] >= df_data_aligned.index)]

                # adjust simulation length to data
                df_sim_noq_resampled = interp(df_sim_noq, df_data_cut.index)

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
                print(fpa_identifier)

                plot_dir = Path("/mnt/d/datasets/20220707_processed_data")
                plot_dir.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(2, 1, figsize=(15, 10))
                df_data_cut.plot(legend=False, ax=ax[0])
                ax[0].set_title("data")
                ax[0].axvline(x=t_first_extraction)
                df_sim_noq_resampled.plot(legend=False, ax=ax[1])
                ax[1].set_title("simulation")
                plt.setp(ax, ylim=ax[0].get_ylim(), xlim=ax[0].get_xlim())
                plt.tight_layout()
                plt.savefig(plot_dir / f"{fpa_identifier}.png")
                plt.close(fig)
