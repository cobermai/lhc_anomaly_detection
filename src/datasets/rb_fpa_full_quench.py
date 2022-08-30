import os
from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from src.dataset import Dataset
from src.utils.hdf_tools import load_from_hdf_with_regex
from src.utils.utils import interp
from src.modeling.sec_quench import get_df_time_window
from src.utils.dataset_utils import align_u_diode_data, drop_quenched_magnets, u_diode_simulation_to_df, \
    u_diode_data_to_df, data_to_xarray, get_u_diode_data_alignment_timestamps

class RBFPAFullQuench(Dataset):
    """
    Subclass of Dataset to specify dataset selection. This dataset contains downloaded and simulated u diode data
    during a primary quench.
    """
    def __init__(self,
                 dataset_path: Path,
                 context_path: Path,
                 metadata_path: Path,
                 data_path: Path,
                 simulation_path: Path,
                 acquisition_summary_path: Optional[Path] = None,
                 plot_dataset_path: Optional[Path] = None):
        super().__init__(dataset_path,
                         context_path,
                         metadata_path,
                         data_path,
                         simulation_path,
                         acquisition_summary_path,
                         plot_dataset_path)
        self.el_position_features = ["R_1", "R_2", "RRR_1", "RRR_2"]
        self.event_el_position_features = ['I_Q_M', 'Delta_t(iQPS-PIC)']
        self.event_features = ["I_end_2_from_data",
                               "R_EE_odd",
                               "R_EE_even",
                               "t_EE_odd",
                               "t_EE_even",
                               "dI_dt_from_data"]

    @staticmethod
    def generate_el_position_features(mp3_fpa_df_subset: pd.DataFrame,
                                      rb_magnet_metadata_subset: pd.DataFrame,
                                      el_position_features: list,
                                      event_el_position_features: list) -> pd.DataFrame:
        """
        generates features dependent on el. position (e.g. magnet inductance) from magnet metadata and mp3_fpa excel
        :param mp3_fpa_df_subset: mp3 fpa Excel data with data from one event
        :param rb_magnet_metadata_subset: rb magnet metadata of circuit where event happened
        :param el_position_features: list of features dependent on electrical position
        :param event_el_position_features: list of features dependent on electrical position and event
        :return: DataFrame with el_position_features, index contains el position, columns contain el_position_features
        and event_el_position_features
        """
        # add el_position_features
        df_el_position_features = rb_magnet_metadata_subset[el_position_features].reset_index(drop=True)

        # add event_el_position_features
        df_el_position_features[event_el_position_features] = 0
        event_el_position = rb_magnet_metadata_subset[rb_magnet_metadata_subset.Name.isin(
            mp3_fpa_df_subset['Position'].values)]["#Electric_circuit"].values - 1
        df_el_position_features.loc[event_el_position, event_el_position_features] = \
            mp3_fpa_df_subset[event_el_position_features].values

        return df_el_position_features


    @staticmethod
    def generate_event_features(mp3_fpa_df_subset: pd.DataFrame, event_features: list) -> pd.DataFrame:
        """
        generates features dependent on event (e.g. current) from mp3 excel
        :param mp3_fpa_df_subset: mp3 fpa Excel data with data from one event
        :param event_features: list of features dependent on event
        :return: DataFrame with el_position_features, index is 0 (only one row), columns contain event_features
        """
        circuits = ['RB.A81',
                    'RB.A12',
                    'RB.A23',
                    'RB.A34',
                    'RB.A45',
                    'RB.A56',
                    'RB.A67',
                    'RB.A78']

        # add event features
        df_event_features = mp3_fpa_df_subset.reset_index(drop=True).loc[0, event_features].to_frame().T

        # add circuit as one hot encoded vector
        df_event_features.loc[0, circuits] = [int(mp3_fpa_df_subset['Circuit Name'].values[0] == c) for c in circuits]

        return df_event_features

    @staticmethod
    def generate_data(mp3_fpa_df_subset: pd.DataFrame,
                      data_path: Path,
                      simulation_path: Path,
                      reference_index: Optional[list] = None) -> tuple:
        """
        load and process data and simulation
        :param mp3_fpa_df_subset: DataFrame with mp3 data of quenched magnets
        :param data_path: path to hdf5 data
        :param simulation_path: path to hdf5 simulations
        :param reference_index: time index of data, if none the index of the first data signal is taken as reference
        :return: list of dataframes with data and simulation
        """
        fpa_identifier = mp3_fpa_df_subset.fpa_identifier.values[0]
        all_quenched_magnets = mp3_fpa_df_subset.Position.values
        quench_times = mp3_fpa_df_subset["Delta_t(iQPS-PIC)"].values / 1e3

        # load data pm
        data_dir = data_path / (fpa_identifier + ".hdf5")
        data_pm = load_from_hdf_with_regex(file_path=data_dir, regex_list=["VoltageNQPS.*U_DIODE"])
        df_data_pm = u_diode_data_to_df(data_pm, len_data=len(data_pm[0]))

        # load data nxcals
        data_nxcals = load_from_hdf_with_regex(file_path=data_dir, regex_list=["VoltageNXCALS.*U_DIODE"])
        df_data_nxcals = u_diode_data_to_df(data_nxcals, len_data=len(data_nxcals[0]))

        # load simulation
        simulation_dir = simulation_path / (fpa_identifier + ".hdf")
        data_sim = load_from_hdf_with_regex(file_path=simulation_dir, regex_list=["0v_magf"])
        df_sim = u_diode_simulation_to_df(data_sim, circuit_name=mp3_fpa_df_subset["Circuit Name"].values[0])

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
        if reference_index is None:
            time_frame_pm = [-0.25, 1.4]
            df_data_pm_cut = get_df_time_window(df=df_data_pm_aligned, timestamp=0, time_frame=time_frame_pm)
            time_frame_nxcals = [time_frame_pm[1], np.inf]
            df_data_nxcals_cut = get_df_time_window(df=df_data_nxcals_noq, timestamp=0,
                                                    time_frame=time_frame_nxcals)
            df_data_cut = pd.concat([df_data_pm_cut, df_data_nxcals_cut.dropna()])
            reference_index = df_data_cut.index  # [::2]
            df_data_cut = df_data_cut.loc[reference_index]
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

        return df_data_cut, df_sim_noq_resampled

    def generate_dataset(self, fpa_identifiers: list):
        """
        generates xarray.DataArray for each fpa identifier. Dataset includes u diode pm data and simulation
        :param fpa_identifiers: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        """
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        # load and process mp3 excel
        mp3_fpa_df = pd.read_csv(self.context_path)
        mp3_fpa_df = mp3_fpa_df[mp3_fpa_df.fpa_identifier.isin(fpa_identifiers)]
        mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['timestamp_fgc', 'Circuit Name'])
        # load and process magnet metadata
        rb_magnet_metadata = pd.read_csv(self.metadata_path)
        rb_magnet_metadata = rb_magnet_metadata.sort_values("#Electric_circuit")

        reference_index = None #only works if ds is calculated in one go, TODO: load reference index from folder
        for fpa_identifier in fpa_identifiers:
            # if dataset already exists
            if not os.path.isfile(self.plot_dataset_path / f"{fpa_identifier}.png"):
                print(fpa_identifier)
                mp3_fpa_df_subset = mp3_fpa_df[mp3_fpa_df.fpa_identifier == fpa_identifier]
                rb_magnet_metadata_subset = rb_magnet_metadata[rb_magnet_metadata.Circuit ==
                                                               mp3_fpa_df_subset['Circuit Name'].values[0]]

                df_el_position_features = self.generate_el_position_features(mp3_fpa_df_subset,
                                                                             rb_magnet_metadata_subset,
                                                                             self.el_position_features,
                                                                             self.event_el_position_features)

                df_event_features = self.generate_event_features(mp3_fpa_df_subset,
                                                                 self.event_features)

                df_data, df_sim = self.generate_data(mp3_fpa_df_subset,
                                                     self.data_path,
                                                     self.simulation_path,
                                                     reference_index)
                if reference_index is None:
                    reference_index = df_data.index

                # add data and simulation
                xr_array = data_to_xarray(df_data=df_data,
                                          df_simulation=df_sim,
                                          df_el_position_features=df_el_position_features,
                                          df_event_features=df_event_features,
                                          event_identifier=fpa_identifier)
                xr_array.to_netcdf(self.dataset_path / f"{fpa_identifier}.nc")




                if self.plot_dataset_path:
                    self.plot_dataset_path.mkdir(parents=True, exist_ok=True)
                    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
                    ax[0].plot(df_data.values)
                    ax[0].set_title("Data")
                    ax[0].set_ylabel("Voltage / V")
                    ax[1].plot(df_sim.values)
                    ax[1].set_title("Simulation")
                    ax[1].set_xlabel("Samples")
                    ax[1].set_ylabel("Voltage / V")
                    plt.setp(ax, ylim=ax[0].get_ylim(), xlim=ax[0].get_xlim())
                    plt.tight_layout()
                    plt.savefig(self.plot_dataset_path / f"{fpa_identifier}.png")
                    plt.close(fig)


