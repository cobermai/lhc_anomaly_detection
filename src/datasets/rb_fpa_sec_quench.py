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
from src.modeling.sec_quench import get_df_time_window, get_sec_quench_frame_exclude_quench
from src.utils.dataset_utils import align_u_diode_data, drop_quenched_magnets, u_diode_simulation_to_df, \
    u_diode_data_to_df, data_to_xarray, get_u_diode_data_alignment_timestamps

class RBFPASecQuench(Dataset):
    """
    Subclass of Dataset to specify dataset selection. This dataset contains downloaded and simulated u diode data
    during a primary quench.
    Paths must be given to regenerate dataset.
    """
    def __init__(self,
                 dataset_path: Optional[Path] = None,
                 context_path: Optional[Path] = None,
                 metadata_path: Optional[Path] = None,
                 data_path: Optional[Path] = None,
                 simulation_path: Optional[Path] = None,
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

        # load data nxcals
        data_dir = data_path / (fpa_identifier + ".hdf5")
        data_nxcals = load_from_hdf_with_regex(file_path=data_dir, regex_list=["VoltageNXCALS.*U_DIODE"])
        df_data_nxcals = u_diode_data_to_df(data_nxcals, len_data=len(data_nxcals[0]))

        # load simulation
        simulation_dir = simulation_path / (fpa_identifier + ".hdf")
        data_sim = load_from_hdf_with_regex(file_path=simulation_dir, regex_list=["0v_magf"])
        df_sim = u_diode_simulation_to_df(data_sim, circuit_name=mp3_fpa_df_subset["Circuit Name"].values[0])

        # save magnet order for later usage
        magnet_list = df_sim.columns

        # drop quenched magnet
        time_frame_after_quench = [0.2, 2]
        sec_quenches = get_sec_quench_frame_exclude_quench(df_data=df_data_nxcals,
                                                           all_quenched_magnets=mp3_fpa_df_subset.Position.values,
                                                           quench_times=quench_times,
                                                           time_frame=time_frame_after_quench)

        return sec_quenches, None

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

                sec_quenches, _ = self.generate_data(mp3_fpa_df_subset,
                                                     self.data_path,
                                                     self.simulation_path,
                                                     reference_index)

                for i, df_data in enumerate(sec_quenches):
                # add data and simulation
                    xr_array = data_to_xarray(df_data=df_data,
                                              df_simulation=df_data,
                                              df_el_position_features=df_el_position_features,
                                              df_event_features=df_event_features,
                                              event_identifier=fpa_identifier)
                    xr_array.to_netcdf(self.dataset_path / f"{fpa_identifier}_{i}.nc")

                    if self.plot_dataset_path:
                        self.plot_dataset_path.mkdir(parents=True, exist_ok=True)
                        fig, ax = plt.subplots( figsize=(7, 5))
                        df_data.plot(ax=ax, legend=False)
                        ax.set_title("Data")
                        ax.set_ylabel("Voltage / V")
                        plt.tight_layout()
                        plt.savefig(self.plot_dataset_path / f"{fpa_identifier}_{i}.png")
                        plt.close(fig)


