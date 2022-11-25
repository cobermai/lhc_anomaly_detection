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

class RBFPASnapshotsEEPlateau2(Dataset):
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

    def select_events(self) -> list:
        """
        generates list of events to load
        :param mp3_fpa_df: DataFrame with mp3 fpa Excel file
        :param acquisition_summary_path: optional file path if data is manually analyzed, must be .xlsx
        :return: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        """
        context_data_path = Path(self.context_path)
        snapshot_context_df = pd.read_csv(context_data_path)
        fpa_identifiers = snapshot_context_df[snapshot_context_df["VoltageNQPS.*U_DIODE"] == 1].fpa_identifier.values
        return fpa_identifiers

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

        # load data
        data_dir = data_path / (fpa_identifier + ".hdf5")
        data = load_from_hdf_with_regex(file_path=data_dir, regex_list=['VoltageNQPS.*U_DIODE'])
        df_data = u_diode_data_to_df(data, len_data=len(data[0]))

        # load simulation
        simulation_dir = simulation_path / (fpa_identifier + ".hdf")
        data_sim = load_from_hdf_with_regex(file_path=simulation_dir, regex_list=["0v_magf"])
        df_sim = u_diode_simulation_to_df(data_sim, circuit_name=mp3_fpa_df_subset["Circuit"].values[0])

        # save magnet order for later usage
        magnet_list = df_sim.columns

        # sometimes only noise is stored, mean must be in window -1, -10
        mean_range = [-1.5, -10]
        df_data_noq = df_data.drop(
            df_data.columns[~(mean_range[0] > df_data.mean()) | (mean_range[1] > df_data.mean())],
            axis=1)

        # align with energy extraction timestamp
        ee_margins = [-0.25, 0.25]
        t_first_extraction = get_u_diode_data_alignment_timestamps(df_sim, ee_margins=ee_margins)
        df_data_aligned = align_u_diode_data(df_data=df_data_noq.copy(),
                                             t_first_extraction=t_first_extraction,
                                             ee_margins=ee_margins)

        # cut out time frame to analyze
        t_first_extraction = min(float(mp3_fpa_df_subset['Delta_t(EE_odd-PIC)'].values[0]) / 1000,
                                 float(mp3_fpa_df_subset['Delta_t(EE_even-PIC)'].values[0]) / 1000)
        time_frame = [0.65, 0.95]
        n_samples = 660
        df_data_cut = get_df_time_window(df=df_data_aligned,
                                         timestamp=t_first_extraction,
                                         time_frame=time_frame,
                                         n_samples=n_samples)

        # adjust simulation length to data
        df_sim_noq_resampled = interp(df_sim, df_data_cut.index)

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
        mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['timestamp_fgc', 'Circuit'])
        # load and process magnet metadata
        rb_magnet_metadata = pd.read_csv(self.metadata_path)
        rb_magnet_metadata = rb_magnet_metadata.sort_values("#Electric_circuit")

        reference_index = None
        for fpa_identifier in fpa_identifiers:
            # if dataset already exists
            if not os.path.isfile(self.plot_dataset_path / f"{fpa_identifier}.png"):
                print(fpa_identifier)
                mp3_fpa_df_subset = mp3_fpa_df[mp3_fpa_df.fpa_identifier == fpa_identifier]
                rb_magnet_metadata_subset = rb_magnet_metadata[rb_magnet_metadata.Circuit ==
                                                               mp3_fpa_df_subset['Circuit'].values[0]]


                df_data, df_sim = self.generate_data(mp3_fpa_df_subset,
                                                     self.data_path,
                                                     self.simulation_path,
                                                     reference_index)
                if reference_index is None:
                    reference_index = df_data.index

                # add data and simulation
                xr_array = data_to_xarray(df_data=df_data,
                                          df_simulation=df_sim,
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
