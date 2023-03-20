import os
from collections import namedtuple
from pathlib import Path, PureWindowsPath
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

class RBFPAFullQuench_Snapshot(Dataset):
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
                      reference_index: Optional[list] = None,
                      metadata: Optional[pd.DataFrame] = None) -> tuple:
        """
        load and process data and simulation
        :param mp3_fpa_df_subset: DataFrame with mp3 data of quenched magnets
        :param data_path: path to hdf5 data
        :param simulation_path: path to hdf5 simulations
        :param reference_index: time index of data, if none the index of the first data signal is taken as reference
        :return: list of dataframes with data and simulation
        """
        fpa_identifier = mp3_fpa_df_subset.fpa_identifier.values[0]
        timestamp_fgc = int(fpa_identifier.split("_")[-1])

        # load data
        data_dir = data_path / (fpa_identifier + ".hdf5")
        data = load_from_hdf_with_regex(file_path=data_dir, regex_list=['VoltageNQPS.*U_DIODE'])
        df_data = u_diode_data_to_df(data, len_data=len(data[0]), sort_circuit=fpa_identifier.split("_")[1])
        magnet_list = df_data.columns.values

        # sometimes only noise is stored, std must be > 3, mean must be in window -1, -10
        mean_range = [-1.5, -10]
        min_std = 1
        drop_columns = df_data.columns[(df_data.std() < min_std) |
                                           (df_data.mean() > mean_range[0]) |
                                           (df_data.mean() < mean_range[1])]
        df_data_noq = df_data.drop(drop_columns, axis=1)

        # cut out time frame to analyze
        if timestamp_fgc < 1526582397220000000: # data before 2018 has smaller plateau
            # align with energy extraction timestamp
            ee_margins = [-0.25, 0.55]
            t_first_extraction = 0.34
            df_data_aligned, offset_ts = align_u_diode_data(df_data=df_data_noq.copy(),
                                                 method="timestamp_EE",
                                                 t_first_extraction=t_first_extraction,
                                                 ee_margins=ee_margins,
                                                 metadata=metadata)
        else:
            # align with energy extraction timestamp
            ee_margins = [-0.25, 0.45]
            t_first_extraction = 0.1
            df_data_aligned, offset_ts = align_u_diode_data(df_data=df_data_noq.copy(),
                                                 method="timestamp_EE",
                                                 t_first_extraction=t_first_extraction,
                                                 ee_margins=ee_margins,
                                                 metadata=metadata)


        # add quenched magnets again for continuity
        dropped_columns_data = magnet_list[~np.isin(magnet_list, df_data_aligned.columns)]
        df_data_aligned[dropped_columns_data] = np.nan
        # bring into electrical order again
        df_data_cut = df_data_aligned[magnet_list]

        return df_data_cut, None

    def generate_dataset(self, fpa_identifiers: list):
        """
        generates xarray.DataArray for each fpa identifier. Dataset includes u diode pm data and simulation
        :param fpa_identifiers: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        """
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        # load and process mp3 excel
        mp3_fpa_df = pd.read_csv(self.context_path)
        df_metadata = pd.read_csv(self.metadata_path)


        reference_index = None
        for fpa_identifier in fpa_identifiers:
            # if dataset already exists
            if not os.path.isfile(self.plot_dataset_path / f"{fpa_identifier}_snapshots.png"):
                print(fpa_identifier)
                mp3_fpa_df_subset = mp3_fpa_df[mp3_fpa_df.fpa_identifier == fpa_identifier]
                df_metadata_subset = df_metadata[df_metadata["Circuit"] == fpa_identifier.split("_")[1]]\
                    .sort_values("#Electric_circuit")

                df_data, _ = self.generate_data(mp3_fpa_df_subset,
                                                self.data_path,
                                                self.simulation_path,
                                                reference_index,
                                                df_metadata_subset)
                if reference_index is None:
                    reference_index = df_data.index

                # add data and simulation
                xr_array = data_to_xarray(df_data=df_data,
                                          df_simulation=None,
                                          df_el_position_features=None,
                                          df_event_features=None,
                                          event_identifier=fpa_identifier)
                xr_array.to_netcdf(self.dataset_path / f"{fpa_identifier}.nc")


                if self.plot_dataset_path:
                    self.plot_dataset_path.mkdir(parents=True, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(15, 10))
                    df_data.plot(legend=False, ax=ax)
                    ax.set_title(f"{mp3_fpa_df_subset['date'].values[0]}\n{fpa_identifier}")
                    ax.set_ylabel("Voltage / V")
                    ax.set_xlabel("Time / s")
                    plt.tight_layout()
                    plt.grid()
                    plt.savefig(self.plot_dataset_path / f"{fpa_identifier}_snapshots.png")
                    plt.close(fig)
