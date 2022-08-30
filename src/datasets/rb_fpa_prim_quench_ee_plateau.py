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

class RBFPAPrimQuenchEEPlateau(Dataset):
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
    def load_dataset(fpa_identifiers: list, dataset_path: Path) -> xr.Dataset:
        """
        load DataArray from given list of fpa_identifiers
        :param fpa_identifiers: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        :param dataset_path: path to datasets
        :return: DataArray with dims (event, type, el_position, time)
        """
        dataset = []
        for fpa_identifier in fpa_identifiers:
            ds_dir = dataset_path / f"{fpa_identifier}.nc"
            if os.path.isfile(ds_dir):
                fpa_event_data = xr.load_dataset(ds_dir)
                dataset.append(fpa_event_data['data'].loc[{'time': slice(0, 1)}])

        dataset_full = xr.concat(dataset, dim="event")
        return dataset_full
