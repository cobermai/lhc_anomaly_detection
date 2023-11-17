import os
from pathlib import Path
from typing import Optional, List

import pandas as pd
from matplotlib import pyplot as plt

from src.dataset import Dataset
from src.utils.dataset_utils import u_diode_data_to_df, data_to_xarray, get_sec_quench_frame_exclude_quench, \
    add_exp_trend_coeff_to_xr
from src.utils.hdf_tools import load_from_hdf_with_regex


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
                 acquisition_summary_path: Optional[Path] = None,
                 plot_dataset_path: Optional[Path] = None):
        super().__init__(dataset_path,
                         context_path,
                         metadata_path,
                         data_path,
                         acquisition_summary_path,
                         plot_dataset_path)

    @staticmethod
    def process_fpa_event(fpa_df: pd.DataFrame, data_path: Path, metadata_path: Path) -> List[pd.DataFrame]:
        """
        load and process data
        :param fpa_df: DataFrame with mp3 data of quenched magnets
        :param data_path: path to hdf5 data
        :param metadata_path: path to file "RB_position_context.csv"
        :return: list of dataframes with data
        """
        fpa_identifier = fpa_df.fpa_identifier.values[0]
        quench_times = fpa_df["Delta_t(iQPS-PIC)"].values / 1e3

        # load data nxcals
        data_dir = data_path / (fpa_identifier + ".hdf5")
        data_nxcals = load_from_hdf_with_regex(file_path=data_dir, regex_list=["VoltageNXCALS.*U_DIODE"])
        df_data_nxcals = u_diode_data_to_df(data_nxcals,
                                            rb_position_context_path=metadata_path,
                                            len_data=len(data_nxcals[0]))

        # drop quenched magnet
        time_frame_after_quench = [0.2, 2]
        sec_quenches = get_sec_quench_frame_exclude_quench(df_data=df_data_nxcals,
                                                           all_quenched_magnets=fpa_df.Position.values,
                                                           quench_times=quench_times,
                                                           time_frame=time_frame_after_quench)

        return sec_quenches

    def generate_dataset(self, fpa_identifiers: list, add_exp_trend_coeff: Optional[bool] = False):
        """
        generates xarray.DataArray for each fpa identifier. Dataset includes u diode pm data
        :param fpa_identifiers: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        """
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        # load and process mp3 excel
        mp3_fpa_df = pd.read_csv(self.context_path)
        mp3_fpa_df = mp3_fpa_df[mp3_fpa_df.fpa_identifier.isin(fpa_identifiers)]

        for fpa_identifier in fpa_identifiers:
            # if dataset already exists
            if not os.path.isfile(self.plot_dataset_path / f"{fpa_identifier}.png"):
                print(fpa_identifier)
                mp3_fpa_df_subset = mp3_fpa_df[mp3_fpa_df.fpa_identifier == fpa_identifier]

                sec_quenches = self.process_fpa_event(mp3_fpa_df_subset,
                                                      self.data_path,
                                                      self.metadata_path)

                for i, df_data in enumerate(sec_quenches):
                    # add data
                    xr_array = data_to_xarray(df_data=df_data, event_identifier=fpa_identifier)
                    if add_exp_trend_coeff:
                        xr_array = add_exp_trend_coeff_to_xr(ds=xr_array, data_var="data")
                        fit_coefficients = xr_array['polyfit_coefficients'].values
                    else:
                        fit_coefficients = None
                    xr_array.to_netcdf(self.dataset_path / f"{fpa_identifier}_{i}.nc")

                    if self.plot_dataset_path:
                        self.plot_dataset_path.mkdir(parents=True, exist_ok=True)
                        self.plot_data(df_data=df_data,
                                       plot_path=self.plot_dataset_path / f"{fpa_identifier}.png",
                                       fit_coefficients=fit_coefficients)


