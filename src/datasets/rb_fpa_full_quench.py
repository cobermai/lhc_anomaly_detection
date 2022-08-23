import os
from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from lhcsmapi.metadata.MappingMetadata import MappingMetadata

from src.dataset import Dataset
from src.utils.hdf_tools import load_from_hdf_with_regex
from src.utils.utils import interp
from lhcsmapi.metadata.MappingMetadata import MappingMetadata

data = namedtuple("data", ["X", "context", "idx"])

class RBFPAFullQuench(Dataset):
    """
    Subclass of Dataset to specify dataset selection. This dataset contains downloaded and simulated u diode data
    during a primary quench.
    """

    @staticmethod
    def select_events(mp3_fpa_df: pd.DataFrame, acquisition_summary_path: Optional[Path] = None) -> list:
        """
        generates list of events to load
        :param mp3_fpa_df: DataFrame with mp3 fpa Excel file
        :param acquisition_summary_path: optional file path if data is manually analyzed, must be .xlsx
        :return: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        """
        mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=["timestamp_fgc", "Circuit Name"])
        # only events > 2014 (1388530800000000000), string to unix timestamp with lhcsmapi.Time.to_unix_timestamp()
        # only events = 2021 (1608530800000000000), string to unix timestamp with lhcsmapi.Time.to_unix_timestamp()
        # test 1636530800000000000
        lower_limit = 1623530800000000000
        mp3_fpa_df_period = mp3_fpa_df_unique[mp3_fpa_df_unique["timestamp_fgc"] >= lower_limit].reset_index(drop=True)

        if acquisition_summary_path:
            df_acquisition = pd.read_excel(acquisition_summary_path)
            df_to_analyze = mp3_fpa_df_period.merge(df_acquisition,
                                                    left_on=["Circuit Name", "timestamp_fgc"],
                                                    right_on=["Circuit Name", "timestamp_fgc"],
                                                    how="left")
            mp3_fpa_df_period = df_to_analyze[(df_to_analyze["VoltageNQPS.*U_DIODE"] == 1) &
                                              (df_to_analyze["VoltageNXCALS.*U_DIODE"] == 1) &
                                              (df_to_analyze["simulation_data"] == 1)]

        fpa_identifiers = [f"{row['Circuit Family']}_{row['Circuit Name']}_{int(row['timestamp_fgc'])}" for i, row in mp3_fpa_df_period.iterrows()]
        return fpa_identifiers

    @staticmethod
    def select_context(mp3_fpa_df: pd.DataFrame, fpa_identifiers: list) -> list:
        """
        generates context data for modeling
        :param mp3_fpa_df: mp3 fpa Excel data
        :param fpa_identifiers: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        :return: DataFrame with context data for modeling
        """
        mp3_fpa_df = mp3_fpa_df[mp3_fpa_df.fpa_identifier.isin(fpa_identifiers)]
        circuits = np.array(['RB.A81',
                             'RB.A12',
                             'RB.A23',
                             'RB.A34',
                             'RB.A45',
                             'RB.A56',
                             'RB.A67',
                             'RB.A78'])

        mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['timestamp_fgc', 'Circuit Name'])
        df_metadata_full = MappingMetadata.read_layout_details("RB")

        context_len = 8 + 154 + 5
        context_data = np.zeros((len(fpa_identifiers), context_len))

        #scaling
        R_EE_odd = mp3_fpa_df["U_EE_max_ODD"].values / mp3_fpa_df["I_Q_M"].values  # float
        R_EE_even = mp3_fpa_df["U_EE_max_EVEN"] / mp3_fpa_df["I_Q_M"].values# float
        t_EE_odd = mp3_fpa_df["Delta_t(EE_odd-PIC)"].astype(int).values / 1000 # float
        t_EE_even = mp3_fpa_df["Delta_t(EE_even-PIC)"].astype(int).values / 1000

        context_scaling = {
            "I_end_2_from_data": [mp3_fpa_df["I_Q_M"].mean(), mp3_fpa_df["I_Q_M"].std()],  # float
            "R_EE_odd": [R_EE_odd.mean(), R_EE_odd.std()],
            "R_EE_even": [R_EE_even.mean(), R_EE_even.std()],
            "t_EE_odd": [t_EE_odd.mean(), t_EE_odd.std()],
            "t_EE_even":[t_EE_even.mean(), t_EE_even.std()],
            "I_Q_M": [mp3_fpa_df_unique["I_Q_M"].mean(), mp3_fpa_df_unique["I_Q_M"].std()]}


        for i, fpa_identifier in enumerate(fpa_identifiers):

            mp3_fpa_df_subset = mp3_fpa_df[mp3_fpa_df.fpa_identifier == fpa_identifier]
            circuit_name = mp3_fpa_df_subset["Circuit Name"].values[0]

            #scale signals
            I_end_2_from_data = mp3_fpa_df_subset["I_Q_M"].max()
            I_end_2_from_data = (I_end_2_from_data - context_scaling["I_end_2_from_data"][0]) / context_scaling["I_end_2_from_data"][1]

            R_EE_odd = float(mp3_fpa_df_subset["U_EE_max_ODD"].values[0]) / float(mp3_fpa_df_subset["I_Q_M"].max())
            R_EE_odd = (R_EE_odd - context_scaling["R_EE_odd"][0]) / context_scaling["R_EE_odd"][1]

            R_EE_even = float(mp3_fpa_df_subset["U_EE_max_EVEN"].values[0]) / float(mp3_fpa_df_subset["I_Q_M"].max())
            R_EE_even = (R_EE_even - context_scaling["R_EE_even"][0]) / context_scaling["R_EE_even"][1]

            t_EE_odd = float(mp3_fpa_df_subset["Delta_t(EE_odd-PIC)"].values[0]) / 1000
            t_EE_odd = (t_EE_odd - context_scaling["t_EE_odd"][0]) / context_scaling["t_EE_odd"][1]

            t_EE_even = float(mp3_fpa_df_subset["Delta_t(EE_even-PIC)"].values[0]) / 1000
            t_EE_even = (t_EE_even - context_scaling["t_EE_even"][0]) / context_scaling["t_EE_even"][1]

            df_metadata = df_metadata_full[df_metadata_full.Circuit == circuit_name].sort_values("#Electric_circuit")
            # t_shifts_mask = np.zeros(154)
            current_level_mask = np.zeros(154)
            quenched_magnets = "MB." + mp3_fpa_df_subset["Position"].values
            quenched_magnet_position = [df_metadata[df_metadata.Magnet == q]["#Electric_circuit"].values[0] - 1 for q in
                                        quenched_magnets]
            # t_shifts_mask[quenched_magnet_position] = list(mp3_fpa_df_subset["Delta_t(iQPS-PIC)"].dropna() / 1000)
            #I_Q_M = (mp3_fpa_df_subset['I_Q_M'].dropna() - context_scaling["I_Q_M"][0]) / context_scaling["I_Q_M"][1]
            current_level_mask[quenched_magnet_position] = 1 #list(I_Q_M)

            context_data_template = {
                "selected_circuit": (circuits == mp3_fpa_df_subset["Circuit Name"].values[0]) + 0,
                # , str -> list (n_circuits: int)
                "I_end_2_from_data": [I_end_2_from_data],  # float
                "R_EE_odd": [R_EE_odd],  # float
                "R_EE_even": [R_EE_even],  # float
                "t_EE_odd": [t_EE_odd],  # float
                "t_EE_even": [t_EE_even],  # float
                # list (n_quenches: float)) -> list (n_magnets: float):
                "current_level_quenches": current_level_mask}
                # list (n_quenches: float) -> list (n_magnets: float):
                #"t_shifts": t_shifts_mask}

            context_data[i] = np.array([x for v in context_data_template.values() for x in v], dtype="float32")
        return context_data

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
        from src.modeling.sec_quench import get_df_time_window
        from src.utils.dataset_utils import align_u_diode_data, drop_quenched_magnets, u_diode_simulation_to_df, \
            u_diode_data_to_df, data_to_xarray, get_u_diode_data_alignment_timestamps

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
                                               (mp3_fpa_df["Circuit Name"] == circuit_name)]
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
                    reference_index = df_data_cut.index[::2]
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

    @staticmethod
    def train_valid_test_split(dataset: xr.DataArray,
                               context_data: pd.DataFrame,
                               split_mask: Optional[np.array] = None) -> tuple:
        """
        method to split data set into training, validation and test set
        :param dataset: DataArray with dims (event, type, el_position, time)
        :param context_data: mp3 fpa Excel data with selected context for modeling
        :param split_mask: array of shape (3,len(dataset["events]))
        with bool, specifying which events to put in training set
        :return: tuple with (train, valid, test) xr.DataArrays of dims (event, type, el_position, time)
        """

        fgc_timestamps = np.array([int(x.split("_")[2]) for x in dataset["event"].values])
        if split_mask is None:  # default is train=valid: data from events>2021
            split_fgc_interval_train = [1608530800000000000, np.inf]
            split_mask_train = (split_fgc_interval_train[0] < fgc_timestamps) & \
                               (fgc_timestamps < split_fgc_interval_train[1])
            split_mask = np.array((split_mask_train, split_mask_train, ~split_mask_train))

        idx = np.arange(len(dataset))
        train = data(dataset[split_mask[0]], context_data[split_mask[0]], idx[split_mask[0]])
        valid = None #dataset[split_mask[1]]
        test = data(dataset[split_mask[2]], context_data[split_mask[2]], idx[split_mask[2]])
        return train, valid, test
