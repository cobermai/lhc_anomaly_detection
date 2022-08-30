import os
import typing
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from src.visualisation.xarray_visualisation import plot_xarray_event


class Dataset(ABC):
    """
    abstract class which acts as a template to create datasets
    """

    def __init__(self,
                 dataset_path: Path,
                 context_path: Path,
                 metadata_path: Path,
                 data_path: Path,
                 simulation_path: Path,
                 acquisition_summary_path: Optional[Path] = None,
                 plot_dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path
        self.context_path = context_path
        self.metadata_path = metadata_path
        self.data_path = data_path
        self.simulation_path = simulation_path
        self.acquisition_summary_path = acquisition_summary_path
        self.plot_dataset_path = plot_dataset_path

    def select_events(self) -> list:
        """
        generates list of events to load
        :param mp3_fpa_df: DataFrame with mp3 fpa Excel file
        :param acquisition_summary_path: optional file path if data is manually analyzed, must be .xlsx
        :return: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        """
        mp3_fpa_df = pd.read_csv(self.context_path)
        mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=["timestamp_fgc", "Circuit Name"])
        # only events > 2014 (1388530800000000000), string to unix timestamp with lhcsmapi.Time.to_unix_timestamp()
        # only events = 2021 (1608530800000000000), string to unix timestamp with lhcsmapi.Time.to_unix_timestamp()
        # test 1636530800000000000
        lower_limit = 1388530800000000000 #1623530800000000000
        mp3_fpa_df_period = mp3_fpa_df_unique[mp3_fpa_df_unique["timestamp_fgc"] >= lower_limit].reset_index(drop=True)

        if self.acquisition_summary_path:
            df_acquisition = pd.read_excel(self.acquisition_summary_path)
            df_to_analyze = mp3_fpa_df_period.merge(df_acquisition,
                                                    left_on=["Circuit Name", "timestamp_fgc"],
                                                    right_on=["Circuit Name", "timestamp_fgc"],
                                                    how="left")
            mp3_fpa_df_period = df_to_analyze[(df_to_analyze["VoltageNQPS.*U_DIODE"] == 1) &
                                              (df_to_analyze["VoltageNXCALS.*U_DIODE"] == 1) &
                                              (df_to_analyze["simulation_data"] == 1)]

        fpa_identifiers = [f"{row['Circuit Family']}_{row['Circuit Name']}_{int(row['timestamp_fgc'])}"
                           for i, row in mp3_fpa_df_period.iterrows()]
        return fpa_identifiers

    @abstractmethod
    def generate_dataset(self, fpa_identifiers: list):
        """
        abstract method to generate dataset
        """

    @staticmethod
    def load_dataset(fpa_identifiers: list, dataset_path: Path, data_vars: str=[]) -> xr.Dataset:
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

    @staticmethod
    def train_valid_test_split(dataset: xr.DataArray,
                               split_mask: Optional[np.array] = None) -> xr.DataArray:
        """
        method to split data set into training, validation and test set
        :param dataset: DataArray with dataset
        :param split_mask: array of shape (3, len(dataset["events]))
        with bool, specifying which events to put in training set
        :return: dataset: DataArray with new coords (train, valid, test)
        """

        fgc_timestamps = np.array([int(x.split("_")[2]) for x in dataset["event"].values])
        if split_mask is None:  # default is train=valid: data from events>2021
            split_fgc_interval_train = [1608530800000000000, np.inf]
            split_mask_train = (split_fgc_interval_train[0] < fgc_timestamps) & \
                               (fgc_timestamps < split_fgc_interval_train[1])
            split_mask = np.array((split_mask_train, split_mask_train, ~split_mask_train))

        dataset.coords["is_train"] = ("event", split_mask[0])
        dataset.coords["is_valid"] = ("event", split_mask[1])
        dataset.coords["is_test"] = ("event", split_mask[2])
        return dataset

    @staticmethod
    def scale_dataset(dataset: xr.DataArray, axis: Optional[tuple] = (2, 3)) -> xr.DataArray:
        """
        standard scales data by subtracting mean of event and dividing through overall standard deviation
        :param dataset: any concrete subclass of DatasetCreator to specify dataset selection
        :param axis: path to dataset
        """
        # def scale_data_vars(scale_dims):
        dataset["data"].attrs["scale_dims"] = ("el_position", "time")
        dataset["simulation"].attrs["scale_dims"] = ("el_position", "time")
        dataset["el_position_feature"].attrs["scale_dims"] = ("el_position", "event")
        dataset["event_feature"].attrs["scale_dims"] = "event"

        for split in ['is_train', 'is_valid', 'is_test']:
            for data_var in list(dataset.keys()):
                split_events = dataset.loc[{'event': dataset.coords[split]}]['event'].values

                data = dataset[data_var].loc[split_events]
                data_mean = data.mean(dim=dataset[data_var].attrs["scale_dims"], keep_attrs=True)
                data_std = data.std(dim=dataset[data_var].attrs["scale_dims"], keep_attrs=True)

                dataset[data_var].loc[split_events] = (data - data_mean) / data_std
                dataset[data_var].attrs[f"{split}_scale_coef"] = (data_mean.values, data_std.values)
                #plot_xarray_event(dataset, data_var, idx=0)
        return dataset


def load_dataset(creator: "DatasetCreator",
                 dataset_path: Path,
                 context_path: Path,
                 metadata_path: Path,
                 data_path: Path,
                 simulation_path: Path,
                 acquisition_summary_path: Optional[Path] = None,
                 plot_dataset_path: Optional[Path] = None,
                 generate_dataset: Optional[bool] = False,
                 split_mask: Optional[np.array] = None,
                 scale_dataset: Optional[np.array] = None) -> xr.Dataset:
    """
    load dataset, dataset specific options can be changed in the dataset creator
    :param creator: any concrete subclass of DatasetCreator to specify dataset selection
    :param dataset_path: path to dataset
    :param dataset_path: path where to store datasets
    :param context_path: path to mp3 Excel file, must be .csv
    :param metadata_path: path to magnet metadata, must be .csv
    :param data_path: path to hdf5 data
    :param simulation_path: path to hdf5 simulations
    :param acquisition_summary_path: optional file path if data is manually analyzed, must be .xlsx
    :param plot_dataset_path: path to plot dataset events
    :param generate_dataset: flag to indicate whether dataset should be recreated
    :param split_mask: array of shape (3,len(dataset["events]))
    with bool, specifying which events to put in training set
    :return: Dataset with dims ('event', 'el_position', 'mag_feature_name', 'event_feature_name', 'time')
    """

    ds = creator(dataset_path=dataset_path,
                 context_path=context_path,
                 metadata_path=metadata_path,
                 data_path=data_path,
                 simulation_path=simulation_path,
                 acquisition_summary_path=acquisition_summary_path,
                 plot_dataset_path=plot_dataset_path)

    fpa_identifiers = ds.select_events()

    if generate_dataset:
        ds.generate_dataset(fpa_identifiers=fpa_identifiers)

    dataset = ds.load_dataset(fpa_identifiers=fpa_identifiers, dataset_path=dataset_path)
    dataset = ds.train_valid_test_split(dataset=dataset, split_mask=split_mask)
    if scale_dataset:
        dataset = ds.scale_dataset(dataset=dataset)

    return dataset
