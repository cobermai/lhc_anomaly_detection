import typing
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

data = namedtuple("data", ["X", "y", "idx"])


class Dataset(ABC):
    """
    abstract class which acts as a template to create datasets
    """

    @staticmethod
    @abstractmethod
    def select_events(context_path: Path) -> list:
        """
        abstract method to select events for dataset
        """

    @staticmethod
    @abstractmethod
    def generate_dataset(fpa_identifiers: list, dataset_path: Path, context_path: Path, data_path: Path,
                         simulation_path: Path, plot_dataset_path: Optional[Path]):
        """
        abstract method to generate dataset
        """

    @staticmethod
    def load_dataset(fpa_identifiers: list, dataset_path: Path) -> xr.DataArray:
        """
        load DataArray from given list of fpa_identifiers
        :param fpa_identifiers: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        :param dataset_path: path to datasets
        :return: DataArray with loaded data
        """
        dataset = []
        for fpa_identifier in fpa_identifiers:
            fpa_event_data = xr.load_dataarray(dataset_path / f"{fpa_identifier}.nc")
            dataset.append(fpa_event_data)

        dataset_full = xr.concat(dataset, dim="event")
        return dataset_full

    @staticmethod
    @abstractmethod
    def train_valid_test_split(X_data_array: xr.DataArray,
                               splits: Optional[tuple] = None,
                               manual_split: list = None) -> tuple:
        """
        abstract method to split data set into training, validation and test set
        """

    @staticmethod
    @abstractmethod
    def scale_data(train: data, valid: data, test: data,
                   manual_scale: Optional[list] = None) -> tuple:
        """
        scales data for with sklearn standard scaler.
        Note that this function can be overwritten in the concrete dataset selection class.
        :param train: data for training of type named tuple
        :param valid: data for validation of type named tuple
        :param test: data for testing of type named tuple
        :param manual_scale: list that specifies groups which are scaled separately
        :return: train, valid, test: Tuple with data of type named tuple
        """


def load_dataset(creator: "DatasetCreator",
                 dataset_path: Path,
                 context_path: Path,
                 acquisition_summary_path: Optional[Path],
                 data_path: Path,
                 simulation_path: Path,
                 plot_dataset_path: Optional[Path],
                 generate_dateset: Optional[bool] = False) -> xr.DataArray:
    """
    load dataset, dataset specific options can be changed in the dataset creator
    :param creator: any concrete subclass of DatasetCreator to specify dataset selection
    :param dataset_path: path to dataset
    :param dataset_path: path where to store datasets
    :param context_path: path to mp3 Excel file, must be .csv
    :param acquisition_summary_path: optional file path if data is manually analyzed, must be .xlsx
    :param data_path: path to hdf5 data
    :param simulation_path: path to hdf5 simulations
    :param plot_dataset_path: path to plot dataset events
    :param generate_dateset: flag to indicate whether dataset should be recreated
    """
    fpa_identifiers = creator.select_events(context_path=context_path,
                                            acquisition_summary_path=acquisition_summary_path)

    if generate_dateset:
        creator.generate_dataset(fpa_identifiers=fpa_identifiers,
                                 dataset_path=dataset_path,
                                 context_path=context_path,
                                 data_path=data_path,
                                 simulation_path=simulation_path,
                                 plot_dataset_path=plot_dataset_path)

    dataset = creator.load_dataset(fpa_identifiers, dataset_path)
    return dataset
