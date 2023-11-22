import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

from src.utils.dataset_utils import data_to_xarray, add_exp_trend_coeff_to_xr
from src.utils.frequency_utils import exponential_func


class Dataset(ABC):
    """
    abstract class which acts as a template to create datasets. Paths must be given to regenerate dataset.
    """

    def __init__(self,
                 dataset_path: Optional[Path] = None,
                 context_path: Optional[Path] = None,
                 metadata_path: Optional[Path] = None,
                 data_path: Optional[Path] = None,
                 acquisition_summary_path: Optional[Path] = None,
                 plot_dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path
        self.context_path = context_path
        self.metadata_path = metadata_path
        self.data_path = data_path
        self.acquisition_summary_path = acquisition_summary_path
        self.plot_dataset_path = plot_dataset_path

    def select_events(self) -> list:
        """
        generates list of events to load
        default is to load all files from manually filtered folder in acquisition_summary_path
        :return: list of strings which defines event, i.e. "<Circuit Family>_<Circuit Name>_<timestamp_fgc>"
        """
        return [filename.split('.png')[0] for filename in os.listdir(self.acquisition_summary_path)]

    @staticmethod
    @abstractmethod
    def process_fpa_event(fpa_df: pd.DataFrame, data_path: Path, metadata_path: Path) -> pd.DataFrame:
        """
        abstract method to generate dataset for fpa event
        :param fpa_df: DataFrame with mp3 data of this fpa event
        :param data_path: path to hdf5 data
        :param metadata_path: path to file "RB_position_context.csv"
        :return: dataframe with data
        """

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
        event_el_position = [rb_magnet_metadata_subset[rb_magnet_metadata_subset.Name == magnet]
                             ["#Electric_circuit"].values[0] - 1
                             for magnet in mp3_fpa_df_subset['Position'].values]
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
    def load_dataset(fpa_identifiers: list,
                     dataset_path: Path,
                     join: str = "inner",
                     drop_data_vars: Optional[list] = None,
                     location: Optional[dict] = None) -> xr.Dataset:
        """
        load DataArray from given list of fpa_identifiers
        :param fpa_identifiers: list of strings which defines event, i.e. "<Circuit Family>_<Circuit
        Name>_<timestamp_fgc>"
        :param dataset_path: path to datasets
        :param join: how to join the different datasets
        :param drop_data_vars: data_vars to load, default is all
        :param location: location (e.g. timespan: {'time':slice(0.1, 1.4)}) to load, default is all
        :return: DataArray with dims (event, type, el_position, time)
        """
        if drop_data_vars is None:
            drop_data_vars = []

        dataset = []
        for fpa_identifier in fpa_identifiers:
            ds_dir = dataset_path / f"{fpa_identifier}.nc"
            if os.path.isfile(ds_dir):
                fpa_event_data = xr.load_dataset(ds_dir)

                if location is None:
                    dataset.append(fpa_event_data.drop_vars(drop_data_vars))
                else:
                    dataset.append(fpa_event_data.drop_vars(drop_data_vars).loc[location])

        dataset_full = xr.concat(dataset, dim="event", join=join)
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
    def scale_dataset(dataset: xr.DataArray) -> xr.DataArray:
        """
        standard scales data by subtracting mean of event and dividing through overall standard deviation
        :param dataset: any concrete subclass of DatasetCreator to specify dataset selection
        :param axis: path to dataset
        """
        scale_dims = {'data': ("el_position", "time"),
                      'simulation': ("el_position", "time"),
                      'el_position_feature': ("el_position", "event"),
                      'event_feature': "event"}

        for data_var in list(dataset.keys()):
            dataset[data_var].attrs["scale_dims"] = scale_dims[data_var]

        for split in ['is_train', 'is_valid', 'is_test']:
            for data_var in list(dataset.keys()):
                split_events = dataset.loc[{'event': dataset.coords[split]}]['event'].values

                data = dataset[data_var].loc[split_events]
                data_mean = data.mean(dim=dataset[data_var].attrs["scale_dims"], keep_attrs=True)
                data_std = data.std(dim=dataset[data_var].attrs["scale_dims"], keep_attrs=True)

                dataset[data_var].loc[split_events] = (data - data_mean) / data_std
                dataset[data_var].attrs[f"{split}_scale_coef"] = (data_mean.values, data_std.values)
        return dataset

    @staticmethod
    def log_scale_data(X: np.array, vmin: float = 1e-5, vmax: float = 1e-2) -> np.array:
        """
        function min/max scales log10 data X with values vmin, vmax. Applies clipping. TODO: implement in scale_dataset
        :param X: data of shape (
        :param vmin: minimal value, set to 0
        :param vmax: minimal value, set to 1
        :return: scaled data
        """
        X_log = np.log10(X)
        vmin_log = np.log10(vmin)
        vmax_log = np.log10(vmax)
        X_std = (X_log - vmin_log) / (vmax_log - vmin_log)

        # clip
        X_std[X_std < 0] = 0
        X_std[X_std > 1] = 1
        return X_std

    @staticmethod
    def exp_scale_data(X: np.array, vmin: float = 1e-5, vmax: float = 1e-2):
        """
        function min/max scales data X with values vmin, vmax and exponent 10. Reverses log_scale_data
        :param X: data of shape (
        :param vmin: minimal value, set to 0
        :param vmax: minimal value, set to 1
        :return: scaled data
        """
        vdiff = np.log10(vmax) - np.log10(vmin)
        x_log = np.log10(vmin) + vdiff * X
        return 10 ** x_log

    @staticmethod
    def detrend_dim(da: xr.Dataset, dim: str = "time", data_var: str = "data", deg: int = 1) -> xr.Dataset:
        """
        subtract trend of data along given dimension
        :param da: DataArray to detrend
        :param dim: dimension to calculate the trend from , default ist time dimension
        :param data_var: data_var to detrend, default is to detrend data
        :param deg: degree of trend, default is a linear trend, i.e. deg=1
        :return: Dataset with subtracted trend
        """
        data = da.copy()
        if deg == -1:  # exp fit
            fit_coeff = da['polyfit_coefficients']
            p = fit_coeff.values.reshape(fit_coeff.shape[:-1] + (-1, 1))
            t = da["time"].values.reshape((1, 1, -1))
            fit = exponential_func(t, p[:, :, 0, :], p[:, :, 1, :], p[:, :, 2, :])
        else:
            fit_coeff = data[data_var].polyfit(dim=dim, deg=deg)
            fit = xr.polyval(data[dim], fit_coeff.polyfit_coefficients)

        data[data_var] = data[data_var] - fit
        return data.merge(fit_coeff, compat='override')

    @staticmethod
    def trend_dim(da: xr.Dataset, dim: str = "time", data_var: str = "data") -> xr.Dataset:
        """
        add trend of data along given dimension
        :param da: DataArray with data_var and polyfit_coefficients as data variables
        :param dim: dimension to calculate the trend from , default ist time dimension
        :param data_var: data_var to detrend, default is to detrend data
        :return: Dataset with added trend
        """
        if "tau" in da["polyfit_coefficient_names"]:
            fit_coeff = da['polyfit_coefficients']
            p = fit_coeff.values.reshape(fit_coeff.shape[:-1] + (-1, 1))
            t = da["time"].values.reshape((1, 1, -1))
            fit = exponential_func(t, p[:, :, 0, :], p[:, :, 1, :], p[:, :, 2, :])
        else:
            fit = xr.polyval(da[dim], da.polyfit_coefficients)
        da[data_var] = da[data_var] + fit

        return da

    @staticmethod
    def pad_data(da: xr.DataArray,
                 dim: str = "time",
                 interp_method: str = "nearest",
                 pad_len: Optional[int] = None) -> xr.DataArray:
        """
        pad signal at end and beginning
        :param da: DataArray to pad
        :param dim: dimension to calculate the trend from , default ist time dimension
        :param interp_method: interpolation method to fill pad with, default is to take last value of start/beginning
        :param pad_len: length of padding, efault is to pad with half of the data length on each side
        :return: DataArray with padded values
        """
        if pad_len is None:
            pad_len = int(len(da.time) / 2)

        # fill nan of input data
        da = da.interpolate_na(dim=dim, method=interp_method, fill_value="extrapolate")

        # define times when to pad
        dt = (da.time[1] - da.time[0]).values
        start_pad = np.linspace(da.time[0] - pad_len * dt, da.time[0] - dt, pad_len)
        end_pad = np.linspace(da.time[-1] + dt, da.time[-1] + pad_len * dt, pad_len)
        time_pad = np.hstack((start_pad, da.time, end_pad))

        # pad data extrapolate
        dataset_padded = da.interp(time=time_pad, method='nearest')  # fill with index with nan
        dataset_padded = dataset_padded.interpolate_na(dim=dim, method=interp_method, fill_value="extrapolate")

        return dataset_padded

    @staticmethod
    def lowpass_filter(data: np.array, cutoff: float, fs: float, order: int) -> np.array:
        """
        butterworth lowpass filters numpy array
        :param data: data
        :param cutoff: cuttoff frequency
        :param fs: sampling frequency
        :param order: order of butterworth filter, the higher, the more reponse time it needs
        :return: filtered numpy array
        """
        b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def lowpass_filter_DataArray(self, da: xr.DataArray, cutoff: float, order: int):
        """
        butterworth lowpass filters xarray dataarray
        :param da: data array
        :param cutoff: cuttoff frequency
        :param order: order of butterworth filter, the higher, the more reponse time it needs
        :return: filtered xarray dataarray
        """
        fs = 1 / (da.time.values[1] - da.time.values[0]),
        da_filtered = xr.apply_ufunc(self.lowpass_filter,
                                     da,
                                     cutoff,
                                     fs,
                                     order)
        return da_filtered

    @staticmethod
    def plot_data(df_data: pd.DataFrame, plot_path: Path, fit_coefficients: Optional[np.array] = None):
        """
        Plots data from a DataFrame and, if provided, an exponential fit trend.
        :param df_data: DataFrame containing the data to be plotted. Columns represent different data series.
        :param plot_path: Path object representing the file path where the plot will be saved.
        :param fit_coefficients: Optional NumPy array containing parameters for the exponential fit.
                        If provided, a second plot of the trend is generated.
        """

        bool_na = ~df_data.isna().all().values

        # Decide the number of subplots based on fit_coefficients
        subplot_count = 2 if fit_coefficients is not None else 1
        fig, axes = plt.subplots(subplot_count, 1, figsize=(15, 10))

        # Plotting the data
        ax_data = axes[0] if subplot_count > 1 else axes
        ax_data.plot(df_data.values.T[bool_na].T)
        ax_data.set_title(f"Data {len(df_data.dropna(axis=1, how='all').columns)}")
        ax_data.set_ylabel("Voltage / V")

        # Plotting the trend if fit_coefficients is provided
        if fit_coefficients is not None:
            trend = np.array([exponential_func(df_data.index, *p) for p in fit_coefficients])
            ax_trend = axes[1] if subplot_count > 1 else axes
            ax_trend.plot(df_data.index, trend[bool_na].T)
            ax_trend.set_title("Trend")
            ax_trend.set_ylabel("Voltage / V")

        plt.tight_layout()
        plt.grid()
        plt.savefig(plot_path)
        plt.close(fig)

    def generate_dataset(self, fpa_identifiers: list, add_exp_trend_coeff: Optional[bool] = False):
        """
        generates xarray.DataArray for each fpa identifier. Dataset includes u diode pm data
        :param fpa_identifiers: list of strings which defines event,
        i.e. "<Circuit Family>_<Circuit Name>_<timestamp_fgc>"
        :param add_exp_trend_coeff: calculate time-consuming exponential fit and add coefficients to dataset
        """
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        # load and process mp3 excel
        mp3_fpa_df = pd.read_csv(self.context_path)

        for i, fpa_identifier in enumerate(fpa_identifiers):
            # if dataset already exists
            if not os.path.isfile(self.plot_dataset_path / f"{fpa_identifier}.png"):
                print(f"{i}/{len(fpa_identifiers)}: {fpa_identifier}")
                mp3_fpa_df_subset = mp3_fpa_df[mp3_fpa_df.fpa_identifier == fpa_identifier]

                df_data = self.process_fpa_event(mp3_fpa_df_subset,
                                                 self.data_path,
                                                 self.metadata_path)

                # add data
                xr_array = data_to_xarray(df_data=df_data, event_identifier=fpa_identifier)
                if add_exp_trend_coeff:
                    xr_array = add_exp_trend_coeff_to_xr(ds=xr_array, data_var="data")
                    fit_coefficients = xr_array['polyfit_coefficients'].values
                else:
                    fit_coefficients = None
                xr_array.to_netcdf(self.dataset_path / f"{fpa_identifier}.nc")

                if self.plot_dataset_path:
                    self.plot_dataset_path.mkdir(parents=True, exist_ok=True)
                    self.plot_data(df_data=df_data,
                                   plot_path=self.plot_dataset_path / f"{fpa_identifier}.png",
                                   fit_coefficients=fit_coefficients)

def load_dataset(creator: "DatasetCreator",
                 dataset_path: Path,
                 context_path: Optional[Path] = None,
                 data_path: Optional[Path] = None,
                 metadata_path: Optional[Path] = None,
                 acquisition_summary_path: Optional[Path] = None,
                 plot_dataset_path: Optional[Path] = None,
                 generate_dataset: Optional[bool] = False,
                 add_exp_trend_coeff: Optional[bool] = False,
                 split_mask: Optional[np.array] = None,
                 scale_dataset: Optional[bool] = False,
                 drop_data_vars: Optional[list] = None,
                 location: Optional[dict] = None) -> xr.Dataset:
    """
    load dataset, dataset specific options can be changed in the dataset creator
    :param creator: any concrete subclass of DatasetCreator to specify dataset selection
    :param dataset_path: path where to store datasets
    :param context_path: path to mp3 Excel file, must be .csv, required if generate_dataset=True
    :param metadata_path: path to magnet metadata, must be .csv, required if generate_dataset=True
    :param data_path: path to hdf5 data, required if generate_dataset=True
    :param acquisition_summary_path: optional file path if data is manually analyzed, must be .xlsx
    :param plot_dataset_path: path to plot dataset events
    :param generate_dataset: flag to indicate whether dataset should be recreated
    :param add_exp_trend_coeff: calculate time-consuming exponential fit and add coefficients when generating dataset
    :param split_mask: array of shape (3,len(dataset["events]))
    with bool, specifying which events to put in training set
    :param scale_dataset: flag to indicate whether dataset should be scaled
    :param drop_data_vars: data_vars to load, default is all
    :param location: location to load, default is all
    :return: Dataset
    """

    ds = creator(dataset_path=dataset_path,
                 context_path=context_path,
                 metadata_path=metadata_path,
                 data_path=data_path,
                 acquisition_summary_path=acquisition_summary_path,
                 plot_dataset_path=plot_dataset_path)

    fpa_identifiers = ds.select_events()

    if generate_dataset:
        ds.generate_dataset(fpa_identifiers=fpa_identifiers,
                            add_exp_trend_coeff=add_exp_trend_coeff)

    dataset = ds.load_dataset(fpa_identifiers=fpa_identifiers,
                              dataset_path=dataset_path,
                              drop_data_vars=drop_data_vars,
                              location=location)

    dataset = ds.train_valid_test_split(dataset=dataset, split_mask=split_mask)
    if scale_dataset:
        dataset = ds.scale_dataset(dataset=dataset)

    return dataset
