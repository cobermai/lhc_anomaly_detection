from typing import Callable

import numpy as np
import xarray as xr
from scipy.fft import fft, ifft


def polar_to_complex(amplitude: np.array, phase: np.array) -> np.array:
    """
    transforms polar coordinates into complex values
    :param amplitude: real valued amplitude of polar coordinates
    :param phase: real valued phase of polar coordinates
    :return: complex valued array
    """
    return amplitude * np.exp(1j * phase)


def complex_to_polar(x: np.array) -> np.array:
    """
    transforms complex values into polar coordinates
    :param x: complex valued array
    :return: real valued amplitude and phase
    """
    amplitude = np.abs(x)

    # threshold = np.max(amplitude)/100000
    # x[x>threshold] = 0
    phase = np.arctan2(np.imag(x), np.real(x))  # *180/np.pi
    return amplitude, phase


def get_hilbert_transform(data: np.array, f_window: Callable = np.ones):
    """
    calculates scaled hilbert transformation without ifft
    :param data: input data of shape (n_samples)
    :param f_window: window function for fft, default is no window
    :return: real valued amplitude and phase after hilbert transformation
    """
    N = len(data)
    data_windowed = data * f_window(N)

    f = fft(data_windowed)
    index = np.arange(N)

    complexf = 1j * f
    posF = index[1: int(np.floor(N / 2)) + np.mod(N, 2)]
    negF = index[int(np.ceil(N / 2)) + 1 - np.mod(N, 2):N]

    f[posF] = f[posF] - 1j * complexf[posF]
    f[negF] = f[negF] + 1j * complexf[negF]

    return f


def get_fft_of_DataArray(data: xr.DataArray,
                         f_lim: tuple = None,
                         f_window: Callable = np.ones) -> xr.DataArray:
    """
    calculates fft with hilbert transformation of DataArray, creates new coord frequency
    :param data: DataArray containing time-series data with coords (el_position, event, time)
    :param f_window: window function for fft, default is no window (np.ones)
    :param f_lim: min and max frequency, rest is filled with 0
    :return: DataArray containing frequency data with coords (el_position, event, frequency)
    """
    data_fft = xr.apply_ufunc(get_hilbert_transform,
                              data,
                              f_window,
                              input_core_dims=[['time'], []],
                              output_core_dims=[['frequency']],
                              exclude_dims={"time"},
                              vectorize=True)

    dt = data.time[1].values - data.time[0].values
    frequency_range = np.arange(len(data.time)) / dt / len(data.time)
    data_fft = data_fft.assign_coords(frequency=frequency_range)

    if f_lim:
        f_lim_bool = (data_fft.frequency < f_lim[0]) | (data_fft.frequency > f_lim[1])
        data_fft[:, :, f_lim_bool] = 0

    return data_fft


def get_ifft_of_DataArray(data: xr.DataArray, f_window: Callable = np.ones) -> xr.DataArray:
    """
    calculates ifft of DataArray, creates new coord time
    :param data: DataArray containing complex frequency data with coords (el_position, event, frequency)
    :return: DataArray containing frequency data with coords (el_position, event, frequency)
    """
    data_ifft = xr.apply_ufunc(ifft,
                               data,
                               input_core_dims=[['frequency']],
                               output_core_dims=[['time']],
                               exclude_dims={"frequency"},
                               vectorize=True)

    df = data.frequency[1].values - data.frequency[0].values
    time_range = np.arange(len(data.frequency)) / df / len(data.frequency)
    data_ifft = data_ifft.assign_coords(time=time_range)

    data_ifft_window = np.real(data_ifft) / f_window(len(data_ifft.time)) + 1j * np.imag(data_ifft)

    return data_ifft_window


def scale_fft_amplitude(data: xr.DataArray, f_window: Callable = np.ones):
    """
    calculates scaled amplitude of complex DataArray
    :param data: DataArray containing complex frequency data with coords (el_position, event, frequency)
    :param f_window: window function for fft, default is no window (np.ones)
    :return: DataArray containing real valued frequency amplitudes of data with coords (el_position, event, frequency)
    """
    N = len(data.frequency)
    window_gain = sum(f_window(N)) / N

    amplitude = xr.apply_ufunc(np.abs, data)
    data_amplitude = amplitude / N / window_gain

    return data_amplitude

