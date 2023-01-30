from typing import Callable

import numpy as np
from scipy.fft import fft, fftfreq
import xarray as xr


def get_fft_amplitude(x: np.array, f_window: Callable = np.hanning) -> np.array:
    """
    calculate amplitude of Fast Fourier Transformation of data x, removes nan and folds data with hanning window
    :param x: input data of shape (n_samples)
    :param f_window: window function for fft, default is hanning window
    :return: fft of data of shape (n_samples/2)
    """
    N = len(x)
    if np.isnan(x).all():
        y_FFT = fft(np.nan_to_num(x))
        y_FFT = np.zeros_like(y_FFT) * np.nan
    else:
        x = x[~np.isnan(x)]
        window_gain = sum(f_window(1000)) / 1000
        x = x * f_window(len(x)) / window_gain
        y_FFT = fft(np.nan_to_num(x))

    # Hilbert Transform
    amplitude = np.abs(y_FFT[0:N // 2]) / N
    amplitude[1:] = amplitude[1:] * 2
    return amplitude


def get_fft_of_DataArray(data: xr.DataArray,
                         cutoff_frequency: int = None,
                         f_window: Callable = np.hanning) -> xr.DataArray:
    """
    calculates fft of DataArray, creates new coord frequency
    :param data: DataArray containing time-series data with coords (el_position, event, time)
    :param cutoff_frequency: max frequency
    :param f_window: window function for fft, default is hanning window
    :return: DataArray containing frequency data with coords (el_position, event, frequency)
    """
    data_fft = xr.apply_ufunc(get_fft_amplitude,
                              data,
                              f_window,
                              input_core_dims=[['time'], []],
                              output_core_dims=[['frequency']],
                              exclude_dims={"time"},
                              vectorize=True)

    dt = data[{'event': 0}].time[1].values - data[{'event': 0}].time[0].values
    frequency_range = fftfreq(len(data[0].time), dt)[:len(data[0].time) // 2]
    data_fft = data_fft.assign_coords(frequency=frequency_range)

    if cutoff_frequency:
        data_fft = data_fft.where(data_fft.frequency < cutoff_frequency, drop=True)

    return data_fft
