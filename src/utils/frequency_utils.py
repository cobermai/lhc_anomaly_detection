import numpy as np
from scipy.fft import fft, fftfreq
import xarray as xr


def get_fft_amplitude(x: np.array) -> np.array:
    """
    calculate amplitude of Fast Fourier Transformation of data x, removes nan and folds data with hanning window
    :param x: input data of shape (n_samples)
    :return: fft of data of shape (n_samples/2)
    """
    N = len(x)
    if np.isnan(x).all():
        y_FFT = fft(np.nan_to_num(x))
        y_FFT = np.zeros_like(y_FFT) * np.nan
    else:
        x = x[~np.isnan(x)]
        x = x * np.hanning(len(x))
        # x = pd.DataFrame(x).rolling(3).median().values.reshape(-1) #salt and peper noise
        y_FFT = fft(np.nan_to_num(x))
    return 2.0 / N * np.abs(y_FFT[0:N // 2])


def get_fft_of_DataArray(data: xr.DataArray, cutoff_frequency: int = None) -> xr.DataArray:
    """
    calculates fft of DataArray, creates new coord frequency
    :param data: DataArray containing time-series data with coords (el_position, event, time)
    :param cutoff_frequency: max frequency
    :return: DataArray containing frequency data with coords (el_position, event, frequency)
    """
    data_fft = xr.apply_ufunc(get_fft_amplitude,
                              data,
                              input_core_dims=[['time']],
                              output_core_dims=[['frequency']],
                              exclude_dims={"time"},
                              vectorize=True)

    dt = data[{'event': 0}].time[1].values - data[{'event': 0}].time[0].values
    frequency_range = fftfreq(len(data[0].time), dt)[:len(data[0].time) // 2]
    data_fft = data_fft.assign_coords(frequency=frequency_range)

    if cutoff_frequency:
        data_fft = data_fft.where(data_fft.frequency < cutoff_frequency, drop=True)

    return data_fft

