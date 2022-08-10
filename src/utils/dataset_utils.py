from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from lhcsmapi.metadata.MappingMetadata import MappingMetadata


def u_diode_data_to_df(data: list, len_data: int = 5500) -> pd.DataFrame:
    """
    puts list of df with u diode data in dataframe
    :param data: list of df with u diode data
    :param len_data: len to cut signals to if to long/short
    :return: dataframe with U_Diode_signals
    """
    data_columns = [df.columns.values[0].split("/")[1] for df in data]
    data_new = np.zeros((len(data_columns), len_data)) * np.nan
    time = np.zeros(len_data) * np.nan

    for i, df in enumerate(data):
        df = df[~df.index.duplicated(keep="first")]  # datapoints are sometimes logged twice
        data_new[i, :len(df.values)] = df.values[:len_data][:, 0]
    time[:len(df.index.values)] = df.index.values[:len_data]  # TODO: interpolate index, not take first one

    df_data_nxcals = pd.DataFrame(np.transpose(np.array(data_new)), columns=data_columns, index=time)
    return df_data_nxcals


def u_diode_simulation_to_df(data_sim: list, circuit_name: str) -> pd.DataFrame:
    """
    puts list of df with u diode simulation in dataframe, to have same format as data df
    :param data: list of df with simulation data
    :param circuit_name: len to cut signals to if to long/short
    :return: dataframe with u diode simulation
    """
    df_simulation_all = pd.concat(data_sim, axis=1)
    sorted_columns = [f"V(0v_magf{i})" for i in range(1, len(df_simulation_all.filter(regex="0v_magf").columns) + 1)]
    df_simulation = df_simulation_all[sorted_columns]

    # simulation numbering is sorted by #Electric_circuit
    df_metadata = MappingMetadata.read_layout_details("RB")
    df_metadata = df_metadata[df_metadata.Circuit == circuit_name].sort_values("#Electric_circuit")

    magnet_names = df_metadata.Magnet.apply(lambda x: x + ":U_DIODE_RB").values
    df_simulation.columns = magnet_names
    return df_simulation


def drop_quenched_magnets(df: pd.DataFrame, all_quenched_magnets: list, quench_times: list,
                          max_time: int) -> pd.DataFrame:
    """
    drops all magnets columns which quenched before max_time
    :param df: dataframe with data or simulation, columns are of format "MB.<magnet>:U_DIODE_RB"
    :param all_quenched_magnets: list of strings with names of quenched magnets
    :param quench_times: list of int with time of quench for each quenched magnet
    :param max_time: magnets which quenched before max_time will be dropped
    :return:
    """
    quench_within_frame = ["MB." + all_quenched_magnets[i] + ":U_DIODE_RB" for i, t in enumerate(quench_times) if
                           (t < max_time)]
    df = df.drop(columns=quench_within_frame)
    return df


def get_u_diode_data_alignment_timestamps(df: pd.DataFrame, ee_margins: list = [-0.25, 0.25],
                                          medfilt_size: int = 51) -> list:
    """
    gets timestamp of first energy extraction from data, used for data alignment.
    timestamp is first index, where meanfiltered data > threshold and meanfiltered derivative > delta threshold
    :param df: df with data, magnets in columns, time in index
    :param ee_margins: timeframe where first energy extraction takes place
    :param medfilt_size: size of meanfilter, should be odd
    :return: list of timestamps where first energy extraction is triggered
    """
    df_filt = df.rolling(medfilt_size, center=True).median()
    df_diff_filt = df_filt[(ee_margins[0] < df_filt.index) & (df_filt.index < ee_margins[1])].diff()
    alignment_timestamps = df_diff_filt.idxmin().to_list()
    return alignment_timestamps


def align_u_diode_data(df_data: pd.DataFrame, t_first_extraction: Union[float, int, list],
                       shift_th: int = 0) -> pd.DataFrame:
    """
    align u diode data, which is often shifted due to wrong triggers
    :param df_data: df with data, magnets are columns, time is index
    :param t_first_extraction: int with timestamp of first energy extraction for all magnets, if list: timestamp of each magnet
    :param shift_th: only shift if time difference is bigger than shift_th
    :return: df with aligned data
    """
    offset_ts = get_u_diode_data_alignment_timestamps(df_data)

    for i, c in enumerate(df_data.columns):
        # index, where time is closest to alignment ts
        if type(t_first_extraction) == list:
            zero_index = np.argmin(abs(df_data.index.values - t_first_extraction[i]))
        else:
            zero_index = np.argmin(abs(df_data.index.values - t_first_extraction))

        # index, where time is closest to offset_ts
        delta_index = np.argmin(abs(df_data.index.values - offset_ts[i]))

        shift_index = zero_index - delta_index
        if abs(shift_index) > shift_th:
            df_data[c] = df_data[c].shift(shift_index)
    return df_data.dropna()

def data_to_xarray(df_data: pd.DataFrame, df_simulation: pd.DataFrame, event_identifier: str) -> xr.DataArray:
    """
    puts data and simulation dataframe in one xarray DataArray
    :param df_data: dataframe with data
    :param df_simulation: dataframe with simulations. columns, index are similar to df_data
    :param event_identifier: name of event
    :return:
    """
    # https://rabernat.github.io/research_computing_2018/xarray.html#:~:text=1%3A%20Xarray%20Fundamentals-,Xarray%20data%20structures,potentially%20share%20the%20same%20coordinates
    n_magnets=154
    coords = {"event":[event_identifier],
              "type":["data", "simulation"],
              "el_position": np.arange(n_magnets),
              "time": df_data.index}

    ds = xr.DataArray(data=np.expand_dims(np.array((df_data.values.T, df_simulation.values.T)), axis=0), coords=coords)
    return ds
