from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit

from src.utils.frequency_utils import exponential_func


def u_diode_data_to_df(data: list, len_data: int = 5500, sort_circuit = None) -> pd.DataFrame:
    """
    puts list of df with u diode data in dataframe
    :param data: list of df with u diode data
    :param len_data: len to cut signals to if to long/short
    :param sort_with_metadata: sort df with metadata
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

    if sort_circuit is not None:
        # simulation numbering is sorted by #Electric_circuit
        meta_data_path = Path("../data/RB_metadata.csv")  # TODO: take path as argument
        df_metadata = pd.read_csv(meta_data_path, index_col=False)  # MappingMetadata.read_layout_details("RB")
        df_metadata = df_metadata[df_metadata.Circuit == sort_circuit].sort_values("#Electric_circuit")
        magnet_names = df_metadata.Magnet.apply(lambda x: x + ":U_DIODE_RB").values

        is_data_available = np.isin(magnet_names, df_data_nxcals.columns.values)
        df_data_nxcals.loc[:, magnet_names[~is_data_available]] = np.nan
        df_data_nxcals = df_data_nxcals[magnet_names]

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
    meta_data_path = Path("../data/RB_metadata.csv")  # TODO: take path as argument
    df_metadata = pd.read_csv(meta_data_path, index_col=False)  # MappingMetadata.read_layout_details("RB")
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
    df = df.drop(columns=quench_within_frame, errors='ignore')
    return df


def get_u_diode_data_alignment_timestamps(df: pd.DataFrame,
                                          ee_margins: list,
                                          medfilt_size: int = 51,
                                          metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    gets timestamp of first energy extraction from data, used for data alignment.
    timestamp is index, where derivative is max and voltage is smaller than voltage_threshold
    :param df: df with data, magnets in columns, time in index
    :param ee_margins: timeframe where first energy extraction takes place
    :param medfilt_size: size of meanfilter, should be odd
    :return: list of timestamps where first energy extraction is triggered
    """
    voltage_threshold = [-5, 0]
    df_filt = df.rolling(medfilt_size, center=True).median()
    df_diff_filt = df_filt[(ee_margins[0] < df_filt.index) &
                           (df_filt.index < ee_margins[1])].diff()
    alignment_timestamps = df_diff_filt[(df_filt > voltage_threshold[0]) & (df_filt < voltage_threshold[1])].idxmin().to_list()
    magnets = [c.split(":")[0] for c in df.columns.values]
    df_offset = pd.DataFrame(alignment_timestamps, index=magnets, columns=["offset"])


    if metadata is not None:
        metadata = metadata.set_index("Magnet")
        #metadata.loc[magnets, "offset_ts"] = alignment_timestamps
        #crate_offset = metadata.groupby("QPS Crate")["offset_ts"].median()
        #df_offset = metadata.apply(lambda x: crate_offset[x["QPS Crate"]], axis=1).to_frame(name ="offset")
        df_offset.loc[magnets, "QPS Crate"] = metadata.loc[magnets, "QPS Crate"].values

    return df_offset


def align_u_diode_data(df_data: pd.DataFrame,
                       method="timestamp_EE",
                       t_first_extraction: Optional[Union[float, int, list]] = None,
                       ee_margins: list = [-0.25, 0.4],
                       metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    align u diode data, which is often shifted due to wrong triggers
    :param df_data: df with data, magnets are columns, time is index
    :param t_first_extraction: int with timestamp of first energy extraction for all magnets,
    if list: timestamp of each magnet, signals will be aligned to this timestamp
    :param ee_margins: timeframe where first energy extraction takes place
    :return: df with aligned data
    """
    # timestamp of first energy extraction
    offset_ts = get_u_diode_data_alignment_timestamps(df_data, ee_margins, metadata=metadata)

    for i, c in enumerate(df_data.columns):
        # index, where time is closest to alignment ts
        magnet = c.split(":")[0]
        if method == "timestamp_EE_list":
            zero_index = np.argmin(abs(df_data.index.values - t_first_extraction[i]))
        elif method == "timestamp_EE":
            zero_index = np.argmin(abs(df_data.index.values - t_first_extraction))

        # index, where time is closest to offset_ts
        delta_index = np.argmin(abs(df_data.index.values - offset_ts.loc[magnet, "offset"]))

        shift_index = zero_index - delta_index
        df_data[c] = df_data[c].shift(shift_index)

        offset_ts.loc[magnet, "shift"] = shift_index

    if metadata is not None:
        return df_data, offset_ts
    else:
        return df_data


def data_to_xarray(df_data: pd.DataFrame,
                   event_identifier: str,
                   df_simulation: Optional[pd.DataFrame] = None,
                   df_el_position_features: Optional[pd.DataFrame] = None,
                   df_event_features: Optional[pd.DataFrame] = None) -> xr.Dataset:
    """
    puts data and simulation dataframe in one xarray DataArray
    https://rabernat.github.io/research_computing_2018/xarray.html#:~:text=1%3A%20Xarray%20Fundamentals-,Xarray%20data%20structures,potentially%20share%20the%20same%20coordinates
    :param df_data: DataFrame with data, index contains time, columns contain el position
    :param df_simulation: DataFrame with simulations, index contains time, columns contain el position
    :param df_el_position_features: DataFrame with features dependent on el. position (e.g. magnet inductance),
    index contains el position, columns features
    :param df_event_features: DataFrame with features dependent on event (e.g. current),
    index is 0 (only one row), columns contain event_features
    :param event_identifier: name of event
    :return: Dataset with dims ('event', 'el_position', 'mag_feature_name', 'event_feature_name', 'time')
    """
    n_magnets = 154

    ds = xr.Dataset(
        data_vars={'data': (('el_position', 'time'), df_data.astype("float32").values.T)},
        coords={'event': [event_identifier],
                'el_position': np.arange(n_magnets),
                'time': df_data.index})

    if df_simulation is not None:
        ds['simulation'] = (('el_position', 'time'), df_simulation.astype("float32").values.T)

    if df_el_position_features is not None:
        ds.coords['el_position_feature_name'] = df_el_position_features.columns
        ds['el_position_feature'] = (('el_position', 'el_position_feature_name'),
                                     df_el_position_features.astype("float32").values),
    if df_event_features is not None:
        ds.coords['event_feature_name'] = df_event_features.columns
        ds['event_feature'] = ('event_feature_name', df_event_features.astype("float32").values.reshape(-1))

    return ds

def add_exp_trend_coeff(ds, data_var):
    df_data = pd.DataFrame(ds[data_var].values.T, columns=ds.el_position, index=ds.time)
    p0 = [0, 0, np.nanmean(df_data)]
    exp_fit = df_data.fillna(0).apply(
        lambda x: curve_fit(f=exponential_func, xdata=x.index, ydata=x, p0=p0, maxfev=10000)[0], axis=0)
    ds.coords['polyfit_coefficient_names'] = ["amplitude", "tau", "offset"]
    ds['polyfit_coefficients'] = (('el_position', 'polyfit_coefficient_names'),
                                        exp_fit.values.T)
    return ds