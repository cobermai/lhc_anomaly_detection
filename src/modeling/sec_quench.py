import numpy as np
import pandas as pd
from lhcsmapi.metadata.MappingMetadata import MappingMetadata
from scipy import signal
from scipy.signal import find_peaks, peak_prominences


def get_sec_quench_frame_exclude_quench(df_data: pd.DataFrame,
                                        all_quenched_magnets: list,
                                        quench_times: list,
                                        time_frame: list) -> list:
    """
    spliting dataframe with nxcals u diode data into list of dataframes around secondary quench
    :param df_data: dataframe with nxcals u diode data
    :param all_quenched_magnets: list of string with quenched magnets
    :param quench_times: list of ints with quench times
    :param time_frame: timeframe to analyze after quench
    :return: list of dataframes
    """
    sec_quenches = []
    for i, row in enumerate(all_quenched_magnets):
        delta = quench_times[i]
        quench_within_frame = ["MB." + all_quenched_magnets[i] + ":U_DIODE_RB" for i, t in enumerate(quench_times)
                               if (t < delta + time_frame[1])]
        df_subset = get_df_time_window(df=df_data, timestamp=delta, time_frame=time_frame)
        sec_quenches.append(df_subset.drop(columns=quench_within_frame))
    return sec_quenches[1:]


def get_df_time_window(df, timestamp, time_frame):
    """
    cuts time_frame window out of datafame
    :param df_data: dataframe with nxcals u diode data
    :param timestamp: integer with time center
    :param time_frame: list which defines area around timestamp
    :return: dataframes
    """
    # mask: defines time window
    mask = (df.index > timestamp - time_frame[0]) & (df.index < timestamp + time_frame[1])
    df_subset = df.loc[mask]
    return df_subset


def get_std_of_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    calculate standard deviation of dU
    :param df: df with magnets in columns header
    :return: new df with magnet in row
    """
    df_std = df.diff().std(axis=0).to_frame().rename(columns={0: "dstd"})
    df_std["Magnet"] = [index.split(":")[0] for index in df_std.index.values]
    return df_std


def sort_by_metadata(df: pd.DataFrame, circuit: str, quenched_magnet: str, by: str = "Position") -> pd.DataFrame:
    """
    :param df: df to sort
    :param circuit: str with circuit name
    :param quenched_magnet: str with quenched magnet name
    :param by: column name of metadata to sort by, metadata columns:
        'Circuit', 'Magnet', 'Position', 'B1_aperture', 'Diode_type', 'Correctors', 'EE place', '#Electric_EE',
        '#Electric_circuit', 'Cryostat', 'Cryostat2'
    :return: sorted dataframe, where index of quenched_magnet is 0
    """
    df["Circuit"] = circuit

    df_metadata = MappingMetadata.read_layout_details("RB")
    df_metadata_circuit = df_metadata[df_metadata.Circuit == circuit].reset_index(drop=True)
    df_std_meta = df_metadata_circuit.merge(df, left_on=['Circuit', 'Magnet'], right_on=['Circuit', 'Magnet'],
                                            how="left")
    df_std_meta = df_std_meta.sort_values(by=by).reset_index(drop=True)
    quenched_magnet_pos = df_std_meta[df_std_meta.Magnet == "MB." + quenched_magnet].index.values[0]
    df_std_meta["distance_to_quench"] = df_std_meta.index.values - quenched_magnet_pos

    return df_std_meta


def fit_window_to_data(window: np.array, len_data: int, data_center: int) -> np.array:
    """
    adjust window to data, cuts window if longer than data
    :param window: window
    :param len_data: len of data
    :param data_center: center of data, where window is added
    :return:
    """
    window_len = len(window)
    new_window = np.zeros(len_data)
    window_beginning = data_center - int(np.rint(window_len) / 2)
    window_end = data_center + int(np.rint(window_len) / 2)
    if not (window_len % 2) == 0:
        window_end += 1  # include center

    min_index = max(window_beginning, 0)
    max_index = min(window_end, len_data)

    min_window_index = min(0, window_beginning)
    max_window_index = window_len - max(0, window_end - len_data)

    new_window[min_index:max_index] = window[-min_window_index:max_window_index]
    return new_window


def get_weighted_average(series: pd.Series, weight: np.array) -> float:
    """
    calculates weighted_average, nan are ignored
    :param series: pd.Series with data
    :param weight: weights from 0 to 1 with len of series
    :return: weighted average
    """
    df = series.to_frame()
    df["weights"] = weight
    df["product"] = df.weights * series
    weighted_average = df.dropna()["product"].sum() / df.dropna()["weights"].sum()

    return weighted_average


def get_dstd_score(
        df: pd.DataFrame,
        window_len: int,
        window_function: signal.windows = signal.windows.boxcar,
        **kwargs) -> float:
    """
    calculates weighted average with given window function, and divides it by global average
    :param df: dataframe containing dstd for each magnet
    :param window_len: len of window to use
    :param window_function: window function
    :param kwargs: additional arguments used by window_function
    :return: dstd_score

    Default window is equal to:
    mask = (df.distance_to_quench < window_len) & (df.distance_to_quench > -window_len)
    df[mask].dropna().dstd.mean() / df[~mask].dropna().dstd.mean()
    """
    quenched_magnet_pos = df[df["distance_to_quench"] == 0].index.values[0]
    window = fit_window_to_data(window_function(window_len, **kwargs), len(df), quenched_magnet_pos)
    dstd_score = get_weighted_average(df.dstd, weight=window) / get_weighted_average(df.dstd, weight=1 - window)
    return dstd_score


def calc_wiggle_area(df, medfilt_len=5):
    """
    calculates the amount of neighbouring magnets containing a wiggle
    :param df: dataframe containing distance_to_quench and dstd of each magnet
    :param medfilt_len: len of medfilt, used to smoothen curve
    :return: amount of neighbouring magnets containing a wiggle
    """
    df["dstd_medfilt"] = df.dstd.rolling(medfilt_len).median()
    idx_lower_mean_right = df[(df.distance_to_quench > 0) & (df.dstd_medfilt < df.dstd_medfilt.mean())]\
        .distance_to_quench.min()
    idx_lower_mean_left = df[(df.distance_to_quench < 0) & (df.dstd_medfilt < df.dstd_medfilt.mean())]\
        .distance_to_quench.max()
    return max(0, abs(idx_lower_mean_left)) + max(0, abs(idx_lower_mean_right))


def peak12_ratio(series: pd.Series, meanfilt_len: int = 15, **kwargs) -> float:
    """
    returns the ratio between the highest two points, applies mean filter and interpolates missing values
    :param series: data
    :param meanfilt_len: length of mean filter
    :param kwargs: additional arguments for function find_peaks
    :return: between the highest two points
    """
    df_interpol = series.interpolate(method='linear')
    df_median = df_interpol.rolling(meanfilt_len).mean()

    peaks = find_peaks(df_median.values, **kwargs)
    df_sorted = df_median[peaks[0]].sort_values(ascending=False)
    if len(df_sorted) > 1:
        return df_sorted.iloc[0] / df_sorted.iloc[1]
    else:
        return 0


def get_sec_quench_features(
        df_quench_frame: pd.DataFrame,
        df_mp3_subset: pd.DataFrame,
        time_frame_after_quench: list,
        sec_quench_number: int) -> pd.DataFrame:
    """
    calculates features of given data period
    :param df_quench_frame: dataframe with frame of u diode data around secondary quench
    :param df_mp3_subset: context data: subset of mp3 fpa excel of fpa event
    :param time_frame_after_quench: amount of quenches within intervall
    :param sec_quench_number: number of secondary fpa event
    :return: dataframe with features of fpa event
    """
    event_index = df_mp3_subset.iloc[sec_quench_number + 1].name
    quench_times = df_mp3_subset["Delta_t(iQPS-PIC)"].values / 1e3

    # define dataframe to log results into
    df_results = df_mp3_subset.loc[event_index, ['Circuit Family', 'Circuit Name', 'timestamp_fgc', 'Position']]

    df_std = get_std_of_diff(df=df_quench_frame)

    df_std_meta_pos = sort_by_metadata(
        df=df_std,
        quenched_magnet=df_mp3_subset.Position.values[sec_quench_number + 1],
        circuit=df_results['Circuit Name'],
        by="Position")

    df_std_meta_elpos = sort_by_metadata(
        df=df_std,
        quenched_magnet=df_mp3_subset.Position.values[sec_quench_number + 1],
        circuit=df_results['Circuit Name'],
        by="#Electric_EE")

    df_results["sec_quench_number"] = sec_quench_number

    idx_0 = df_quench_frame.index[0]
    idx_min = df_quench_frame.min(axis=1).idxmin()
    idx_max = df_quench_frame.max(axis=1).idxmax()
    df_results["start_time"] = idx_0
    df_results["min_time"] = idx_min - idx_0
    df_results["max_time"] = idx_max - idx_0

    df_results["min_amplitude"] = df_quench_frame.loc[idx_min].mean() - \
        df_quench_frame.loc[idx_min].min()
    df_results["max_amplitude"] = df_quench_frame.loc[idx_max].max() - \
        df_quench_frame.loc[idx_max].mean()
    df_results["dU_max"] = df_quench_frame.diff().max().max()
    df_results["dU_min"] = df_quench_frame.diff().min().min()
    df_results["dstd_max"] = df_std.dstd.max()
    df_results["dstd_mean"] = df_std.dstd.mean()
    df_results["el_peak12_ratio"] = peak12_ratio(series=df_std_meta_elpos.dstd, distance=15, width=3)

    quenches_within_frame = [q for q in quench_times
                             if (q > quench_times[sec_quench_number + 1] - time_frame_after_quench[0]) &
                             (q < quench_times[sec_quench_number + 1] + time_frame_after_quench[1])]

    df_results["n_other_quenches"] = len(quenches_within_frame)

    df_results["dstd_score_pos_15"] = get_dstd_score(df_std_meta_pos, window_len=15)
    df_results["dstd_score_elpos_15"] = get_dstd_score(df_std_meta_elpos, window_len=15)
    df_results["dstd_score_pos_15_exp"] = get_dstd_score(df_std_meta_pos, window_len=15,
                                                         window_function=signal.windows.exponential)
    df_results["dstd_score_elpos_15_exp"] = get_dstd_score(df_std_meta_elpos, window_len=15,
                                                           window_function=signal.windows.exponential)
    df_results["dstd_score_pos_15_tuk"] = get_dstd_score(df_std_meta_pos, window_len=15,
                                                         window_function=signal.windows.tukey)
    df_results["dstd_score_elpos_15_tuk"] = get_dstd_score(df_std_meta_elpos, window_len=15,
                                                           window_function=signal.windows.tukey)

    threshold = df_std_meta_pos.dstd.mean() + 2 * df_std_meta_pos.dstd.std()
    df_results["wiggle_magnets"] = len(df_std_meta_pos[df_std_meta_pos.dstd.abs() > threshold])
    df_results["wiggle_area_pos"] = calc_wiggle_area(df_std_meta_pos)
    df_results["wiggle_area_elpos"] = calc_wiggle_area(df_std_meta_elpos)
    return df_results.to_frame().transpose()
