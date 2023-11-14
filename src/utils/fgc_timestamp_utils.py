import os
from pathlib import Path

import numpy as np
import pandas as pd
from lhcsmapi.Time import Time
from lhcsmapi.pyedsl.dbsignal.post_mortem.PmDbRequest import PmDbRequest
from lhcsmapi.metadata import signal_metadata

from src.utils.utils import load_acquisition_log


def find_real_fgc_timestamp(circuit_name: str, fgc_datetime: str) -> list:
    """
    returns real fgc timestamp given circuit name and mp3 fgc date
    :param circuit_name: string with circuit
    :param fgc_datetime: string with mp3 fgc date
    :return: list with real fgc timestamps
    """
    fgc_timestamp = Time.to_unix_timestamp(fgc_datetime)
    metadata_fgc = signal_metadata.get_signal_metadata('RB', circuit_name, 'PC', 'PM', fgc_timestamp)

    one_sec_in_ns = 1e9
    start_time = fgc_timestamp - one_sec_in_ns
    end_time = fgc_timestamp + one_sec_in_ns

    source_timestamp_fgc = PmDbRequest.find_events(metadata_fgc['source'],
                                                   metadata_fgc['system'],
                                                   metadata_fgc['className'],
                                                   t_start=start_time,
                                                   t_end=end_time)

    return [(circuit_name, el[1]) for el in source_timestamp_fgc]


def get_fgc_timestamp(mp3_df: pd.Series) -> int:
    """
    getting a list of real fgc timestamps, given the timestamps from the mp3 excel file
    :param mp3_df: df with 'Circuit Name' and "fgc_datetime" (format: '%Y-%m-%d %H:%M:%S.%f')
    :return: real fgc timestamp
    """
    if not "fgc_datetime" in mp3_df.index.values:
        mp3_df['fgc_datetime'] = f"{mp3_df['Date (FGC)']} {mp3_df['Time (FGC)']}".replace("00:00:00 ", "")

    real_fgc_timestamps = find_real_fgc_timestamp(mp3_df['Circuit Name'], mp3_df['fgc_datetime'])

    if not real_fgc_timestamps:
        return None

    _, real_fgc_timestamp = real_fgc_timestamps[0]
    return int(real_fgc_timestamp)


def get_fgc_timestamp_missing(mp3_df: pd.Series) -> int:
    """
    getting a list of real fgc timestamps, given the timestamps from the mp3 excel file with wrong hours
    :param mp3_df: df from mp3 fpa excel
    :return: real fgc timestamp
    """
    date_time_str = f"{mp3_df['Date (FGC)']} {mp3_df['Time (FGC)']}".replace("00:00:00", "")
    for t in range(6, 24):  # LHC operation from 6:00 to 24:00
        date_time_str_new = date_time_str.replace(" 00:", f" {t}:")
        real_fgc_timestamps = find_real_fgc_timestamp(mp3_df['Circuit Name'], date_time_str_new)
        if real_fgc_timestamps:
            _, real_fgc_timestamp = real_fgc_timestamps[0]
            return int(real_fgc_timestamp)


def select_fgc_period(mp3_df: pd.DataFrame, lower_threshold: str, upper_threshold: str) -> pd.DataFrame:
    """
    function selects time period from mp3 excel file to analyze
    :param mp3_df: df from mp3 fpa excel
    :param lower_threshold: lower threshold date
    :param upper_threshold: upper threshold date
    :return: real excel file within given period
    """
    lower_threshold_unix = Time.to_unix_timestamp(lower_threshold)
    upper_threshold_unix = Time.to_unix_timestamp(upper_threshold)
    mp3_fpa_df_period = mp3_df[(mp3_df['timestamp_fgc'] >= lower_threshold_unix) &
                               (mp3_df['timestamp_fgc'] <= upper_threshold_unix)].reset_index(drop=True)

    return mp3_fpa_df_period


def select_fgc_not_downloaded(context_path: Path, mp3_df: pd.DataFrame) -> pd.DataFrame:
    """
    gathers all .csv files from path
    :param path: path where acquisition is stored
    :param mp3_df: df from mp3 fpa excel
    :return: mp3 df with fpa events not downloaded
    """
    if os.path.exists(context_path):
        df_context = load_acquisition_log(path=context_path)
        dowloaded_fgc_ts = df_context[df_context.download_complete == True].timestamp_fgc.values

        mp3_fpa_df_to_download = mp3_df[~mp3_df.timestamp_fgc.isin(dowloaded_fgc_ts)]
    else:
        mp3_fpa_df_to_download = mp3_df

    return mp3_fpa_df_to_download


def create_fpa_identifier(circuit_type: str, circuit_name: str, timestamp_fgc: int) -> str:
    """
    creates identifier, unique for each fast power abort. e.g. "RB_RB.A45_1544300287820000000"
    :param circuit_type: circuit type
    :param circuit_name: name of circuit
    :param timestamp_fgc: fgc timestamp of fpa event
    :return: mp3 df with fpa events not downloaded
    """
    if np.isnan(timestamp_fgc):
        return ""
    else:
        return f"{circuit_type}_{circuit_name}_{int(timestamp_fgc)}"
