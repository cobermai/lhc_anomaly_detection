from pathlib import Path

import pandas as pd
from lhcsmapi.Time import Time
from lhcsmapi.metadata.SignalMetadata import SignalMetadata
from lhcsmapi.pyedsl.dbsignal.post_mortem.PmDbRequest import PmDbRequest


def find_real_fgc_timestamp(circuit_name: str, fgc_datetime: str) -> list:
    """
    returns real fgc timestamp given circuit name and mp3 fgc date
    :param circuit_name: string with circuit
    :param fgc_datetime: string with mp3 fgc date
    :return: list with real fgc timestamps
    """
    fgc_timestamp = Time.to_unix_timestamp(fgc_datetime)
    metadata_fgc = SignalMetadata.get_circuit_signal_database_metadata(
        'RB', circuit_name, 'PC', 'PM', timestamp_query=fgc_timestamp)

    one_sec_in_ns = 1e9
    start_time = fgc_timestamp - one_sec_in_ns
    end_time = fgc_timestamp + one_sec_in_ns

    source_timestamp_fgc = PmDbRequest.find_events(metadata_fgc['source'],
                                                   metadata_fgc['system'],
                                                   metadata_fgc['className'],
                                                   t_start=start_time,
                                                   t_end=end_time)

    return [(circuit_name, el[1]) for el in source_timestamp_fgc]


def get_fgc_timestamp(mp3_df: pd.DataFrame) -> int:
    """
    getting a list of real fgc timestamps, given the timestamps from the mp3 excel file
    :param mp3_df: df from mp3 fpa excel
    :return: real fgc timestamp
    """
    date_time_str = f"{mp3_df['Date (FGC)']} {mp3_df['Time (FGC)']}".replace("00:00:00 ", "")

    real_fgc_timestamps = find_real_fgc_timestamp(mp3_df['Circuit Name'], date_time_str)

    if not real_fgc_timestamps:
        return None

    _, real_fgc_timestamp = real_fgc_timestamps[0]
    return int(real_fgc_timestamp)


def get_fgc_timestamp_missing(mp3_df: pd.DataFrame) -> int:
    """
    getting a list of real fgc timestamps, given the timestamps from the mp3 excel file with wrong hours
    :param mp3_df: df from mp3 fpa excel
    :return: real fgc timestamp
    """
    date_time_str = f"{mp3_df['Date (FGC)']} {mp3_df['Time (FGC)']}".replace("00:00:00", "")
    for t in range(6, 24): # LHC operation from 6:00 to 24:00
        date_time_str_new = date_time_str.replace(" 00:", f" {t}:")
        real_fgc_timestamps = find_real_fgc_timestamp(mp3_df['Circuit Name'], date_time_str_new)
        if real_fgc_timestamps:
            _, real_fgc_timestamp = real_fgc_timestamps[0]
            return int(real_fgc_timestamp)

def process_mp3_excel(data_dir: Path, mp3_file_name: str):
    """
    process mp3 excel file for further analysis
    :param data_dir: path to mp3 excel file
    :param mp3_file_name: mp3 excel file name
    """
    mp3_fpa_df_raw = pd.read_excel(data_dir / (mp3_file_name + ".xlsx"), engine='openpyxl')

    # First row contains units, 9 rows contain only "Before Notebooks" and "After Notebooks" information
    mp3_fpa_df = mp3_fpa_df_raw.dropna(subset=['Date (FGC)', 'Circuit Name'])

    # look up real fgc timestamp
    mp3_fpa_df['timestamp_fgc'] = mp3_fpa_df.apply(get_fgc_timestamp, axis=1)

    # some fgc timestamps have wrong hours
    mp3_fpa_df_primary_missing = mp3_fpa_df[(mp3_fpa_df.timestamp_fgc.isna()) & (mp3_fpa_df['Nr in Q event'] == 1)]
    mp3_fpa_df_primary_missing['timestamp_fgc'] = mp3_fpa_df_primary_missing.apply(get_fgc_timestamp_missing, axis=1)
    found_fgc_timestamps_df = mp3_fpa_df_primary_missing["timestamp_fgc"].dropna()
    mp3_fpa_df.loc[found_fgc_timestamps_df.index, "timestamp_fgc"] = mp3_fpa_df_primary_missing[
        "timestamp_fgc"].dropna().values

    mp3_fpa_df.to_csv(data_dir / (mp3_file_name + "_processed.csv"))


if __name__ == "__main__":
    data_dir = Path("../../data")
    mp3_file_name = "RB_TC_extract_2021_11_22"

    process_mp3_excel(data_dir=data_dir, mp3_file_name=mp3_file_name)




