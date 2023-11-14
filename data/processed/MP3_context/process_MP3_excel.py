from pathlib import Path

import pandas as pd

from src.utils.fgc_timestamp_utils import get_fgc_timestamp, get_fgc_timestamp_missing, create_fpa_identifier


def process_mp3_excel(input_dir: Path, output_dir: Path, mp3_file_name: str):
    """
    process mp3 excel file for further analysis
    :param input_dir: path to mp3 excel file
    :param output_dir: path to store processed file
    :param mp3_file_name: mp3 excel file name
    """
    mp3_fpa_df_raw = pd.read_excel(input_dir / (mp3_file_name + ".xlsx"), engine='openpyxl')

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

    mp3_fpa_df["fpa_identifier"] = mp3_fpa_df.apply(lambda row: create_fpa_identifier(row['Circuit Family'],
                                                                                      row['Circuit Name'],
                                                                                      row['timestamp_fgc']), axis=1)

    mp3_fpa_df.to_csv(output_dir / (mp3_file_name + "_processed.csv"))



if __name__ == "__main__":
    data_dir = Path("../data/raw/MP3_context/")
    output_dir = Path("")
    mp3_file_name = "RB_TC_extract_2023_03_13"

    process_mp3_excel(input_dir=data_dir, output_dir=output_dir, mp3_file_name=mp3_file_name)
