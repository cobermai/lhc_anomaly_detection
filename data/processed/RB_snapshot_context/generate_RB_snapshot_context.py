from pathlib import Path
import json

import pandas as pd

from src.utils.fgc_timestamp_utils import get_fgc_timestamp, create_fpa_identifier



def generate_RB_snapshot_context(input_dir: Path, output_dir: Path, file_name: str):
    """
    process mp3 excel file for further analysis
    :param input_dir: path to mp3 excel file
    :param output_dir: path to store processed file
    :param file_name: file name to snapshot tests dates
    """
    with open(input_dir / (file_name + ".json"), "r") as file:
        json_data = file.read()
        d = json.loads(json_data)
        file.close()

    df_snapshot_raw = pd.DataFrame(d["list_timestamp_fgc_string"], index=d["list_cases"])
    df_snapshot_melted = df_snapshot_raw.T.melt(var_name=['case'], value_name='snapshot_date', ignore_index=False)

    df_snapshot_melted['fgc_datetime'] = pd.to_datetime(df_snapshot_melted['snapshot_date'],
                                                        format='%Y%m%d-%H%M%S.%f').dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    df_snapshot = df_snapshot_melted.reset_index().rename(columns={"index": 'Circuit Name'})
    df_snapshot["Circuit Family"] = "RB"
    df_snapshot['timestamp_fgc'] = df_snapshot.apply(get_fgc_timestamp, axis=1)

    df_snapshot["fpa_identifier"] = df_snapshot.apply(lambda row: create_fpa_identifier(row['Circuit Family'],
                                                                                        row['Circuit Name'],
                                                                                        row['timestamp_fgc']), axis=1)

    df_snapshot.to_csv(output_dir / (file_name + ".csv"))


if __name__ == "__main__":
    data_dir = Path("../../raw/MP3_context/")
    output_dir = Path("")
    file_name = "RB_snapshot_context"

    generate_RB_snapshot_context(input_dir=data_dir, output_dir=output_dir, file_name=file_name)
