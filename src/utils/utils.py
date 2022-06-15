from pathlib import Path
import glob

import pandas as pd

def log_acquisition(identifier: dict, log_data: dict, log_path: Path) -> None:
    """
    method stores logs data to given csv, if identifier not exists, a new line is created
    :param identifier: dict to specify location to log data
    :param log_data: dict data to log
    :param log_path: directory where csv is stored
    """
    file_name = f"{identifier['circuit_type']}_{identifier['circuit_name']}_{identifier['timestamp_fgc']}.csv"
    file_path = log_path / file_name

    if not file_path.is_file():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(identifier, index=[0])
    else:
        df = pd.read_csv(file_path)

    # add context data
    for key, value in log_data.items():
        df.loc[0, key] = value
    df.to_csv(file_path, index=False)

def load_acquisition_log(path: Path) -> pd.DataFrame:
    """
    gathers all .csv files from path
    :param path: path where acquisition is stored
    :return: df with acquisition log
    """
    files = sorted(glob.glob(str(path / "*.csv")))
    df = pd.concat(map(pd.read_csv, files))
    return df

def flatten_list(stacked_list: list) -> list:
    """
    method flattens list
    :param stacked_list: list with dim > 1
    :return: list with dim = 1
    """
    return [item for sublist in stacked_list for item in sublist]