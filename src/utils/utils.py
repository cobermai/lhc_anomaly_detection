from pathlib import Path

import pandas as pd

def log_acquisition(identifier: dict, log_data: dict, log_path: Path) -> None:
    """
    method stores logs data to given csv, if identifier not exists, a new line is created
    :param log_data: dict data to log
    :param log_path: directory where csv is stored
    """

    if not log_path.is_file():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(identifier, index=[0])
    else:
        df = pd.read_csv(log_path)
        # add identifier if not existing
        if not df[identifier.keys()].isin(
                identifier.values()).all(axis=1).values[-1]:
            df_new = pd.DataFrame(identifier, index=[0])
            df = pd.concat([df, df_new], axis=0)

    # add context data
    for key, value in log_data.items():
        df.loc[df[identifier.keys()].isin(
            identifier.values()).all(axis=1), key] = value
    df.to_csv(log_path, index=False)

def flatten_list(stacked_list: list) -> list:
    """
    method flattens list
    :param stacked_list: list with dim > 1
    :return: list with dim = 1
    """
    return [item for sublist in stacked_list for item in sublist]