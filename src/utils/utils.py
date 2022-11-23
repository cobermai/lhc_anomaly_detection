from pathlib import Path
import glob
from typing import Optional, Callable, Union

import numpy as np
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


def interp(df, new_index):
    """
    Return a new DataFrame with all columns values interpolated
    to the new_index values.
    :param df: old dataframe to resample
    :param new_index: index of new dataframe
    :rtype:
    """
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name
    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)
    return df_out


def dict_to_df_meshgrid(param_grid: dict, add_type: bool = True) -> pd.DataFrame:
    """
    function generates a mesh-grid  pandas dataframe, which can be iterated over for sensitivity analysis
    :param param_grid: dictionary with variables to iterate over
    :param add_type: add type of each column based on first value of each key
    :return: mesh-grid to iterate over for sensitivity analysis
    """
    vary_values = list(map(param_grid.get, param_grid.keys()))
    meshgrid = np.array(np.meshgrid(*vary_values)).T.reshape(-1, len(param_grid.keys()))
    df_meshgrid = pd.DataFrame(meshgrid, columns=param_grid.keys())
    if add_type:
        df_meshgrid = df_meshgrid.astype({k: type(v[0]) for k, v in param_grid.items()})
    return df_meshgrid


def nanargsort(array: np.array) -> np.array:
    """
    argsort with dropna
    :param array: array to sort
    :return: index of sorted array without nan
    """
    return np.argsort(array)[(np.sort(array) >= 0)]


def pd_dict_filt(df: pd.DataFrame, filt: Optional[dict]=None):
    """
    filters pandas DataFrame by dictonary
    :param df: DataFrame to filer
    :param filt: dict with key columns and args values, equal to df[df[key]== values]
    :return: filtered DataFrame
    """
    if filt is None:
        return df
    else:
        # multi index allows faster query, add if it does not exist,
        if not all([f in df.index.names for f in filt]):
            df = df.set_index(list(filt.keys()), drop=False, append=True)

        level = list(np.argwhere(np.isin(df.index.names, list(filt.keys()))).flatten())  # position of multindex
        return df.xs(tuple(filt.values()), level=level)


def merge_array(array: np.array,
                merge_index: Union[list, np.array],
                axis: int = -1,
                func: Callable = np.mean) -> np.array:
    """
    merge columns of array
    e.g. array=a; merge_index=[2, 3, [4, 5]] -> out: np.array([a[2], a[3], np.mean(a[4], a[5])])
    :param array: array to perform merge on, dim: (..., row, columns)
    :param merge_index: list of indices to merge
    :param axis: axis to merge on, default is to merge on columns
    :param func: function used to merge, default is mean
    :return: filtered DataFrame
    """
    matrix_merged = [list(func(array[..., i], axis=axis)) if isinstance(i, list) else list(array[..., i]) for i in
                     merge_index]
    return np.moveaxis(matrix_merged, 0, -1)



