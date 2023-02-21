from pathlib import Path, PurePosixPath
import re

import h5py
import pandas as pd
import numpy as np

from src.utils.utils import log_acquisition


def acquisition_to_hdf5(acquisition: "DataAcquisition",
                        file_dir: Path,
                        context_dir_name: str = "context",
                        failed_queries_dir_name: str = "failed",
                        data_dir_name: str = "data") -> None:
    """
    method stores acquisition data as hdf5, and logs both successful and failed queries as csv
    :param acquisition: DataAcquisition class to query data from
    :param file_dir: directory to store data and log data
    :param context_dir_name: name of directory to store context data
    :param failed_queries_dir_name: name of directory to store context data of failed queries
    :param data_dir_name: name of directory to store data
    """
    context_path = file_dir / context_dir_name
    context_path.mkdir(parents=True, exist_ok=True)
    failed_queries_path = file_dir / failed_queries_dir_name
    failed_queries_path.mkdir(parents=True, exist_ok=True)
    data_dir = file_dir / data_dir_name
    data_dir.mkdir(parents=True, exist_ok=True)

    identifier = {'circuit_type': acquisition.circuit_type,
                  'circuit_name': acquisition.circuit_name,
                  'timestamp_fgc': acquisition.timestamp_fgc}

    group_name = acquisition.__class__.__name__
    file_name = f"{identifier['circuit_type']}_{identifier['circuit_name']}_{identifier['timestamp_fgc']}.hdf5"

    try:
        list_df = acquisition.get_signal_data()

        for df in list_df:
            if isinstance(df, pd.DataFrame):
                if not df.empty:

                    df_to_hdf(file_path=data_dir / file_name, df=df, hdf_dir=group_name)
                    context_data = {f"{group_name + '_' + str(df.columns.values[0])}": len(df)}

                    log_acquisition(
                        identifier=identifier,
                        log_data=context_data,
                        log_path=context_path)
                else:
                    log_acquisition(
                        identifier=identifier,
                        log_data={group_name: "empty DataFrame returned"},
                        log_path=failed_queries_path)
            else:
                log_acquisition(
                    identifier=identifier,
                    log_data={group_name: "no DataFrame returned"},
                    log_path=failed_queries_path)

    except Exception as e:
        log_acquisition(
            identifier=identifier,
            log_data={group_name: str(e)},
            log_path=failed_queries_path)


def df_to_hdf(file_path: Path, df: pd.DataFrame, hdf_dir: str = ""):
    """
    converts DataFrame into hdf files. For each column a group is created.
    Each column is a group with one index dataset and one value dataset
    :param file_path: path of the hdf file to write
    :param df: DataFrame to write to hdf5
    :param hdf_dir: directory inside hdf5 file where to store DataFrame
    """
    with h5py.File(file_path, "a") as f:
        for column in df.columns:
            append_or_overwrite_hdf_group(file=f,
                                          hdf_path=f"{hdf_dir}/{df[column].name}/values",
                                          data=df[column].values)
            append_or_overwrite_hdf_group(file=f,
                                          hdf_path=f"{hdf_dir}/{df[column].name}/index",
                                          data=df[column].index.values)


def append_or_overwrite_hdf_group(file: h5py.File, hdf_path: str, data: np.array):
    """
    append data to h5py file, appends if group not exists, overwrite it
    TODO: last group in list gives error: "Unable to open object (bad header version number)"
    :param file: opened h5 file
    :param hdf_path: path within h5 file
    :param data: data to add
    """
    if hdf_path in file:
        del file[hdf_path]
        file[hdf_path] = data
    else:
        file[hdf_path] = data


def hdf_to_df(file_path: Path, hdf_dir: str = "") -> pd.DataFrame:
    """
    converts hdf file into dataframe, given the path to the data in the hdf file
    :param file_path: path of the hdf file to load
    :param hdf_dir: directory inside hdf5 file where DataFrame is stored
    : return: DataFrame with values and index of given hdf path
    """
    with h5py.File(file_path, "r") as f:
        data = np.array(f[hdf_dir].get("values"))
        index = np.array(f[hdf_dir].get("index"))
        colum_name = hdf_dir
    return pd.DataFrame(data, index=index, columns=[colum_name])


def get_hdf_tree(file_path: Path, regex_list: list = ['']) -> pd.DataFrame:
    """
    get paths of datasets in hdf file which contain regex
    :param file_path: path of the hdf file to extract file tree
    :param regex_list: list of strings which file should contain
    : return: list of paths with regex in hdf5 file
    """

    def extract_file_tree(name, node):
        """
        append parent path of hdf dataset file
        :param name: name of dataset
        :param node: node of dataset
        """
        if isinstance(node, h5py.Dataset):
            parent_path = str(PurePosixPath(name).parent)
            if not parent_path in file_tree:
                file_tree.append(parent_path)
        return None

    with h5py.File(file_path, "r") as f:
        file_tree = []
        f.visititems(extract_file_tree)

        file_tree_filtered = [t for t in file_tree for r in regex_list if bool(re.search(r, t))]

    return file_tree_filtered


def load_from_hdf_with_regex(file_path: Path, regex_list: list = ['']) -> list:
    """
    get datasets in hdf file which contain regex
    :param file_path: path of the hdf file to extract file tree
    :param regex_list: list of strings which file should contain
    : return: list of DataFrames with data of regex
    """
    hdf_paths = get_hdf_tree(file_path=file_path, regex_list=regex_list)
    data = []
    for hdf_path in hdf_paths:
        data.append(hdf_to_df(file_path=file_path, hdf_dir=hdf_path))
    return data


def data_list_to_df(data: list) -> pd.DataFrame:
    """
    transforms list of dataframes into single dataframe. Dataframes must have equal length.
    :param data: list of dataframes
    :return: single dataframe
    """
    data_values = np.array([df[df.columns[0]].to_list() for df in data]).T
    data_index = data[0].index  # TODO: interpolate index, not take first one
    data_columns = [df.columns.values[0].split("/")[1] for df in data]
    return pd.DataFrame(data_values, index=data_index, columns=data_columns)