from pathlib import Path

import h5py
import pandas as pd
import numpy as np

from src.utils.utils import log_acquisition

def acquisition_to_hdf5(acquisition: "DataAcquisition", file_dir: Path) -> None:
    """
    method stores acquisition data as hdf5, and logs both successful and failed queries as csv
    :param acquisition: DataAcquisition class to query data from
    :param file_dir: directory to store data and log data
    """
    context_path = file_dir / "context_1"
    failed_queries_path = file_dir / "failed_1"
    data_dir = file_dir / "data_1"
    data_dir.mkdir(parents=True, exist_ok=True)

    identifier = {'circuit_type': acquisition.circuit_type,
                  'circuit_name': acquisition.circuit_name,
                  'timestamp_fgc': acquisition.timestamp_fgc}

    group_name = acquisition.__class__.__name__
    try:
        list_df = acquisition.get_signal_data()
        for df in list_df:
            if isinstance(df, pd.DataFrame):
                if not df.empty:
                    file_name = f"{identifier['circuit_type']}_{identifier['circuit_name']}_{identifier['timestamp_fgc']}.hdf5"
                    df_to_hdf(file_path=data_dir / file_name , df=df, hdf_dir=group_name)
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
    :param file_path: path of the hdf file to write.
    :param df: DataFrame to write to hdf5
    :param hdf_dir: directory inside hdf5 file where to store DataFrame
    """
    with h5py.File(file_path, "a") as f:
        for column in df.columns:
            append_or_overwrite_hdf_group(file=f,
                                          hdf_path=f"{hdf_dir}/{df[column].name}/values",
                                          data=df[column].values)
        append_or_overwrite_hdf_group(file=f,
                                      hdf_path=f"{hdf_dir}/index",
                                      data=df[column].index.values)


def append_or_overwrite_hdf_group(file: h5py.File, hdf_path: str, data: np.array):
    """
    append data to h5py file, appends if group not exists, overwrite it it does
    file: opened h5 file
    h5_path: path within h5 file
    data: data to add
    """
    if hdf_path in file:
        del file[hdf_path]
        file[hdf_path] = data
    else:
        file[hdf_path] = data
