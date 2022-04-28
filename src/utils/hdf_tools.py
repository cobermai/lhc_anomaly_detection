from pathlib import Path

import h5py
import pandas as pd

from src.utils.utils import log_acquisition

def acquisition_to_hdf5(acquisition: "DataAcquisition", file_dir: Path) -> None:
    """
    method stores acquisition data as hdf5, and logs both successful and failed queries as csv
    :param acquisition: DataAcquisition class to query data from
    :param file_dir: directory to store data and log data
    """
    context_path = file_dir / "context_data.csv"
    failed_queries_path = file_dir / "failed_queries.csv"
    data_dir = file_dir / "data"
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
                    file_name = f"{acquisition.circuit_type}_{acquisition.circuit_name}_{acquisition.timestamp_fgc}.hdf5"
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
    :param file_path: path of the hdf file to write.
    :param df: DataFrame to write to hdf5
    :param hdf_dir: directory inside hdf5 file where to store DataFrame
    """
    with h5py.File(file_path, "a") as f:
        for column in df.columns:
            grp = f.create_group(hdf_dir + "/" + df[column].name)
            grp.create_dataset("values", data=df[column].values)
            grp.create_dataset("index", data=df[column].index)
