from pathlib import Path

import h5py
import pandas as pd


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
