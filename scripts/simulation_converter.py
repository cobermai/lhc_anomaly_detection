import glob
import os
import subprocess
from pathlib import Path

import pandas as pd
from steam_nb_api.utils import CSD_Reader

from src.utils.hdf_tools import df_to_hdf


def cir2csd(cir_path: Path, pspice_path: Path,
            script_name: str = "cir2csd.sh", cir_name: str = "Circuit.cir", csd_name: str = "Circuit.csd"):
    """
    Conversion of pspice .cir file to .csd simulation file. Execution only on Windows. Pspice must be installed.
    :param cir_path: path to .cir file, no unc path possible
    :param pspice_path: path to pspice executable
    :param script_name: name of script to create in cir_path
    :param cir_name: name of .cir file to load from cir_path
    :param csd_name: name of csd file, used to check if it exists
    """
    if not os.path.isfile(cir_path / script_name):
        with open(cir_path / script_name, "w") as file:
            pspice_dir = str(pspice_path).replace("\\", "\\\\")
            cir_dir = str(cir_path / cir_name).replace("\\", "\\\\")
            file.write("#!/usr/bin/bash\n")
            file.write(f"start \"\" \"{pspice_dir}\" \"{cir_dir}\"")
        os.system(f"chmod +x {cir_path / script_name}")

    if not os.path.isfile(cir_path / csd_name):
        script_dir = str(cir_path / script_name).replace("\\", "\\\\")
        subprocess.call(f"{script_dir}", shell=True)


def csd2hdf(csd_path: Path, csd_name: str = "Circuit.csd", hdf_name: str = "simulation.hdf5") -> pd.DataFrame:
    """
    Conversion of csd files to hdf files with the STEAM CSD reader.
    :param csd_path: path of csd file
    :param csd_name: name of csd file, to load data from
    :param hdf_name: name of hdf file, to store data
    :return: dataframe with loaded simulation data
    """
    if not os.path.isfile(csd_path / hdf_name):
        csd = CSD_Reader.CSD_read(csd_path / csd_name)
        df = pd.DataFrame.from_dict(csd.data_dict)

        hdf_path = Path(csd_path) / hdf_name
        df_to_hdf(file_path=hdf_path, df=df)
        return df


if __name__ == "__main__":
    # bash script execution does not support unc path
    file_path_local = Path("C:\\Users\\cobermai\\STEAM\\PSPICE\\")
    pspice_path = Path("C:\\EDA\\Cadence\\OrCAD_166\\tools\\pspice\\psp_cmd.exe")

    cir_files = sorted(glob.glob(str(file_path_local / "*//*t.cir")))

    for file in cir_files:
        parent_path = Path(file).parents[0]
        cir2csd(cir_path=parent_path, pspice_path=pspice_path)
        df = csd2hdf(csd_path=parent_path)
