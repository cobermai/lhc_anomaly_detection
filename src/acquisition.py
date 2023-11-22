import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List

import pandas as pd
from PIL import Image
from pyspark.sql import SparkSession

from src.utils.hdf_tools import acquisition_to_hdf5, load_from_hdf_with_regex
from src.utils.utils import log_acquisition
from src.visualisation.visualisation import plot_hdf


class DataAcquisition(ABC):
    """
    Abstract class which acts as a template to download LHC circuit data.
    Questions:
    * EE_U_DUMP_RES_PM: query both 'EE_ODD', 'EE_EVEN'?, take only first element of list?
    * EE_T_RES_PM: what is t_res_odd_1_df ? - not implemented yet
    * VOLTAGE_NXCALS: whats the best way to pass spark?
    * VOLTAGE_LOGIC_IQPS: do I need u_qds_dfs2 from second board (A/B)?
    * LEADS: can I pass system as list
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 ):
        """
        :param circuit_type: lhc circuit name
        :param circuit_name: lhc sector name
        :param timestamp_fgc: fgc event timestamp
        """
        self.circuit_type = circuit_type
        self.circuit_name = circuit_name
        self.timestamp_fgc = timestamp_fgc

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame, list]:
        """
        Method to find correct timestamp for selected signal, the default is a fgc timestamp.
        :return: fgc timestamp as int or DataFrame of ints
        """
        return self.timestamp_fgc

    @abstractmethod
    def get_signal_data(self) -> list:
        """
        Abstract method to get selected signal.
        :return: list of dataframes containing queried signals
        """


def download_data(fpa_identifiers: List[str],
                  signal_groups: list,
                  output_dir: Path,
                  plot_regex: Optional[List[str]] = None,
                  spark: Optional[SparkSession] = None):
    """
    Download data using signal_groups for all FPA events
    :param fpa_identifiers: list of FPA identifiers to download
    :param signal_groups: list of classes from src/acquisition
    :param output_dir: path to store data
    :param plot_regex: list of regex signals to query.
    Regex of signals can be found in src/acquisition/acquisition_example.png
    :param spark: spark object to query data from NXCALS
    """

    date_suffix = datetime.now().strftime('%Y%m%d')
    plot_dir = output_dir / Path(f'{date_suffix}_data_plots')
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    for fpa_identifier in fpa_identifiers:
        try:
            fpa_identifier_split = fpa_identifier.split("_")
            fpa_identifier_dict = {'circuit_type': fpa_identifier_split[0],
                                   'circuit_name': fpa_identifier_split[1],
                                   'timestamp_fgc': int(fpa_identifier_split[2])}

            plot_path = plot_dir / (fpa_identifier + ".png")
            file_path = output_dir / Path(f'{date_suffix}_data') / (fpa_identifier + ".hdf5")

            if not os.path.isfile(file_path):
                for signal_group in signal_groups:
                    group = signal_group(**fpa_identifier_dict, spark=spark)
                    acquisition_to_hdf5(acquisition=group,
                                        file_dir=output_dir,
                                        context_dir_name=f"{date_suffix}_context",
                                        failed_queries_dir_name=f"{date_suffix}_failed",
                                        data_dir_name=f"{date_suffix}_data")
                log_acquisition(identifier=fpa_identifier_dict, log_data={"download_complete": True},
                                log_path=output_dir / f"{date_suffix}_context")

                if plot_regex is not None:
                    if os.path.isfile(file_path):
                        data = load_from_hdf_with_regex(file_path)
                        plot_hdf(data=data, column_regex=plot_regex, fig_path=plot_path)
        except:
            # Create an empty image with transparency (RGBA mode)
            empty_image = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0))
            # Save the image as a PNG file
            empty_image.save(f'{fpa_identifier}.png')



    
