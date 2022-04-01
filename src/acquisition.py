from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd

from src.utils.hdf_tools import df_to_hdf
from src.utils.utils import log_acquisition

class DataAcquisition(ABC):
    """
    abstract class which acts as a template to download LHC circuit data
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

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """
        method to find correct timestamp for selected signal, default is fgc timestamp
        :return: fgc timestamp as int or DataFrame of ints
        """
        return self.timestamp_fgc

    @abstractmethod
    def get_signal_data(self) -> list:
        """
        abstract method to get selected signal
        :return: list of dataframes containing queried signals
        """


    def to_hdf5(self, file_dir: Path) -> None:
        """
        method stores data as hdf5, and logs both successful and failed queries as csv
        :param file_dir: directory to store data and log data
        """
        context_path = file_dir / "context_data.csv"
        failed_queries_path = file_dir / "failed_queries.csv"
        data_dir = file_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        fpa_identifier = {'circuit_type': self.circuit_type,
                      'circuit_name': self.circuit_name,
                      'timestamp_fgc': self.timestamp_fgc}

        group_name = self.__class__.__name__
        try:
            list_df = self.get_signal_data()
            for df in list_df:
                if isinstance(df, pd.DataFrame):
                    if not df.empty:
                        file_name = f"{self.circuit_type}_{self.circuit_name}_{self.timestamp_fgc}.hdf5"
                        df_to_hdf(file_path=data_dir / file_name, df=df, hdf_dir=group_name)

                        context_data = {f"{group_name + '_' + str(df.columns.values[0])}": len(df)}
                        log_acquisition(
                            identifier=fpa_identifier,
                            log_data=context_data,
                            log_path=context_path)
                    else:
                        log_acquisition(
                            identifier=fpa_identifier,
                            log_data={group_name: "empty DataFrame returned"},
                            log_path=failed_queries_path)
                else:
                    log_acquisition(
                        identifier=fpa_identifier,
                        log_data={group_name: "no DataFrame returned"},
                        log_path=failed_queries_path)

        except Exception as e:
            log_acquisition(
                identifier=fpa_identifier,
                log_data={group_name: str(e)},
                log_path=failed_queries_path)
