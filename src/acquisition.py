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
