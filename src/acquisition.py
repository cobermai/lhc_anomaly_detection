from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


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

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
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





    
