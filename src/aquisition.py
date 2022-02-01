from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union

class DataAcquisition(ABC):
    """
    abstract class which acts as a template to download LHC cicuit data
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
                 spark: Optional[object] = None
                 ):
        """
        Specifies data to query from
        """
        self.circuit_type = circuit_type
        self.circuit_name = circuit_name
        self.timestamp_fgc = timestamp_fgc
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """
        method to find correct timestamp for selected signal, default is fgc timestamp
        """
        return self.timestamp_fgc

    @abstractmethod
    def get_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """

def acquire_data(creator: DataAcquisition, circuit_type, circuit_name, timestamp_fgc) -> list:
    return creator(circuit_type, circuit_name, timestamp_fgc).get_signal_data()
