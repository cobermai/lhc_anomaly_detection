from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union
from pathlib import Path

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
                 hdf_dir: Path,
                 context_dir: Path
    ):
        """
        Specifies data to query from
        """
        self.circuit_type = circuit_type,
        self.circuit_name = circuit_name,
        self.timestamp_fgc = timestamp_fgc,
        self.hdf_dir = hdf_dir,
        self.context_dir = context_dir

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

    def log_transformation(self) -> None:
        """
        abstract method to store meta data
        """

    def to_hdf5(self) -> None:
        """
        abstract method to store data
        """
        data = self.get_signal_data()
        for df in data:
            file_name = f"{self.circuit_type}_{self.circuit_type}_{self.timestamp_fgc}_{df.columns.values[0]}.pkl"
            df.to_pickle(self.hdf_dir / file_name)
            self.log_transformation()



