from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from src.acquisition import DataAcquisition
from src.utils.utils import flatten_list
import pandas as pd
from typing import Optional, Union

class LEADS(DataAcquisition):
    """
    Subclass of DataAquistion to query PC_PM
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: object
                 ):
        super(LEADS, self).__init__(circuit_type, circuit_name, timestamp_fgc)
        self.query_builder = RbCircuitQuery(self.circuit_type, self.circuit_name)
        self.system = ['LEADS_ODD', 'LEADS_EVEN']
        self.signal_names = ['U_HTS', 'U_RES']
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """
        method to find correct timestamp for selected signal
        """
        return self.query_builder.find_timestamp_leads(self.timestamp_fgc, self.system)

    def get_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """
        return self.query_builder.query_leads(self.timestamp_fgc, self.signal_timestamp, system=self.system,
                                              signal_names=self.signal_names, spark=self.spark)
