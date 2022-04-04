from typing import Union

import pandas as pd
from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery

from src.acquisition import DataAcquisition


class Leads(DataAcquisition):
    """
    Specifies method to query data for signals of group Leads
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: object
                 ):
        """
        Initializes the Leads class object, inherits from DataAcquisition.
        :param circuit_type: lhc circuit name
        :param circuit_name: lhc sector name
        :param timestamp_fgc: fgc event timestamp
        :param spark: spark object to query data from NXCALS
        """
        super().__init__(circuit_type, circuit_name, timestamp_fgc)
        self.query_builder = RbCircuitQuery(
            self.circuit_type, self.circuit_name)
        self.system = ['LEADS_ODD', 'LEADS_EVEN']
        self.signal_names = ['U_HTS', 'U_RES']
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """ method to find correct timestamp for selected signal """
        return self.query_builder.find_timestamp_leads(
            self.timestamp_fgc, self.system)

    def get_signal_data(self) -> list:
        """ method to get selected signal with specified sigmon query builder and signal timestamp  """
        return self.query_builder.query_leads(
            self.timestamp_fgc,
            self.signal_timestamp,
            system=self.system,
            signal_names=self.signal_names,
            spark=self.spark)
