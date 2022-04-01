from typing import Optional, Union

import pandas as pd
from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery

from src.acquisition import DataAcquisition


class CurrentVoltageDiodeLeadsNXCALS(DataAcquisition):
    """ Specifies method to query data for signals of group CurrentVoltageDiodeLeadsNXCALS """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: Optional[object] = None
                 ):
        """
        Initializes the CurrentVoltageDiodeLeadsNXCALS class object, inherits from DataAcquisition.
        :param circuit_type: lhc circuit name
        :param circuit_name: lhc sector name
        :param timestamp_fgc: fgc event timestamp
        :param spark: spark object to query data from NXCALS
        """
        super().__init__(circuit_type, circuit_name, timestamp_fgc)
        self.query_builder = RbCircuitQuery(
            self.circuit_type, self.circuit_name)
        self.duration = [(50, 's'), (350, 's')]
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """ method to find correct timestamp for selected signal """
        return self.query_builder.find_source_timestamp_qds(
            self.timestamp_fgc, duration=self.duration)

    def get_signal_data(self) -> list:
        """ method to get selected signal with specified sigmon query builder and signal timestamp """
        signals = self.query_builder.query_current_voltage_diode_leads_nxcals(
            self.signal_timestamp, spark=self.spark, duration=self.duration)

        # [df.rename(columns={df.columns.values[0]:df.columns.values[0] + "_LEADS"}) for sublist in signals for df in sublist]
        return signals

