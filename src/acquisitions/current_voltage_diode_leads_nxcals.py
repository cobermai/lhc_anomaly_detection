from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from src.acquisition import DataAcquisition
import pandas as pd
from typing import Optional, Union


class CURRENT_VOLTAGE_DIODE_LEADS_NXCALS(DataAcquisition):
    """
    Specifies method to query data for signals of group CURRENT_VOLTAGE_DIODE_LEADS_NXCALS
    """
    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: Optional[object] = None
                 ):
        """
        Initializes the CURRENT_VOLTAGE_DIODE_LEADS_NXCALS class object, inherits from DataAquistion.
        :param circuit_type: source directory of tdms files
        :param circuit_name: only convert the not yet converted files
        :param timestamp_fgc: number of processes for parallel conversion
        :param spark: number of processes for parallel conversion
        """
        super(CURRENT_VOLTAGE_DIODE_LEADS_NXCALS, self).__init__(circuit_type, circuit_name, timestamp_fgc)
        self.query_builder = RbCircuitQuery(self.circuit_type, self.circuit_name)
        self.duration = [(50, 's'), (350, 's')]
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """
        method to find correct timestamp for selected signal
        """
        return self.query_builder.find_source_timestamp_qds(self.timestamp_fgc, duration=self.duration)

    def get_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """
        signals = self.query_builder.query_current_voltage_diode_leads_nxcals(self.signal_timestamp, spark=self.spark,
                                                                           duration=self.duration)

        return [df.rename(columns={df.columns.values[0]:df.columns.values[0] + "_LEADS"}) for sublist in signals for df in sublist]