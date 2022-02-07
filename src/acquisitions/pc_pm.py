from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from src.acquisition import DataAcquisition
import pandas as pd
from typing import Optional, Union

class PC_PM(DataAcquisition):
    """
    Subclass of DataAquistion to query Power Converter (PC) signals from Post Mortem (PM)
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: Optional[object] = None
                 ):
        super(PC_PM, self).__init__(circuit_type, circuit_name, timestamp_fgc)
        self.signal_names = ['I_MEAS', 'I_A', 'I_EARTH', 'I_EARTH_PCNT', 'I_REF']
        self.query_builder = RbCircuitQuery(self.circuit_type, self.circuit_name)
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """
        return self.query_builder.query_pc_pm(self.signal_timestamp, self.signal_timestamp,
                                              signal_names=self.signal_names)

    def get_reference_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """
        timestamp_fgc_ref = self.query_builder.get_timestamp_ref(col='fgcPm')
        return self.query_builder.query_pc_pm(timestamp_fgc_ref, timestamp_fgc_ref, signal_names=self.signal_names)



