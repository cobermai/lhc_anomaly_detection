from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from src.acquisition import DataAcquisition
import pandas as pd
from typing import Optional, Union


class QH_PM(DataAcquisition):
    """
    Subclass of DataAquistion to query PC_PM
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: Optional[object] = None
                 ):
        super(QH_PM, self).__init__(circuit_type, circuit_name, timestamp_fgc)
        self.signal_names = ['I_HDS', 'U_HDS']
        self.query_builder = RbCircuitQuery(self.circuit_type, self.circuit_name)
        self.duration = [(10, 's'), (500, 's')]
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """
        method to find correct timestamp for selected signal
        """
        return self.query_builder.find_source_timestamp_qh(self.timestamp_fgc, duration=self.duration)

    def get_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """
        signals = self.query_builder.query_qh_pm(self.signal_timestamp, signal_names=self.signal_names)
        return self.flatten_list(signals)

    def get_reference_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """
        signals = self.query_builder.query_qh_pm(self.signal_timestamp, signal_names=self.signal_names, is_ref=True)
        return self.flatten_list(signals)