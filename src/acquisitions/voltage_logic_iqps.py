from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from src.acquisition import DataAcquisition
from src.utils.utils import flatten_list
import pandas as pd
from typing import Optional, Union


class VOLTAGE_LOGIC_IQPS(DataAcquisition):
    """
    Subclass of DataAquistion to query PC_PM
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: Optional[object] = None
                 ):
        super(VOLTAGE_LOGIC_IQPS, self).__init__(circuit_type, circuit_name, timestamp_fgc)
        self.signal_names = ['U_QS0', 'U_1', 'U_2', 'ST_NQD0', 'ST_MAGNET_OK']
        self.query_builder = RbCircuitQuery(self.circuit_type, self.circuit_name)
        self.duration = [(50, 's'), (500, 's')]
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
        signals = self.query_builder.query_voltage_logic_iqps(self.signal_timestamp, signal_names=self.signal_names)
        return flatten_list(signals)

