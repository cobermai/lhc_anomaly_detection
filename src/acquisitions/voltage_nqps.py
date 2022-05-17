from typing import Union

import pandas as pd
from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery

from src.acquisition import DataAcquisition
from src.utils.utils import flatten_list


class VoltageNQPS(DataAcquisition):
    """
    Specifies method to query data for signals of group VoltageNQPS
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: object
                 ):
        """
        Initializes the VoltageNQPS class object, inherits from DataAcquisition.
        :param circuit_type: lhc circuit name
        :param circuit_name: lhc sector name
        :param timestamp_fgc: fgc event timestamp
        :param spark: spark object to query data from NXCALS
        """
        super().__init__(circuit_type, circuit_name, timestamp_fgc)
        self.query_builder = RbCircuitQuery(
            self.circuit_type, self.circuit_name)
        self.duration = [(50, 's'), (500, 's')]
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """ method to find correct timestamp for selected signal """
        source_timestamp_qds_df = self.query_builder.find_source_timestamp_qds(
            self.timestamp_fgc, duration=self.duration)
        source_timestamp_nqps_df = self.query_builder.find_source_timestamp_nqps(
            self.timestamp_fgc, warn_on_missing_pm_buffers=True)
        return [source_timestamp_qds_df, source_timestamp_nqps_df]

    def get_signal_data(self) -> list:
        """ query_voltage_nqps compares qds and nqps timestamp. Then queries from NXCALS or PM  """
        u_nqps_dfs = self.query_builder.query_voltage_nqps(
            self.signal_timestamp[1],
            self.signal_timestamp[0],
            self.timestamp_fgc,
            spark=self.spark)

        return flatten_list(u_nqps_dfs)
