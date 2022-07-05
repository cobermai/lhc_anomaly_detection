from typing import Union

import pandas as pd
from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from pyspark.sql import SparkSession

from src.acquisition import DataAcquisition
from src.utils.utils import flatten_list


class VoltageNQPS(DataAcquisition):
    """
    Specifies method to query data for signals of group VoltageNQPS.
    If a nQps crate is triggered it stores data, which is then saved to PM in high resolution.
    There are 3 magnets per nQps crate + reference magnet.
    A secondary quench magnet on the same nQps crate cannot save data as the nQps buffer is already full.
    However if a secondary quenched magnet is on another nQps crate, high resolution data is stored.
    query_voltage_nqps returns a list of lists, where each list contains dataframes from on
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: SparkSession
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
        pm_data = []
        for index, row in self.signal_timestamp.iterrows():
            wildcard = {'MAGNET': '*', 'QPS_CRATE': row['source']}
            params = resolver.get_params_for_pm_signals('RB', self.circuit_name, 'DIODE_RB', row['timestamp'],
                                                        signals=self.signals, wildcard=wildcard)
            data = query.query_pm_signals_with_resolved_params(params)

            data_processed = processing.SignalProcessing(data).synchronize_time(
                row['timestamp']).convert_index_to_sec().get_dataframes()
            if isinstance(data_processed, list):
                for d in data_processed:
                    pm_data.append(d)
            else:
                pm_data.append(data_processed)

        return pm_data
