from typing import Optional, Union

import pandas as pd
from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from pyspark.sql import SparkSession

from src.acquisition import DataAcquisition


class EEUDumpResPM(DataAcquisition):
    """
    Specifies method to query data for signals of group EEUDumpResPM
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: Optional[SparkSession] = None
                 ):
        """
        Initializes the EEUDumpResPM class object, inherits from DataAcquisition.
        :param circuit_type: lhc circuit name
        :param circuit_name: lhc sector name
        :param timestamp_fgc: fgc event timestamp
        :param spark: spark object to query data from NXCALS
        """
        super().__init__(circuit_type, circuit_name, timestamp_fgc)
        self.systems = ['EE_ODD', 'EE_EVEN']
        self.signal_names = ['U_DUMP_RES']
        self.query_builder = RbCircuitQuery(
            self.circuit_type, self.circuit_name)
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame, list]:
        """ method to find correct timestamp for selected signal """
        timestamps = []
        for system in self.systems:
            signal_timestamp_odd = self.query_builder.find_source_timestamp_ee(
                self.timestamp_fgc, system=system)
            timestamps.append(signal_timestamp_odd.loc[0, 'timestamp'])
        return timestamps

    def get_signal_data(self) -> list:
        """ method to get selected signal with specified sigmon query builder and signal timestamp  """
        dfs = []
        for i, system in enumerate(self.systems):
            df = self.query_builder.query_ee_u_dump_res_pm(
                self.signal_timestamp[i],
                self.timestamp_fgc,
                system=system,
                signal_names=self.signal_names)
            dfs.append(df)
        return dfs
