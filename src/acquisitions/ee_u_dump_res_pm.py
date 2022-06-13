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

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """ method to find correct timestamp for selected signal """
        signal_timestamp = self.query_builder.find_source_timestamp_ee(
            self.timestamp_fgc, system=self.systems)
        return signal_timestamp


    def get_signal_data(self) -> list:
        """ method to get selected signal with specified sigmon query builder and signal timestamp  """
        signal_timestamp_odd = self.query_builder.find_source_timestamp_ee(
            self.timestamp_fgc, system=self.systems[0])
        signal_timestamp_ref_odd = self.signal_timestamp_odd.loc[0, 'timestamp']
        signal_timestamp_even = self.query_builder.find_source_timestamp_ee(
            self.timestamp_fgc, system=self.systems[1])
        signal_timestamp_ref_even = self.signal_timestamp_even.loc[0, 'timestamp']

        U_dump_res_odd = self.query_builder.query_ee_u_dump_res_pm(
            signal_timestamp_ref_odd,
            self.timestamp_fgc,
            system=self.systems[0],
            signal_names=self.signal_names)

        U_dump_res_even = self.query_builder.query_ee_u_dump_res_pm(
            signal_timestamp_ref_even,
            self.timestamp_fgc,
            system=self.systems[1],
            signal_names=self.signal_names)

        return U_dump_res_odd + U_dump_res_even