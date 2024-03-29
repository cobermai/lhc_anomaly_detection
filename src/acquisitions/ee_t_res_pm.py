from typing import Optional, Union

import pandas as pd
from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from pyspark.sql import SparkSession

from src.acquisition import DataAcquisition


class EETResPM(DataAcquisition):
    """ Specifies method to query data for signals of group EETResPM """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: Optional[SparkSession] = None
                 ):
        """
        Initializes the EETResPM class object, inherits from DataAcquisition.
        :param circuit_type: lhc circuit name
        :param circuit_name: lhc sector name
        :param timestamp_fgc: fgc event timestamp
        :param spark: spark object to query data from NXCALS
        """
        super().__init__(circuit_type, circuit_name, timestamp_fgc)
        self.systems = ['EE_ODD', 'EE_EVEN']
        self.signal_names = ['T_RES_BODY_1', 'T_RES_BODY_2', 'T_RES_BODY_3']
        self.query_builder = RbCircuitQuery(
            self.circuit_type, self.circuit_name)
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame, list]:
        """ method to find correct timestamp for selected signal """
        timestamp_df = self.query_builder.find_source_timestamp_ee(self.timestamp_fgc, system=self.systems)
        if not timestamp_df.empty:
            return timestamp_df.loc[0, 'timestamp']
        else:
            return 0

    def get_signal_data(self) -> list:
        """ method to get selected signal with specified sigmon query builder and signal timestamp  """
        return self.query_builder.query_ee_t_res_pm(
            self.signal_timestamp,
            self.timestamp_fgc,
            system=self.systems,
            signal_names=self.signal_names)

    def get_reference_signal_data(self) -> list:
        """ method to get selected reference signal with specified sigmon query builder and signal timestamp  """
        timestamp_fgc_ref = self.query_builder.get_timestamp_ref(col='fgcPm')
        signal_timestamp_ref = self.query_builder.find_source_timestamp_ee(
            timestamp_fgc_ref, system=self.systems).loc[0, 'timestamp']
        return self.query_builder.query_ee_t_res_pm(
            signal_timestamp_ref,
            timestamp_fgc_ref,
            system=self.systems,
            signal_names=self.signal_names)
