from typing import Optional, Union

import pandas as pd
from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from lhcsmapi.api import query,resolver,  processing
from pyspark.sql import SparkSession
from lhcsmapi.Time import Time

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
        dfs = []
        for i, system in enumerate(self.systems):
            df = self.query_builder.query_ee_u_dump_res_pm(
                self.signal_timestamp[i],
                self.timestamp_fgc,
                system=system,
                signal_names=self.signal_names)
                
            duration = (60, 's')
            timestamp = self.timestamp_fgc-50000000

            nxcals_query_params = resolver.get_params_for_nxcals(self.circuit_type, self.circuit_name, 'PIC', timestamp, duration, signals=['ST_ABORT_PIC'])
            # nxcals_query_params are instance of VariableQueryParams
            nxcals_data = query.query_nxcals_by_variables(self.spark, nxcals_query_params)
            
            odd_off = (Time.to_pandas_timestamp(nxcals_data[0].index.values[0])-Time.to_pandas_timestamp(self.timestamp_fgc)).total_seconds()
            even_off = (Time.to_pandas_timestamp(nxcals_data[1].index.values[0])-Time.to_pandas_timestamp(self.timestamp_fgc)).total_seconds()
            if 'ODD' in system:
                df.index = df.index - odd_off
            if 'EVEN' in system:
                df.index = df.index - even_off
            dfs.append(df)
        return dfs