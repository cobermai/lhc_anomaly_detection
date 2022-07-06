from typing import Union

import pandas as pd
from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from lhcsmapi.Time import Time
from pyspark.sql import SparkSession
from lhcsmapi.Time import Time
import numpy as np

from src.acquisition import DataAcquisition
from src.utils.utils import flatten_list


class VoltageLogicIQPS(DataAcquisition):
    """
    Specifies method to query data for signals of group VoltageNQPSFPACrate
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: SparkSession
                 ):
        """
        Initializes the VoltageLogicIQPS class object, inherits from DataAcquisition.
        :param circuit_type: lhc circuit name
        :param circuit_name: lhc sector name
        :param timestamp_fgc: fgc event timestamp
        :param spark: spark object to query data from NXCALS
        """
        super().__init__(circuit_type, circuit_name, timestamp_fgc)
        self.query_builder = RbCircuitQuery(
            self.circuit_type, self.circuit_name)
        self.duration = [(10, 's'), (10, 's')]
        self.signal_names = ['U_QS0', 'U_1', 'U_2']
        self.timestamp_fgc = timestamp_fgc
        self.signal_timestamp = self.get_signal_timestamp()
        self.timestamp_fgc = timestamp_fgc
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame, list]:
        """ method to find correct timestamp for selected signal """
        source_timestamp_qds_df = self.query_builder.find_source_timestamp_qds_board_ab(
            self.timestamp_fgc, duration=self.duration)
        source_timestamp_qds_df.drop_duplicates(subset=['source', 'timestamp'], inplace=True)
        source_timestamp_qds_df.reset_index(drop=True, inplace=True)
        iqps_board_type_df = self.query_builder.query_pm_iqps_board_type(source_timestamp_qds_df=source_timestamp_qds_df)
        source_timestamp_qds_df['iqps_board_type'] = iqps_board_type_df['iqps_board_type']
        return source_timestamp_qds_df

    def include_iqps_board_type(self, signals) -> list:
        """ method to re-include the iqps_board_type into the signal names"""
        # loop through all timestamps
        for i in range(len(self.signal_timestamp['iqps_board_type'])):
            # loop through all signals per timestamp
            for k in range(len(signals[i])):
                source = self.signal_timestamp.loc[i, 'source']
                if self.signal_timestamp.loc[i, 'iqps_board_type'] == '0':
                    appendix = 'A'
                elif self.signal_timestamp.loc[i, 'iqps_board_type'] == '1':
                    appendix = 'B'
                name = f'{source}_{appendix}:{self.signal_names[k]}'
                signals[i][k].columns = [name]
        return signals

    def get_signal_data(self) -> list:
        """ method to get selected signal with specified sigmon query builder and signal timestamp  """
        signals = self.query_builder.query_voltage_logic_iqps(source_timestamp_qds_df=self.signal_timestamp,
                                                              signal_names=self.signal_names, filter_window=3)
        signals = self.include_iqps_board_type(signals)
        signals = flatten_list(signals)

        count = 0
        c = 0
        offsets = []
        for i in range(len(self.signal_timestamp)*len(self.signal_names)):
            offsets.append(signals[i].index.values[0])
        offset = -1*np.min(abs(np.array(offsets)))
        
        for i in range(len(self.signal_timestamp)*3):
            t_fgc = float(self.timestamp_fgc)
            q_fgc = float(self.signal_timestamp.loc[count, 'timestamp'])
            a = Time.to_pandas_timestamp(q_fgc)
            b = Time.to_pandas_timestamp(t_fgc)
            off = (b-a).total_seconds()
            
            start = signals[i].index.values[0]
            signals[i].index = signals[i].index - off + (offset-start)
            
            c = c+1
            if c==3:
                c=0
                count = count +1
        return signals
