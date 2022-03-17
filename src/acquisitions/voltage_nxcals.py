from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from src.acquisition import DataAcquisition
from src.utils.utils import flatten_list
import pandas as pd
from typing import Optional, Union


class VOLTAGE_NXCALS(DataAcquisition):
    """
    Subclass of DataAquistion to query PC_PM
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: object
                 ):
        super(VOLTAGE_NXCALS, self).__init__(circuit_type, circuit_name, timestamp_fgc)
        self.signal_names = [['DIODE_RB', 'U_DIODE_RB'], ['VF', 'U_EARTH_RB']]
        self.query_builder = RbCircuitQuery(self.circuit_type, self.circuit_name)
        self.signal_timestamp = self.get_signal_timestamp()
        self.duration = [(50, 's'), (500, 's')]
        self.spark = spark

    def get_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """
        u_diode = self.query_builder.query_voltage_nxcals(self.signal_names[0][0], self.signal_names[0][1],
                                                          self.timestamp_fgc, spark=self.spark, duration=self.duration)
        u_earth = self.query_builder.query_voltage_nxcals(self.signal_names[1][0], self.signal_names[1][1],
                                                          self.timestamp_fgc, spark=self.spark, duration=self.duration)
        return u_diode + u_earth
