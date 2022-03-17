from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from src.acquisition import DataAcquisition
from src.utils.utils import flatten_list
import pandas as pd
from typing import Optional, Union


class EE_T_RES_PM(DataAcquisition):
    """
    Subclass of DataAquistion to query PC_PM
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: Optional[object] = None
                 ):
        super(EE_T_RES_PM, self).__init__(circuit_type, circuit_name, timestamp_fgc)
        self.systems = ['EE_ODD', 'EE_EVEN']
        self.signal_names = ['T_RES_BODY_1', 'T_RES_BODY_2', 'T_RES_BODY_3']
        self.query_builder = RbCircuitQuery(self.circuit_type, self.circuit_name)
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """
        method to find correct timestamp for selected signal
        """
        return self.query_builder.find_source_timestamp_ee(self.timestamp_fgc, system=self.systems).loc[0, 'timestamp']

    def get_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """
        return self.query_builder.query_ee_t_res_pm(self.signal_timestamp, self.timestamp_fgc, system=self.systems,
                                                    signal_names=self.signal_names)

    def get_reference_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """
        timestamp_fgc_ref = self.query_builder.get_timestamp_ref(col='fgcPm')
        signal_timestamp_ref = self.query_builder.find_source_timestamp_ee(timestamp_fgc_ref, system=self.systems).loc[
            0, 'timestamp']
        return self.query_builder.query_ee_t_res_pm(signal_timestamp_ref, timestamp_fgc_ref, system=self.systems,
                                                    signal_names=self.signal_names)