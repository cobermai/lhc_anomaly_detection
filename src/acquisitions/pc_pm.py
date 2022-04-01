from typing import Optional

from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery

from src.acquisition import DataAcquisition


class PCPM(DataAcquisition):
    """
    Specifies method to query data for signals of group PCPM
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: Optional[object] = None
                 ):
        """
        Initializes the EETResPM class object, inherits from DataAcquisition.
        :param circuit_type: lhc circuit name
        :param circuit_name: lhc sector name
        :param timestamp_fgc: fgc event timestamp
        :param spark: spark object to query data from NXCALS
        """
        super().__init__(circuit_type, circuit_name, timestamp_fgc)
        self.signal_names = [
            'I_MEAS',
            'I_A',
            'I_EARTH',
            'I_EARTH_PCNT',
            'I_REF']
        self.query_builder = RbCircuitQuery(
            self.circuit_type, self.circuit_name)
        self.signal_timestamp = self.get_signal_timestamp()
        self.spark = spark

    def get_signal_data(self) -> list:
        """ method to get selected signal with specified sigmon query builder and signal timestamp  """
        return self.query_builder.query_pc_pm(
            self.signal_timestamp,
            self.signal_timestamp,
            signal_names=self.signal_names)

    def get_reference_signal_data(self) -> list:
        """ method to get selected reference signal with specified sigmon query builder and signal timestamp  """
        timestamp_fgc_ref = self.query_builder.get_timestamp_ref(col='fgcPm')
        return self.query_builder.query_pc_pm(
            timestamp_fgc_ref,
            timestamp_fgc_ref,
            signal_names=self.signal_names)
