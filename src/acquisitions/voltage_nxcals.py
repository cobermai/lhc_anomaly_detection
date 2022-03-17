from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from src.acquisition import DataAcquisition


class VoltageNXCALS(DataAcquisition):
    """
    Specifies method to query data for signals of group VoltageNXCALS
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 spark: object
                 ):
        """
        Initializes the VoltageNXCALS class object, inherits from DataAcquisition.
        :param circuit_type: lhc circuit name
        :param circuit_name: lhc sector name
        :param timestamp_fgc: fgc event timestamp
        :param spark: spark object to query data from NXCALS
        """
        super(
            VoltageNXCALS,
            self).__init__(
            circuit_type,
            circuit_name,
            timestamp_fgc)
        self.signal_names = [['DIODE_RB', 'U_DIODE_RB'], ['VF', 'U_EARTH_RB']]
        self.query_builder = RbCircuitQuery(
            self.circuit_type, self.circuit_name)
        self.signal_timestamp = self.get_signal_timestamp()
        self.duration = [(50, 's'), (500, 's')]
        self.spark = spark

    def get_signal_data(self) -> list:
        """
        method to get selected signal with specified sigmon query builder and signal timestamp
        """
        u_diode = self.query_builder.query_voltage_nxcals(
            self.signal_names[0][0],
            self.signal_names[0][1],
            self.timestamp_fgc,
            spark=self.spark,
            duration=self.duration)
        u_earth = self.query_builder.query_voltage_nxcals(
            self.signal_names[1][0],
            self.signal_names[1][1],
            self.timestamp_fgc,
            spark=self.spark,
            duration=self.duration)
        return u_diode + u_earth
