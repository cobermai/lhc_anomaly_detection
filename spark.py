from nxcals.spark_session_builder import get_or_create, Flavor
from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery
from src.acquisitions.voltage_nqps import VoltageNQPS

spark = get_or_create(flavor=Flavor.YARN_LARGE) # or any other flavor or dict with config

timestamp_fgc = 1436908751420000000

#query_builder = RbCircuitQuery("RB", "RB.A81")
#signals = query_builder.query_voltage_nxcals('DIODE_RB', 'U_DIODE_RB', timestamp_fgc, spark=spark, duration=[(50, 's'), (500, 's')])


fpa_identifier = {'circuit_type': 'RB',
                'circuit_name': 'RB.A81',
                'timestamp_fgc': 1436908751420000000}

group = VoltageNQPS(**fpa_identifier, spark=spark)
signals = group.get_signal_data()

print(signals)
