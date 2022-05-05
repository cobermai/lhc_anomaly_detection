from nxcals.spark_session_builder import get_or_create, Flavor
from lhcsmapi.analysis.RbCircuitQuery import RbCircuitQuery


spark = get_or_create(flavor=Flavor.YARN_LARGE) # or any other flavor or dict with config

timestamp_fgc = 1619196160600000000

query_builder = RbCircuitQuery("RB", "RB.A12")
signals = query_builder.query_voltage_nxcals('DIODE_RB', 'U_DIODE_RB', timestamp_fgc, spark=spark, duration=[(50, 's'), (500, 's')])

print(signals)
