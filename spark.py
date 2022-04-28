from nxcals.spark_session_builder import get_or_create, Flavor
spark = get_or_create(flavor=Flavor.YARN_MEDIUM) # or any other flavor or dict with config
print("finished")