# Anomaly detection in the LHC RB circuit

Repository for analysis of fast power aborts in the LHC main dipole circuit with machine learning.

## Data Sources

| File                                                                                             | Description                           | Author             | Last Update | Source                                                                                                                                                        |
|--------------------------------------------------------------------------------------------------|---------------------------------------|--------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MB_feature_analysis/analysis_conductor_fieldQuality_FPA2021_FTM2023_year2021_FPA2021_TFM2023.csv | Context data of RB magnets            | Emmanuele Ravaioli | 19.05.2023  | [MB feature analysis data](https://gitlab.cern.ch/machine-protection/mb-feature-classification/-/tree/main?ref_type=heads)                                    |
| MB_feature_analysis/RB_Beamscreen_Resistances.csv                                                | Beamscreen resistance of RB magnets   | Marvin Janitschke  | 29.08.2022  | [MB feature analysis data](https://gitlab.cern.ch/machine-protection/mb-feature-classification/-/tree/main?ref_type=heads)                                    |
| MP3_context_data/RB_TC_extract_2023_03_13.xlsx                                                   | Context data of RB quenches           | MP3                | 03.13.2023  | [MP3 database](https://social.cern.ch/community/MP3-onedrive/_layouts/15/WopiFrame.aspx?sourcedoc=/community/MP3-onedrive/QuenchData/RB.xlsx&action=default)  |
| MP3_context_data/snapshot_timestamps.json                                                        | Timestamps of snapshot tests          | MP3                | 03.13.2023  | [MP3 twiki](https://twiki.cern.ch/twiki/bin/viewauth/MP3/FPAinRB)                                                                                             |
| SIGMON_context_data/nQPS_RB_busBarProtectionDetails.csv                                          | Context data of nQPS Busbars          | SIGMON             | 08.09.2022  | [SIGMON busbar metadata](https://gitlab.cern.ch/LHCData/lhc-sm-api/-/blob/dev/lhcsmapi/metadata/busbar/nQPS_RB_busBarProtectionDetails.csv)                   |
| SIGMON_context_data/RB_CrateToDiodeMap.csv                                                       | Map of QPS crates to diode position   | SIGMON             | 08.05.2023  | [SIGMON QPS crate metadata](https://gitlab.cern.ch/LHCData/lhc-sm-api/-/blob/dev/lhcsmapi/metadata/qps_crate/RB_CrateToDiodeMap.csv)                          |
| SIGMON_context_data/RB_LayoutDetails.csv                                                         | Context data of RB circuit            | SIGMON             | 08.09.2022  | [SIGMON magnet_metadata](https://gitlab.cern.ch/LHCData/lhc-sm-api/-/blob/dev/lhcsmapi/metadata/magnet/RB_LayoutDetails.csv)                                  |

## Execution

Execution is possible with scripts or with notebooks.
Access to the AccPy repo is necessary for using the package lhcsmapi. Execution only on Linux, with or outside SWAN.

### Run code with SWAN:
```
Software stack: NXCALS PRO PyTimber PRO
Platform: CentOS 7 (gcc11)
Environment script: /eos/project/l/lhcsm/public/1.6.1.sh
Number of cores: 4
Memory: 10 GB
Spark cluster: BE NXCALS (NXCals)
```

Further documentation available [Here](https://gitlab.cern.ch/LHCData/lhc-sm-hwc).


### Run code outside SWAN:
Install necessary packages:
```
pip install git+https://gitlab.cern.ch/acc-co/devops/python/acc-py-pip-config.git
pip install nxcals
pip install lhcsmapi==1.6.1
```
Open mandatory ports and extend runtime:
```
sudo firewall-cmd --add-port=5001/tcp
sudo firewall-cmd --add-port=5101/tcp
sudo firewall-cmd --add-port=5201/tcp
sudo firewall-cmd --runtime-to-permanent
firewall-cmd --list-ports
```
Start spark session:
```
from nxcals.spark_session_builder import get_or_create, Flavor
spark = get_or_create(flavor=Flavor.YARN_MEDIUM) # or any other flavor or dict with config
```
Further documentation available [Here](https://gitlab.cern.ch/LHCData/lhc-sm-api).

# Hints
### Create requirements
```
pip install pipreqs
pipreqs
```

### Autoformat python file
```
autopep8 --in-place --aggressive --aggressive <filename>
```