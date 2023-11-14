# Anomaly detection in the LHC RB circuit

Repository for analysis of fast power aborts in the LHC main dipole circuit with machine learning.


## Repository structure
```
├── data                                           > Context data used for analysis, big files are not stored on git
│ ├── RB_TC_extract_2021_11_22.xlsx                     > Unprocessed mp3 excel file with all FPA  
│ ├── RB_TC_extract_2021_11_22_processed.csv            > Cleaned mp3 file with fgc timestamps 
│ ├── RB_TC_extract_2021_11_22_processed_filled.csv     > Cleaned mp3 file with fgc timestamps & added features for simulation
│ ├── STEAM_context_data                                > Folder with data necessary for simulation with STEAM
│ └── transformation_overview.png                       > Picture with repository structure
├── notebooks                                      > Notebooks for code presentation
│ ├── RB_FPA_acquisition.ipynb                          > Notebook describing the data acquisition process
│ └── RB_FPA_event_analysis.ipynb                       > Notebook analyzing the mp3 excel file
├── references                                     > References used for analysis
│ ├── Local_TFM_analysis_report.pdf                     > Local Transfer Function Measurement Data Analysis
│ ├── Marvin_Thesis.pdf                                 > Framework for automatic superconducting magnet model generation 
│ │                                                       & validation against transients measured in LHCmagnets
│ ├── RB_QPS_Signals.png                                > Picture of QPS signals in RB circuit
│ └── RB_circuit.png                                    > Picture of RB circuit
├── scripts                                        > Scripts to execute for user
│ ├── download_rb_data.py                               > Script to download data
│ ├── simulate_rb_data.py                               > Script to simulate data
│ └── simulation_converter.py                           > Script to convert simulated .cir file to .hdf5
├── src                                            > Source files, used by the scripts and the notebooks
│ ├── acquisition.py                                    > Abstract acquisition class
│ ├── acquisitions                                      > Detailed acquisition files
│ ├── simulation.py                                     > Simulation file (from Marvin, to be outsourced)
│ ├── simulations                                       > Functions for simulation (from Marvin, to be outsourced)
│ ├── utils                                             > Utility functions hdf_tools, mp3_excel_processing, utils
│ └── visualisation                                     > Functions for visualisation
├── requirements.txt                               > Package requirements
└── README.md                                      > Git readme
```

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


### Generate venv requirements
```
pip install virtualenv
virtualenv venv 
source ./venv/bin/activate
pip install -r requirements.txt
```

# Transformation
The transformation module contains scripts to download and simulate signals in the LHC main dipole circuit.
![alt text](data/transformation_overview.png)

## Acquisition
Access to the AccPy repo is necessary for installing NXCALS. Execution only on Linux.

### Run code outside SWAN:
Install necessary packages:
```
pip install git+https://gitlab.cern.ch/acc-co/devops/python/acc-py-pip-config.git
pip install nxcals
pip install lhcsmapi==1.5.20
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

## Simulation
Execution only on Windows, as PSpice is utilized.
### Using steam sing
* Clone steam-notebooks directory: https://gitlab.cern.ch/steam/steam-notebooks.git and specify path bellow (commit sha: e591c2ebc6ea191fa8ed240cf7cc7361d1a3fae4)
* Required Programs: PSPICE, COSIM, and LEDET
* Contact emmanuele.ravaioli@cern.ch to get get access to COSIM and LEDET
* Clone the steam project "gitlab.cern.ch/steam/steam-models-dev" and provide the link to the folder in the script to execute
* Create personal configurations in "data\STEAM_context_data\analysisSTEAM_example_RB.yaml" and provide the link to the folder in the script to execute

### Update project
In the the "steam-models-dev\steam_models" folder needs to be updated seperately with git. 
After the first clone of the main dir, do:
* git submodule init
* git submodule update
If changes were made, e.g. to change the signals to simulate in "team_models/circuits/RB/modelData_RB.yaml", changes can be stashed:
* git stash
* git push
* git stash apply

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
