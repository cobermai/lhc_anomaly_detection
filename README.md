# Anomaly detection in the LHC RB circuit

Analysis of the LHC RB circuit with ML.

### Generate venv requirements
Generate venv requirements with specific python version. --python is optional
```
pip install virtualenv
virtualenv venv 
source ./venv/bin/activate
pip install -r requirements.txt
```

## Hints
### Install requirements
Access to the AccPy repo is necessary for installing NXCALS.

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
```
Start spark session:
```
from nxcals.spark_session_builder import get_or_create, Flavor
spark = get_or_create(flavor=Flavor.YARN_MEDIUM) # or any other flavor or dict with config
```
### Create requirements
```
pip install pipreqs
pipreqs
```

### Autoformat python file
```
autopep8 --in-place --aggressive --aggressive <filename>
```
