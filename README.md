# Anomaly detection in the LHC RB circuit

Analysis of the LHC RB circuit with ML.

### Generate venv requirements
Generate venv requirements with specific python version. --python is optional
```
virtualenv --python=/usr/bin/python3.9 venv 
.\venv\Scripts\activate
pip install -r requirements.txt
```


## Hints
### Generate requirements
```
pip install pipreqs
pipreqs
```

### Autoformat python file
```
autopep8 --in-place --aggressive --aggressive <filename>
```