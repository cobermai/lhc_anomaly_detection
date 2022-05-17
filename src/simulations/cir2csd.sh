#!/usr/bin/bash

pspice_path=C:\\EDA\\Cadence\\OrCAD_166\\tools\\pspice\\psp_cmd.exe
cir_path=C:\\Users\\cobermai\\cernbox\\SWAN_projects\\lhc-anomaly-detection\\data\\PSPICE\\FPA_20210328_2209_final_Interpolation\\

# shellcheck disable=SC2164
cd "${cir_path}"

# shellcheck disable=SC1073
# shellcheck disable=SC1061
for i in *.cir;
do
  start "" "${pspice_path}" "${i}"
done