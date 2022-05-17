from pathlib import Path
import os
from py4j.java_gateway import launch_gateway, java_import, JavaGateway, JavaObject, GatewayParameters, Py4JNetworkError
from steam_nb_api.utils.STEAMLib_simulations import *
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import numpy as np
from steam_nb_api.utils import arrays as a
import shutil
import copy

import matplotlib.pyplot as pltp
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

from src.simulations.rb_utils import *

Config_Name = ''

LEDET_only = 0  # If 1, the quenches will be simulated in LEDET only
Skip_SetUp = 0  # If set to 1, the set-up will be skipped and you jump straigt to the stitching & results
Interpolation_only = 1  # If set to 1, the set-up will be skipped and results will be obtained by interpolation of previously obtained results
InterpolationType = 'Linear'  # Supported: 'Spline', 'Linear'

# Half-Turns here are arbitrary for now --> Waiting for Zinur
enableQuench = 1  # 0 = no quenches included, 1 yes
quenchHalfTurn_EXT = 40  # Turn set to quench if quench origin = EXT
quenchHalfTurn_INT = 80  # Turn set to quench if quench origin = INT

tEnd = 300
Opts = Options()
Opts.t_0 = [0.000, 0.12, 0.20, 0.3, 0.5]
Opts.t_end = [0.12, 0.20, 0.30, 0.5, 1.1]
Opts.t_step_max = [[1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4]] + [[1.0e-5, 5.0e-5, 1.0e-4, 1.0e-4, 1.0e-4]]
Opts.relTolerance = [8e-4] + [None]
Opts.absTolerance = [5] + [None]
Opts.executionOrder = [1] + [2]
Opts.executeCleanRun = [True, True]

sparseTimeStepping = 100

# Launch a Gateway in a new Java process, this returns port
port = launch_gateway(classpath='../../steam/*')
# JavaGateway instance is connected to a Gateway instance on the Java side
gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
# Get STEAM API Java classes
MutualInductance = gateway.jvm.component.MutualInductance
Netlist = gateway.jvm.netlist.Netlist
CommentElement = gateway.jvm.netlist.elements.CommentElement
GeneralElement = gateway.jvm.netlist.elements.GeneralElement
ACSolverElement = gateway.jvm.netlist.solvers.ACSolverElement
StimulusElement = gateway.jvm.netlist.imports.StimulusElement
ParameterizedElement = gateway.jvm.netlist.elements.ParameterizedElement
GlobalParameterElement = gateway.jvm.netlist.elements.GlobalParameterElement
OutputGeneralElement = gateway.jvm.netlist.elements.OutputGeneralElement
OptionSolverSettingsElement = gateway.jvm.netlist.solvers.OptionSolverSettingsElement
TransientSolverElement = gateway.jvm.netlist.solvers.TransientSolverElement
AutoconvergeSolverSettingsElement = gateway.jvm.netlist.solvers.AutoconvergeSolverSettingsElement
CircuitalPreconditionerSubcircuit = gateway.jvm.preconditioner.CircuitalPreconditionerSubcircuit
TextFile = gateway.jvm.utils.TextFile
CSVReader = gateway.jvm.utils.CSVReader

## find position
def findRBposition(Layout_db, pos, Circuit):
    mag_pos = 'MB.' + pos
    pos_db = pd.read_excel(Layout_db)
    idx = pos_db.index[pos_db['ASSEMBLY_NAME'] == mag_pos][1]
    if pos_db['CIRCUIT_NAME'][idx] != Circuit:
        print('Something went wrong. Position error')
    el_pos = int(pos_db['ELECTRICAL POSITION'][idx])
    return el_pos


## Adjust times in generic stimuli
def changeRBStimuli(StimulusFile, t_EE_1, t_EE_2, EE1=0, EE2=0):
    tempStimulusFile = os.path.join(os.getcwd(), 'tempDir//Gate-EE_Stimuli_TEMP.stl')
    stlString = ''
    with open(StimulusFile, 'r') as file:
        with open(tempStimulusFile, 'w') as tfile:
            for line in file:
                repl = 0
                if 't_EE_1' in line:
                    repl = 1
                    st = 't_EE_1'
                    time = t_EE_1 / 1000
                    if EE1: time = 0.00
                if 't_EE_2' in line:
                    repl = 1
                    st = 't_EE_2'
                    time = t_EE_2 / 1000
                    if EE2: time = 0.00
                if repl:
                    idx_1 = line.find(st) + 7
                    count = 0
                    end = 0
                    k = 0

                    while not end:
                        if line[k + idx_1] == '*':
                            end = 1
                        if (line[k + idx_1] == '+' and count != 0) or (end == 1 and count != 0):
                            if line[idx_1] == '+': idx_1 = idx_1 + 1
                            time = time + float(line[idx_1:idx_1 + count])
                            idx_1 = idx_1 + count
                            count = 0
                            k = 0
                            continue
                        if line[k + idx_1].isdigit() or line[k + idx_1] == '.':
                            count = count + 1
                        k = k + 1
                    repl_idx1 = line.find(st)
                    repl_idx2 = line[repl_idx1:].find('**') + repl_idx1
                    line = line.replace(line[repl_idx1:repl_idx2], str(time))
                stlString = stlString + line
            tfile.write(stlString)
    return tempStimulusFile


## Generate Quench circuit for spec. position
def generateCircuitFile(circuit_name, position_quenching_magnet, R_EE_1, R_EE_2, elPositions=[],
                        libraryPath="C:\\cernbox\\steam-pspice-library\\", FinalRun=0):
    # Launch a Gateway in a new Java process, this returns port
    port = launch_gateway(classpath='../../steam/*')
    # JavaGateway instance is connected to a Gateway instance on the Java side
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
    # Get STEAM API Java classes
    MutualInductance = gateway.jvm.component.MutualInductance

    Netlist = gateway.jvm.netlist.Netlist
    CommentElement = gateway.jvm.netlist.elements.CommentElement
    GeneralElement = gateway.jvm.netlist.elements.GeneralElement
    ParameterizedElement = gateway.jvm.netlist.elements.ParameterizedElement
    GlobalParameterElement = gateway.jvm.netlist.elements.GlobalParameterElement
    OutputGeneralElement = gateway.jvm.netlist.elements.OutputGeneralElement
    TextFile = gateway.jvm.utils.TextFile
    CSVReader = gateway.jvm.utils.CSVReader

    if not isinstance(position_quenching_magnet, list): position_quenching_magnet = [position_quenching_magnet]
    if elPositions: position_quenching_magnet = elPositions
    pathRB = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'RB')
    CircuitParamInputPath = "RB_Circuit_Param_Table.csv"
    r1r2CSVInputPath = os.path.join(pathRB, "RB_R1R2_Sector" + circuit_name[-2:] + "_Table.csv")
    gndCSVInputPath = os.path.join(pathRB, "RB_Gnd_Table.csv")
    voltFeelersCSVInputPath = os.path.join(pathRB, "RB_VoltageFeelers_Table.csv")

    N_MAGS = 154
    INDEX_OUT_NODE = 3

    netlistPath = "netlist.cir"
    netlist = Netlist(netlistPath)

    # Set paths to libraries
    libraryPaths = ["\"" + libraryPath + "RB\\Items\\RB_Diodes.lib\"",
                    "\"" + libraryPath + "RB\\Items\\RB_Thyristors.lib\"",
                    "\"" + libraryPath + "RB\\Items\\RB_Switches.lib\"",
                    "\"" + libraryPath + "RB\\Items\\RB_PC.lib\"",
                    "\"" + libraryPath + "RB\\Items\\RB_MB.lib\"",
                    "\"" + libraryPath + "RB\\Items\\RB_EE.lib\"",
                    "\"" + libraryPath + "magnet\\Items\\magnets_cosimulation.lib\""]

    netlist.setLibraryPaths(a.convert_list_to_string_array(gateway, libraryPaths))
    # Set global parameters
    # Looks for circuit in csv and includes the parameters
    csv = CSVReader(CircuitParamInputPath, ",")
    vecIn = csv.read()
    globalParameters_Parameters = vecIn.get(0).split(csv.getCsvSplitBy())[4:35]

    print(globalParameters_Parameters)
    for row in range(len(vecIn)):
        globalParameters_Val = vecIn.get(row).split(csv.getCsvSplitBy())[0]
        if (globalParameters_Val == circuit_name):
            break;

    N_MAGS = int(float(vecIn.get(row).split(csv.getCsvSplitBy())[5]))
    MagnetName = str(vecIn.get(row).split(csv.getCsvSplitBy())[1])
    t_PC = float(vecIn.get(row).split(csv.getCsvSplitBy())[7])
    globalParameters_Values = vecIn.get(row).split(csv.getCsvSplitBy())[5:36]
    print(f"globalParameters_Values: {globalParameters_Values}")
    globalParameters_Parameters = a.create_string_array(gateway, globalParameters_Parameters)
    globalParameters_Values = a.create_string_array(gateway, globalParameters_Values)

    netlist.add(CommentElement("**** Global parameters ****"))
    netlist.add(GlobalParameterElement(globalParameters_Parameters, globalParameters_Values))
    # Two PCs in parallel
    netlist.add(CommentElement("*"))
    netlist.add(CommentElement("* Two PCs in parallel"))
    netlist.add(GeneralElement("x1_PC", a.create_string_array(gateway, ("1", "2")), "RB_PC_Full"))

    netlist.add(GeneralElement("v1_filterPH", a.create_string_array(gateway, ("2", "3")), "0"))
    netlist.add(GeneralElement("v2_filterPH", a.create_string_array(gateway, ("21", "1")), "0"))

    # HTS lead 1 HOT-COLD
    netlist.add(CommentElement("* HTS lead 1 HOT-COLD"))
    netlist.add(GeneralElement("r1_warm", a.create_string_array(gateway, ("3", "4")), "{R1_Warm}"))
    netlist.add(GeneralElement("v1_warm", a.create_string_array(gateway, ("4", "5")), "{V1_Warm}"))
    netlist.add(GeneralElement("l1_warm", a.create_string_array(gateway, ("5", "6")), "{L1_Warm}"))
    netlist.add(GeneralElement("v1_fake", a.create_string_array(gateway, ("6", "MAG1")), "0"))

    # HTS lead 2 COLD-HOT
    netlist.add(CommentElement("* HTS lead 2 COLD-HOT"))
    netlist.add(GeneralElement("v2_fake", a.create_string_array(gateway, ("MAG77_Out", "7")), "0"))
    netlist.add(GeneralElement("r2_warm", a.create_string_array(gateway, ("7", "8")), "{R2_Warm}"))
    netlist.add(GeneralElement("v2_warm", a.create_string_array(gateway, ("8", "9")), "{V2_Warm}"))
    netlist.add(GeneralElement("l2_warm", a.create_string_array(gateway, ("9", "10")), "{L2_Warm}"))

    # Energy Extractor 1
    netlist.add(CommentElement("* Energy Extractor 1"))
    eeNodes = a.create_string_array(gateway, ("10", "11"))
    eeAttribute = "RB_EE1_1poleEq"
    R_EE_1a = (1.5 * R_EE_1) / 2
    R_EE_1b = (1.5 * R_EE_1)
    eeParameters = a.create_string_array(gateway, ["R_EE_1", "R_EE_2"])
    eeValues = a.create_string_array(gateway, [str(R_EE_1a), str(R_EE_1b)])
    netlist.add(ParameterizedElement("x1_RB_EE1", eeNodes, eeAttribute, eeParameters, eeValues))
    netlist.add(CommentElement("*"))

    # HTS lead 3 HOT-COLD
    netlist.add(CommentElement("* HTS lead 3 HOT-COLD"))
    netlist.add(GeneralElement("r3_warm", a.create_string_array(gateway, ("11", "12")), "{R3_Warm}"))
    netlist.add(GeneralElement("v3_warm", a.create_string_array(gateway, ("12", "13")), "{V3_Warm}"))
    netlist.add(GeneralElement("l3_warm", a.create_string_array(gateway, ("13", "14")), "{L3_Warm}"))
    netlist.add(GeneralElement("v3_fake", a.create_string_array(gateway, ("14", "MAG78")), "0"))

    # HTS lead 4 COLD-HOT
    netlist.add(CommentElement("* HTS lead 4 COLD-HOT"))
    netlist.add(GeneralElement("v4_fake", a.create_string_array(gateway, ("MAG154_Out", "15")), "0"))
    netlist.add(GeneralElement("r4_warm", a.create_string_array(gateway, ("15", "16")), "{R4_Warm}"))
    netlist.add(GeneralElement("v4_warm", a.create_string_array(gateway, ("16", "17")), "{V4_Warm}"))
    netlist.add(GeneralElement("l4_warm", a.create_string_array(gateway, ("17", "18")), "{L4_Warm}"))

    # Energy Extractor 2
    netlist.add(CommentElement("* Energy Extractor 2"))
    eeNodes = a.create_string_array(gateway, ("18", "19"))
    eeAttribute = "RB_EE2_1poleEq"
    R_EE_2a = (1.5 * R_EE_2) / 2
    R_EE_2b = (1.5 * R_EE_2)
    eeParameters = a.create_string_array(gateway, ["R_EE_1", "R_EE_2"])
    eeValues = a.create_string_array(gateway, [str(R_EE_2a), str(R_EE_2b)])
    netlist.add(ParameterizedElement("x1_RB_EE2", eeNodes, eeAttribute, eeParameters, eeValues))
    netlist.add(CommentElement("*"))

    # Bus bar to PC
    netlist.add(CommentElement("* Bus bar to PC"));
    netlist.add(GeneralElement("r5_warm", a.create_string_array(gateway, ("19", "20")), "{R5_Warm}"))
    netlist.add(GeneralElement("l5_warm", a.create_string_array(gateway, ("20", "21")), "{L5_Warm}"))
    netlist.add(CommentElement("*"))

    # Read R1, R2 for Lumped MB Model
    csv = CSVReader(r1r2CSVInputPath, "\t")
    vecIn = csv.read()

    mbParameters = a.create_string_array(gateway, vecIn.get(0).split(csv.getCsvSplitBy()))
    vecR1R2 = CSVReader.convertCsvStringToDoubleVector(csv.getCsvSplitBy(), vecIn[1:len(vecIn)])

    # Magnet series
    netlist.add(CommentElement("*Magnets Series"))
    if len(position_quenching_magnet) > 2:
        Addition = '0'
    else:
        Addition = ''
    for i in range(N_MAGS):
        if (i + 1) in position_quenching_magnet:
            if elPositions:
                idx = elPositions.index(i + 1)
                Addition = str(idx + 1)
            # Add quenching magnet
            netlist.add(CommentElement("* Magnet #" + str(i + 1) + " is quenching"))
            nameMagnet = "x_MB" + str(i + 1)
            nodesMagnet = a.create_string_array(gateway, ("MAG" + str(i + 1),
                                                          "MAG_Mid" + str(i + 1),
                                                          "MAG" + str(i + 2),
                                                          "MAG_Gnd" + str(i + 1)))
            attributeMagnet = "MAGNET_EQ_2_RCpar" + Addition
            parametersMagnet = a.create_string_array(gateway,
                                                     ("L_1", "L_2", "k_1_2", "C_ground_1", "C_ground_2", "R_parallel"))
            valuesMagnet = a.create_string_array(gateway,
                                                 ("L_1", "L_2", "k_1_2", "C_ground_1", "C_ground_2", "R_parallel"))
            netlist.add(ParameterizedElement(nameMagnet, nodesMagnet, attributeMagnet, parametersMagnet, valuesMagnet));

            # Add cold Diode across the magnet
            nameDiode = "x_D" + str(i + 1)
            nodesDiode = a.create_string_array(gateway, ("MAG" + str(i + 1), "MAG" + str(i + 2)))
            attributeDiode = "RB_Protection_Diode"
            if not FinalRun:
                netlist.add(GeneralElement(nameDiode, nodesDiode, "RB_MB_DiodeFwdBypass_6V"))
            else:
                parametersDiode = a.create_string_array(gateway, ("Is", "U_VT", "fTL", "N1", "N2"))
                valuesDiode = a.create_string_array(gateway, ("Is", "U_VT", "fTL", "N1", "N2"))
                netlist.add(ParameterizedElement(nameDiode, nodesDiode, attributeDiode, parametersDiode, valuesDiode));
        else:
            # Add magnet
            nameMagnet = "x_MB" + str(i + 1)
            nodesMagnet = a.create_string_array(gateway, ("MAG" + str(i + 1),
                                                          "MAG_Mid" + str(i + 1),
                                                          "MAG" + str(i + 2),
                                                          "MAG_Gnd" + str(i + 1)))
            attributeMagnet = "MB_Dipole"
            netlist.add(ParameterizedElement(nameMagnet, nodesMagnet, attributeMagnet, mbParameters,
                                             vecR1R2.get(i).getVector()));

            nameDiode = "x_D" + str(i + 1)
            nodesDiode = a.create_string_array(gateway, ("MAG" + str(i + 1), "MAG" + str(i + 2)))
            # if not FinalRun:
            netlist.add(GeneralElement(nameDiode, nodesDiode, "RB_MB_DiodeFwdBypass_6V"))
            # else:
            #    attributeDiode = "RB_Protection_Diode"
            #    parametersDiode = a.create_string_array(gateway, ("Is", "U_VT", "fTL", "N1", "N2", "I_0"))
            #    valuesDiode = a.create_string_array(gateway,     ("Is", "U_VT", "fTL", "N1", "N2", "I_0"))
            #    netlist.add(ParameterizedElement(nameDiode, nodesDiode, attributeDiode, parametersDiode, valuesDiode));

    # Update node names in order to account for EE units to be connected
    netlist.find("x_MB77").setNode(2, "MAG77_Out")
    netlist.find("x_MB154").setNode(2, "MAG154_Out")
    netlist.find("x_D154").setNode(1, "MAG154_Out")
    netlist.find("x_D77").setNode(1, "MAG77_Out")
    netlist.add(CommentElement("*"))

    # Grounding network
    csv = CSVReader(gndCSVInputPath, "\t")
    vecIn = csv.read()
    vecGnd = CSVReader.convertCsvStringToDoubleVector(csv.getCsvSplitBy(), vecIn[1:len(vecIn)])

    netlist.add(CommentElement("*Magnets grounding network"))
    netlist.add(GeneralElement("v_fakeGND", a.create_string_array(gateway, ("GND1", "0")), "0"))

    iGnd = 1;
    for i in range(vecGnd.size()):
        noOfGnds = vecGnd.get(i).getLength()
        nodes = gateway.new_array(gateway.jvm.String, noOfGnds + 2)

        # Nodes in the grounding element
        for j in range(noOfGnds):
            mbName = str.format("x_MB{}", int(vecGnd.get(i).get(j)))
            nodes[j] = netlist.find(mbName).getNode(INDEX_OUT_NODE)

        nodes[noOfGnds] = "GND" + str(iGnd)
        iGnd += 1
        nodes[noOfGnds + 1] = "GND" + str(iGnd)

        name = "x_MbGND" + str(i + 1)
        subcircuitAttribute = "RB_Gnd_Cell" + str(noOfGnds) + "MB"
        netlist.add(GeneralElement(name, nodes, subcircuitAttribute))

    netlist.find("x_MbGND54").setNode(INDEX_OUT_NODE, "GND54_Float")

    # Read voltage feelers
    csv = CSVReader(voltFeelersCSVInputPath, "\t")
    vecIn = csv.read()

    vecVF = CSVReader.convertCsvStringToDoubleVector(csv.getCsvSplitBy(), vecIn[1:len(vecIn)])

    nodes = gateway.new_array(gateway.jvm.String, 2)

    # Voltage Feelers network
    netlist.add(CommentElement("*Voltage feelers network"));

    for i in range(vecVF.size()):
        name = "r1_VF" + str(i + 1)
        subcircuitAttribute = "20e06"
        mbName = str.format("x_MB{}", int(vecVF.get(i).get(0)))
        nodes[0], nodes[1] = netlist.find(mbName).getNode(0), "v_vf" + str(i + 1)
        netlist.add(GeneralElement(name, nodes, subcircuitAttribute))

        name = "r2_VF" + str(i + 1)
        nodes[0], nodes[1] = "v_vf" + str(i + 1), "0"
        subcircuitAttribute = "24e03"
        netlist.add(GeneralElement(name, nodes, subcircuitAttribute))

    # Additional Output Signals
    netlist.add(CommentElement("****** Outputs ---------------------------------------------------------------"))
    netlist.add(CommentElement("*Signals of the voltage across each magnet"))

    nodes = gateway.new_array(gateway.jvm.String, 2)
    valueNodes = gateway.new_array(gateway.jvm.String, 2)

    # Output voltage across each magnet
    for i in range(1, N_MAGS):
        name = "E_ABM_MAG" + str(i)
        magIn = netlist.find("x_MB" + str(i)).getNode(0)
        magOut = netlist.find("x_MB" + str(i)).getNode(2)
        nodes[0], nodes[1] = "v_mag" + str(i), "0"
        valueNodes = a.create_string_array(gateway, (magIn, magOut))
        netlist.add(OutputGeneralElement(name, nodes, valueNodes))

    netlist.add(CommentElement("*Filtered signals of the voltage across each magnet"))

    # RC filter simulating QPS
    nodes = gateway.new_array(gateway.jvm.String, 2)

    for i in range(1, N_MAGS):
        name = "r_filter" + str(i)
        nodes[0], nodes[1] = "v_mag" + str(i), "v_magf" + str(i)
        subcircuitAttribute = "10e03"
        netlist.add(GeneralElement(name, nodes, subcircuitAttribute))

        name = "c_filter" + str(i)
        nodes[0], nodes[1] = "v_magf" + str(i), "0"
        subcircuitAttribute = "100e-09"
        netlist.add(GeneralElement(name, nodes, subcircuitAttribute))

    # Output voltage across 1st aperture magnet
    netlist.add(CommentElement("*Signals of the voltage across each first aperture of magnets"));

    nodes = gateway.new_array(gateway.jvm.String, 2)
    valueNodes = gateway.new_array(gateway.jvm.String, 2)

    for i in range(1, N_MAGS + 1):
        name = "E_ABM_1stAP_MAG" + str(i)
        magIn = netlist.find("x_MB" + str(i)).getNode(0)
        magOut = netlist.find("x_MB" + str(i)).getNode(1)
        nodes[0], nodes[1] = "0v_ApA" + str(i), "0"
        valueNodes = a.create_string_array(gateway, (magIn, magOut))
        netlist.add(OutputGeneralElement(name, nodes, valueNodes))

    # Output voltage across 2nd aperture magnet
    netlist.add(CommentElement("*Signals of the voltage across each second aperture of magnets"));

    nodes = gateway.new_array(gateway.jvm.String, 2)
    valueNodes = gateway.new_array(gateway.jvm.String, 2)

    for i in range(1, N_MAGS + 1):
        name = "E_ABM_2ndAP_MAG" + str(i)
        magIn = netlist.find("x_MB" + str(i)).getNode(1)
        magOut = netlist.find("x_MB" + str(i)).getNode(2)
        nodes[0], nodes[1] = "0v_ApB" + str(i), "0"
        valueNodes = a.create_string_array(gateway, (magIn, magOut))
        netlist.add(OutputGeneralElement(name, nodes, valueNodes))
    netlist.setCosimulationMode(True)

    # Add simulation options as a comment (uncomment before running the simulation)
    netlist.add(CommentElement("* Simulation options"))
    netlist.add(CommentElement("* .OPTIONS"))
    netlist.add(CommentElement("* + RELTOL=0.001"))
    netlist.add(CommentElement("* + VNTOL=0.001"))
    netlist.add(CommentElement("* + ABSTOL=0.001"))
    netlist.add(CommentElement("* + CHGTOL=0.000000000000001"))
    netlist.add(CommentElement("* + GMIN=0.000000000001"))
    netlist.add(CommentElement("* + ITL1=150"))
    netlist.add(CommentElement("* + ITL2=20"))
    netlist.add(CommentElement("* + ITL4=10"))
    netlist.add(CommentElement("* + TNOM=27"))
    netlist.add(CommentElement("* + NUMDGT=8"))

    netlist.add(CommentElement("* .AUTOCONVERGE"))
    netlist.add(CommentElement("* + RELTOL=0.05"))
    netlist.add(CommentElement("* + VNTOL=0.05"))
    netlist.add(CommentElement("* + ABSTOL=0.05"))
    netlist.add(CommentElement("* + ITL1=1000"))
    netlist.add(CommentElement("* + ITL2=1000"))
    netlist.add(CommentElement("* + ITL4=1000"))
    netlist.add(CommentElement("* + PIVTOL=0.0000000001"))

    netlist.setCosimulationMode(True)
    netlistAsListString = netlist.generateNetlistFile("BINARY")
    ## Remove comments for autoconverge
    for i in range(-22, -3):
        netlistAsListString[i] = netlistAsListString[i][1:]
    circuit = circuit_name
    nameFileSING = os.path.join(os.getcwd(),
                                "tempDir//" + circuit + '_forCosim_Q' + str(position_quenching_magnet[0]) + '.cir')
    if len(position_quenching_magnet) > 2:
        nameFileSING = os.path.join(os.getcwd(), "tempDir//" + circuit + '_forCosim_Q_All.cir')
    TextFile.writeMultiLine(nameFileSING, netlistAsListString, False)

    # Display time stamp and end run
    currentDT = datetime.now()
    print(' ')
    # print('Time stamp: ' + str(currentDT))
    print('Temporary netlist file ' + nameFileSING + ' generated.')

    return nameFileSING