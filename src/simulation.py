from steam_nb_api.utils.STEAMLib_simulations import *
# Install required package
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import numpy as np
from steam_nb_api.utils import arrays as a
import shutil
import copy
from py4j.java_gateway import launch_gateway, java_import, JavaGateway, JavaObject, GatewayParameters, \
    Py4JNetworkError
# %matplotlib notebook
import matplotlib.pyplot as plt

# import mpld3
# mpld3.enable_notebook()
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline



def load_rb_data(RB_event_file):
    Layout_db = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'resources',
                             'RB_Layout_Database.xls')
    RB_event_data = pd.read_csv(RB_event_file)
    StimulusFile = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'RB', 'Gate-EE_Stimuli.stl')
    startTime = []
    elPositions = []
    CurrentLevel = []
    finalTime = []
    TimeStamp = ''

    for index, row in RB_event_data.iterrows():
        if row[0] != row[0]: continue
        if index == 0:
            Append = False
        else:
            Append = True
        if index == 0:
            date = RB_event_data['Date (FGC)'][0]
            if '/' in date:
                try:
                    date = datetime.strptime(date, '%m/%d/%Y').date().strftime('%Y%m%d')
                except:
                    date = datetime.strptime(date, '%d/%m/%Y').date().strftime('%Y%m%d')
            else:
                date = datetime.strptime(date, '%Y-%m-%d').date().strftime('%Y%m%d')
            time = RB_event_data['Time (FGC)'][0][:5].replace(':', '')
            TimeStamp = 'FPA_' + date + '_' + time

        if os.path.isdir('tempDir'): shutil.rmtree(os.path.join(os.getcwd(), 'tempDir'))
        os.mkdir('tempDir')
        Circuit = RB_event_data['Circuit Name'][index]
        t_EE_1 = float(RB_event_data['Delta_t(EE_odd-PIC)'][index])
        t_EE_2 = float(RB_event_data['Delta_t(EE_even-PIC)'][index])
        I_00 = float(RB_event_data['I_Q_M'][index] / 2)
        I_initial = float(RB_event_data['I_Q_circ'][index] / 2)
        CurrentLevel.append(I_00)
        R_EE_1 = float(RB_event_data['U_EE_max_ODD'][index]) / float(RB_event_data['I_Q_circ'][index])
        R_EE_2 = float(RB_event_data['U_EE_max_EVEN'][index]) / float(RB_event_data['I_Q_circ'][index])
        Pos = RB_event_data['Position'][index]
        startTime.append(float(RB_event_data['Delta_t(iQPS-PIC)'][index] / 1000))
        start = float(RB_event_data['Delta_t(iQPS-PIC)'][index] / 1000)
        if start > (t_EE_1 / 1000):
            EE1 = 1
        else:
            EE1 = 0
        if start > (t_EE_2 / 1000):
            EE2 = 1
        else:
            EE2 = 0

        newOpts = deepcopy(Opts)
        if I_00 * 2 < 4000:
            newOpts.t_end[-1] = 3
        elif I_00 * 2 < 8000:
            newOpts.t_end[-1] = 2
        finalTime.append(newOpts.t_end[-1])

        if not Skip_SetUp:
            ## Change Stimuli
            tempStimFile = changeRBStimuli(StimulusFile, t_EE_1, t_EE_2, EE1=EE1, EE2=EE2)

        ## Find position
        el_pos = findRBposition(Layout_db, Pos)
        elPositions.append(el_pos)
        TimeStamp_Temp = TimeStamp + '_' + str(int(I_00 * 2)) + 'A_Q' + str(el_pos)

        if not Skip_SetUp:
            ## Generate .cir-file for Quench position
            if not LEDET_only:
                CircuitFile = generateCircuitFile(Circuit, el_pos)
            else:
                CircuitFile = ''

        ## Adjust temporary Circuit_Param_Table
        df = pd.read_csv(os.path.join(os.getcwd(), 'RB_Circuit_Param_Table.csv'))
        df["I00"] = str(RB_event_data['I_Q_M'][index])
        Quench_origin = RB_event_data['Quench origin'][index]
        if Quench_origin == 'EXT':
            new_iqT = quenchHalfTurn_EXT
        elif Quench_origin == 'INT':
            new_iqT = quenchHalfTurn_INT
        else:
            new_iqT = 1
            print('Quench origin not understood. Choose Turn 1.')
        df["i_qT"] = new_iqT
        if LEDET_only:
            df["flag_COSIM"] = 0
        tPC = df["t_PC"][1]
        df.to_csv(os.path.join(os.getcwd(), 'tempDir', 'RB_Circuit_Param_Table.csv'), index=False)

        ## Generate Simulation
        ParameterFile = os.path.join(os.getcwd(), 'tempDir', 'RB_Circuit_Param_Table.csv')
        LSS = LibSim_setup(Circuit, ParameterFile, Opts=newOpts)
        LSS.load_config(Config_Name)
        if not Skip_SetUp:
            LSS.SetUpSimulation(TimeStamp_Temp, [[int(I_initial)]], ManualStimuli=['I_FPA_PC'], Append=Append,
                                ManualCircuit=CircuitFile, AppendStimuli=tempStimFile, HierCOSIM='_' + str(index + 1),
                                convergenceElement='x_MB' + str(el_pos))
        shutil.rmtree(os.path.join(os.getcwd(), 'tempDir'))

    if not Skip_SetUp:
        if LEDET_only:
            LSS.StampBatch(TimeStamp + '_LEDET')
        else:
            LSS.StampBatch(TimeStamp + '_COSIM')


def pspice_setup():
    if not Interpolation_only:
        ## Check if all results are present
        if not LEDET_only:
            Inputfile_stub = os.path.join(LSS.ModelFolder_EOS, 'COSIM_' + Circuit + '_' + TimeStamp, TimeStamp + '_')
            for k in range(len(elPositions)):
                Inputfile = os.path.join(
                    Inputfile_stub + str(int(CurrentLevel[k] * 2)) + 'A_Q' + str(elPositions[k]) + '_' + str(k + 1),
                    'Output', '1_PSPICE', 'Output', 'out_final.csv')
                print(Inputfile)
                if os.path.exists(Inputfile):
                    continue
                else:
                    print(k)
                    raise NameError(
                        'Not all results found. Please check. If you want to continue, please do so, but code will crash.')
        else:
            Inputfile_stub = os.path.join(LSS.ModelFolder_EOS, 'LEDET_model_' + Circuit + '_' + TimeStamp,
                                          TimeStamp + '_')
            for k in range(len(elPositions)):
                Inputfile = os.path.join(
                    Inputfile_stub + str(int(CurrentLevel[k] * 2)) + 'A_Q' + str(elPositions[k]) + '_' + str(k + 1),
                    'LEDET', 'LEDET', 'MB_2COILS', 'Output', 'Txt Files', 'MB_2COILS_VariableHistory_0.txt')
                print(Inputfile)
                if os.path.exists(Inputfile):
                    continue
                else:
                    raise NameError(
                        'Not all results found. Please check. If you want to continue, please do so, but code will crash.')


if __name__ == '__main__':
    RB_event_file = "RB.A78_FPA-2021-03-28-22h09-2021-03-29-01h00.csv"
    # RB_event_file = "RB.A78_FPA-2021-03-28-22h09.csv"

    Config_Name = ''

    LEDET_only = 0  # If 1, the quenches will be simulated in LEDET only
    Skip_SetUp = 0  # If set to 1, the set-up will be skipped and you jump straigt to the stitching & results
    # If set to 1, the set-up will be skipped and results will be obtained by interpolation
    # of previously obtained results
    Interpolation_only = 1
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

    if Interpolation_only:
        Skip_SetUp = 1
        LEDET_only = 0
    if LEDET_only == Interpolation_only and Interpolation_only == 1:
        raise "Please decide if interpolation only and LEDET only"

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

    Inputfile_stub = os.path.join(LSS.ModelFolder_EOS, 'COSIM_' + Circuit + '_' + TimeStamp, TimeStamp + '_')
    Lib_Path = LSS.path_PSPICELib.replace(LSS.EOS_stub_C, LSS.EOS_stub_EOS)
    Lib_Path = Lib_Path.replace('\\', '//')
    PSPICE_folder = LSS.PSPICE_Folder.replace(LSS.EOS_stub_C, LSS.EOS_stub_EOS)
    PSPICE_folder = PSPICE_folder.replace('\\', '//')
    if LEDET_only:
        strLEDET = '_LEDET'
    elif Interpolation_only:
        strLEDET = '_Interpolation'
    else:
        strLEDET = '_COSIM'
    final_dir = os.path.join(PSPICE_folder, TimeStamp + '_final' + strLEDET)
    if not os.path.isdir(final_dir): os.mkdir(final_dir)

    if LEDET_only or Interpolation_only:
        fakeLSS = LibSim_setup('fake', '')
        fakeLSS.ManualStimuli = ['I_FPA_PC']
        if os.path.isdir('tempDir'): shutil.rmtree(os.path.join(os.getcwd(), 'tempDir'))
        os.mkdir('tempDir')
        StimulusFile = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'RB', 'Gate-EE_Stimuli.stl')
        tempStimFile = changeRBStimuli(StimulusFile, t_EE_1, t_EE_2)
        StimFile = os.path.join(os.path.join(final_dir, 'InputsAsStimula.stl'))
        fakeLSS.generateStimuliCOSIM(StimFile, 'RB', [CurrentLevel[0]], 0, float(tPC), tempStimFile,
                                     AppendStimuli=tempStimFile)
    else:
        GeneralStim = os.path.join(
            Inputfile_stub + str(int(CurrentLevel[0] * 2)) + 'A_Q' + str(elPositions[0]) + '_' + str(1), 'Input',
            'PSpice', 'ExternalStimulus.stl')



    timeShift = []
    for k in range(len(elPositions)):
        if not LEDET_only:
            Inputfile_stub = os.path.join(LSS.ModelFolder_EOS, 'COSIM_' + Circuit + '_' + TimeStamp, TimeStamp + '_')
            Inputfile = os.path.join(
                Inputfile_stub + str(int(CurrentLevel[k] * 2)) + 'A_Q' + str(elPositions[k]) + '_' + str(k + 1),
                'Output', '1_PSPICE', 'Model', 'InputsAsStimula.stl')
        else:
            Inputfile_stub = os.path.join(LSS.ModelFolder_EOS, 'LEDET_model_' + Circuit + '_' + TimeStamp,
                                          TimeStamp + '_')
            Inputfile = os.path.join(
                Inputfile_stub + str(int(CurrentLevel[k] * 2)) + 'A_Q' + str(elPositions[k]) + '_' + str(k + 1),
                'LEDET', 'LEDET', 'MB_2COILS', 'Output', 'Txt Files', 'MB_2COILS_VariableHistory_0.txt')

        Outputfile = os.path.join(final_dir, 'InputsAsStimula.stl')
        if k == 0:
            type_stl = 'w'
        else:
            type_stl = 'a'
        tShift = startTime[k]
        if tShift < 0: tShift = 0
        lastTime = finalTime[k]

        if LEDET_only:
            type_stl = 'a'
            WriteStimuliFromLEDET(Inputfile, Outputfile, type_stl, k + 1, tShift)
        elif not Interpolation_only:
            StimulaAdjustment_Time(Inputfile, Outputfile, type_stl, k + 1, tShift, lastTime)
        else:
            timeShift.append(tShift)

    if Interpolation_only:
        writeStimuliFromInterpolation(CurrentLevel, Outputfile, type_stl, timeShift, InterpolationType)

    if not LEDET_only and not Interpolation_only:
        appendGeneralStimuli(GeneralStim, Outputfile)