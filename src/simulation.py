
from steam_nb_api.utils.STEAMLib_simulations import *
from src.simulations.stimuli_utils import *
from src.simulations.rb_utils import findRBposition, changeRBStimuli, generateCircuitFile
# datetime package needs to be imported afterwards, otherwise there will be an error
from datetime import datetime


def simulate_RB_circuit(RB_event_data, final_dir=""):
    ################################# Load RB data from CSV & Set-Up simulations ###############################
    ##################################### Provide all necessary Inputs & Options #####################################

    # Half-Turns here are arbitrary for now --> Waiting for Zinur
    enableQuench = 1  # 0 = no quenches included, 1 yes
    quenchHalfTurn_EXT = 40  # Turn set to quench if quench origin = EXT
    quenchHalfTurn_INT = 80  # Turn set to quench if quench origin = INT

    Opts = Options()
    Opts.t_0 = [0.000, 0.12, 0.20, 0.3, 0.5]
    Opts.t_end = [0.12, 0.20, 0.30, 0.5, 1.1]
    Opts.t_step_max = [[1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4]] + [[1.0e-5, 5.0e-5, 1.0e-4, 1.0e-4, 1.0e-4]]
    Opts.relTolerance = [8e-4] + [None]
    Opts.absTolerance = [5] + [None]
    Opts.executionOrder = [1] + [2]
    Opts.executeCleanRun = [True, True]
    InterpolationType = 'Linear'  # Supported: 'Spline', 'Linear'
    sparseTimeStepping = 100

    Layout_db = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'resources',
                             'RB_Layout_Database.xls')
    Config_Name = ''
    startTime = []
    elPositions = []
    CurrentLevel = []
    finalTime = []
    TimeStamp = ''
    tEnd = 300

    # Iterate through primary quench and secondary quenches
    for index, row in RB_event_data.iterrows():
        if row[0] != row[0]: continue
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
        R_EE_2 = float(RB_event_data['U_EE_max_ODD'][index]) / float(RB_event_data['I_Q_circ'][index])
        Pos = RB_event_data['Position'][index]
        startTime.append(float(RB_event_data['Delta_t(iQPS-PIC)'][index] / 1000))
        start = float(RB_event_data['Delta_t(iQPS-PIC)'][index] / 1000)

        newOpts = deepcopy(Opts)
        if I_00 * 2 < 4000:
            newOpts.t_end[-1] = 3
        elif I_00 * 2 < 8000:
            newOpts.t_end[-1] = 2
        finalTime.append(newOpts.t_end[-1])

        ## Find position
        el_pos = findRBposition(Layout_db, Pos, Circuit)
        elPositions.append(el_pos)
        print(I_00)
        TimeStamp_Temp = TimeStamp + '_' + str(int(I_00 * 2)) + 'A_Q' + str(el_pos)


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
        tPC = df["t_PC"][1]
        df.to_csv(os.path.join(os.getcwd(), 'tempDir', 'RB_Circuit_Param_Table.csv'), index=False)

        ## Generate Simulation
        ParameterFile = os.path.join(os.getcwd(), 'tempDir', 'RB_Circuit_Param_Table.csv')
        LSS = LibSim_setup(Circuit, ParameterFile, Opts=newOpts)
        LSS.load_config(Config_Name)

        shutil.rmtree(os.path.join(os.getcwd(), 'tempDir'))

    ######################## Generate combined PSPICE simulation for the complete duration & stitch together results #######################################
    Inputfile_stub = os.path.join(LSS.ModelFolder_EOS, 'COSIM_' + Circuit + '_' + TimeStamp, TimeStamp + '_')
    Lib_Path = LSS.path_PSPICELib.replace(LSS.EOS_stub_C, LSS.EOS_stub_EOS)
    Lib_Path = Lib_Path.replace('\\', '//')

    if final_dir == "":
        PSPICE_folder = LSS.PSPICE_Folder.replace(LSS.EOS_stub_C, LSS.EOS_stub_EOS)
        PSPICE_folder = PSPICE_folder.replace('\\', '//')
        strLEDET = '_Interpolation'
        final_dir = os.path.join(PSPICE_folder, TimeStamp + '_final' + strLEDET)


    if not os.path.isdir(final_dir): os.mkdir(final_dir)

    fakeLSS = LibSim_setup('fake', '')
    fakeLSS.ManualStimuli = ['I_FPA_PC']
    if os.path.isdir('tempDir'): shutil.rmtree(os.path.join(os.getcwd(), 'tempDir'))
    os.mkdir('tempDir')
    StimulusFile = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'RB', 'Gate-EE_Stimuli.stl')
    tempStimFile = changeRBStimuli(StimulusFile, t_EE_1, t_EE_2)
    StimFile = os.path.join(os.path.join(final_dir, 'InputsAsStimula.stl'))
    fakeLSS.generateStimuliCOSIM(StimFile, 'RB', [CurrentLevel[0]], 0, float(tPC), AppendStimuli=tempStimFile)

    timeShift = []
    for k in range(len(elPositions)):
        Inputfile_stub = os.path.join(LSS.ModelFolder_EOS, 'COSIM_' + Circuit + '_' + TimeStamp, TimeStamp + '_')
        Inputfile = os.path.join(
            Inputfile_stub + str(int(CurrentLevel[k] * 2)) + 'A_Q' + str(elPositions[k]) + '_' + str(k + 1),
            'Output', '1_PSPICE', 'Model', 'InputsAsStimula.stl')

        Outputfile = os.path.join(final_dir, 'InputsAsStimula.stl')
        if k == 0:
            type_stl = 'w'
        else:
            type_stl = 'a'
        tShift = startTime[k]
        if tShift < 0: tShift = 0
        lastTime = finalTime[k]

        # Specific to Interpolation_only
        timeShift.append(tShift)
    writeStimuliFromInterpolation(CurrentLevel, Outputfile, type_stl, timeShift, InterpolationType, sparseTimeStepping)

    ################### Generate new Circuit with all quenched magnets & include all new subcircuits ###################
    Library_file = os.path.join(Lib_Path, 'magnet', 'Items', 'magnets_cosimulation.lib')
    pp = LSS.path_NotebookLib.replace(LSS.EOS_stub_EOS, LSS.EOS_stub_C)
    currentDir_C = os.path.join(pp, 'steam-sing-input', 'STEAMLibrary_simulations', final_dir)
    currentDir_C = currentDir_C.replace('/', '\\')
    currentDir_C = currentDir_C.replace('//', '\\')

    if not os.path.isdir('tempDir'):
        os.mkdir('tempDir')
    CircuitFile = generateCircuitFile(Circuit, elPositions, R_EE_1, R_EE_2, elPositions=elPositions,
                                      libraryPath=LSS.path_PSPICELib,
                                      FinalRun=1)
    shutil.copy(CircuitFile, final_dir)
    CircuitFile = CircuitFile.replace(os.path.join(os.getcwd(), 'tempDir'), final_dir)
    os.rename(CircuitFile, os.path.join(final_dir, 'Circuit.cir'))
    CircuitFile = os.path.join(final_dir, 'Circuit.cir')
    appendGenericMagnetModel(CircuitFile, Library_file, elPositions)
    shutil.rmtree(os.path.join(os.getcwd(), 'tempDir'))
    generateConfFile(final_dir, tEnd, 0.001)


# Execution only on Windows, as PSpice is utilized
if __name__ == "__main__":
    steam_notebooks_dir = 'C:\\Users\\cobermai\\cernbox\\SWAN_projects\\steam-notebooks\\steam-sing-input\\STEAMLibrary_simulations'
    rb_event_files_dir = os.path.abspath(os.path.join(os.pardir, "data", "STEAM_context_data"))
    os.chdir(steam_notebooks_dir)
    os.getcwd()

    # RB_event_file = os.path.join(rb_event_files_dir,"RB.A78_FPA-2021-03-28-22h09-2021-03-29-01h00.csv")
    RB_event_file = "RB.A78_FPA-2021-03-28-22h09.csv"
    RB_event_data = pd.read_csv(RB_event_file)
    simulate_RB_circuit(RB_event_data)