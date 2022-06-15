import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from steam_nb_api.utils.STEAMLib_simulations import *
from src.simulations.rb_utils import findRBposition, changeRBStimuli, generateCircuitFile
# datetime package needs to be imported afterwards, otherwise there will be an error
from datetime import datetime

# Append changed stimuli & models
def appendGenericMagnetModel(cir_file, Library_file, elPositions):
    new_cir = cir_file.replace('.cir', '_final.cir')
    # validStl = ['V_circuit_1_stim','V_circuit_2_stim','V_field_1_stim','V_field_2_stim','R_field_1_stim','R_field_2_stim']
    validStl = ['R_field_1_stim', 'R_field_2_stim']
    lib_element = 'MAGNET_EQ_2_RCpar'

    with open(cir_file, 'r') as cfile:
        stlString = ''
        for line in cfile:
            if '.END' in line: break
            if 'L_1' in line:
                line = line.replace('L_1', 'L1')
            if 'L_2' in line:
                line = line.replace('L_2', 'L2')
            stlString = stlString + line
        stlString = stlString + '\n ******************************************************** \
        Generic models ************************************************************* \n *****\n *****\n\n'

        replace_connection = 0
        for i in range(len(elPositions)):
            with open(Library_file, 'r') as lfile:
                inMod = 0
                for line in lfile:
                    if not inMod:
                        if not lib_element in line:
                            continue
                        else:
                            inMod = 1
                    if lib_element in line: line = line.replace(lib_element, lib_element + str(i + 1))
                    for k in range(len(validStl)):
                        if validStl[k] in line:
                            line = line.replace(validStl[k], validStl[k] + '_' + str(i + 1))
                    if 'STIMULUS' in line and '_' + str(i + 1) not in line[-5:]:
                        replace_connection = 1
                        continue
                    if '1c' in line and replace_connection:
                        line = line.replace('1c', '1a')
                    if '2c' in line and replace_connection:
                        line = line.replace('2c', '2a')
                    if 'L_1' in line:
                        line = line.replace('L_1', 'L1')
                    if 'L_2' in line:
                        line = line.replace('L_2', 'L2')
                    stlString = stlString + line

                    if '.ends' in line:
                        inMod = 0
                        continue
        stlString = stlString + '.END'
        with open(new_cir, 'w') as ncfile:
            ncfile.write(stlString)
    os.remove(cir_file)
    os.rename(new_cir, cir_file)
    print('Final netlist file ' + cir_file + ' generated.')


# Append Conf-File
def generateConfFile(final_dir, endTime, timeStep):
    Conf_File = os.path.join(final_dir, 'Conf.cir')
    with open(Conf_File, 'w') as cfile:
        stlString = ''
        stlString = stlString + '.tran 0.000000 ' + str(endTime) + ' 0.000000' + '\n'
        stlString = stlString + '+ {schedule(' + '\n'
        stlString = stlString + '+ 0.000000, ' + str(timeStep) + ',\n'
        stlString = stlString + '+ 1.000000, 0.1 \n'
        stlString = stlString + '+ )}' + '\n'
        stlString = stlString + '.stmlib "InputsAsStimula.stl"' + '\n'
        stlString = stlString + '.probe/csdf' + '\n'

        fileSignals = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'resources',
                                   "selectedSignals_RB.csv")
        with open(fileSignals) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                try:
                    if str(row[1]): stlString = stlString + '+' + str(row[0]) + ',' + str(row[1]) + '\n'
                except:
                    stlString = stlString + '+' + str(row[0]) + '\n'

        cfile.write(stlString)
        print(Conf_File + ' generated.')

def InterpolateResistance(current_level, Type, plot_interpolation=False):
    max_time = 1.1
    ## Do the interpolation here
    IntpRB_file = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'resources',
                               'Interpolation_Resistance_RB.csv')
    IntpRB_data = pd.read_csv(IntpRB_file)
    col_t = []
    col_r1 = []
    col_r2 = []

    for col in IntpRB_data.columns:
        try:
            _ = int(col)
            idx = IntpRB_data.columns.get_loc(col)
            col_t.append(IntpRB_data.columns[idx])
            col_r1.append(IntpRB_data.columns[idx + 1])
            col_r2.append(IntpRB_data.columns[idx + 2])
        except:
            pass
    data_R1 = IntpRB_data[col_r1].drop([0]).to_numpy().astype(float)
    data_R2 = IntpRB_data[col_r2].drop([0]).to_numpy().astype(float)
    time = IntpRB_data[col_t[-1]].drop([0]).to_numpy().astype(float)

    new_R1 = []
    new_R2 = []
    current_level = np.array([current_level]).reshape(-1, 1)

    x = np.array(col_t).astype(float).reshape(-1, 1)

    for k in range(data_R1.shape[0]):
        new_y = data_R1[k][~np.isnan(data_R1[k])].reshape(-1, )
        new_x = x[~np.isnan(data_R1[k])].reshape(-1, )

        if Type == 'Spline':
            new_x = new_x[::-1]
            new_y = new_y[::-1]

            if len(new_x) <= 3:
                new_R1 = np.append(new_R1, np.nan)
            else:
                spl = UnivariateSpline(new_x, new_y)
                new_R1 = np.append(new_R1, spl(current_level))
        elif Type == 'Linear':
            if current_level <= max(new_x):
                f = interp1d(new_x, new_y)
                new_R1 = np.append(new_R1, f(current_level))
            else:
                new_R1 = np.append(new_R1, np.nan)

        new_y = data_R2[k][~np.isnan(data_R2[k])].reshape(-1, )
        new_x = x[~np.isnan(data_R2[k])].reshape(-1, )
        if Type == 'Spline':
            new_x = new_x[::-1]
            new_y = new_y[::-1]

            if len(new_x) <= 3:
                new_R1 = np.append(new_R1, np.nan)
            else:
                spl = UnivariateSpline(new_x, new_y)
                new_R2 = np.append(new_R2, spl(current_level))
        elif Type == 'Linear':
            try:
                f = interp1d(new_x, new_y)
                new_R2 = np.append(new_R2, f(current_level))
            except:
                new_R2 = np.append(new_R2, np.nan)

    if plot_interpolation:
        f = plt.figure(figsize=(17, 8))
        plt.subplot(1, 2, 1)
        plt.plot(time, new_R1)
        leg = ["Interpolated-" + str(current_level)]
        for i in range(data_R1.shape[1]):
            plt.plot(time, data_R1[:, i])
            leg.append(x[i][0])
        plt.legend(leg)
        plt.xlabel('Time [s]')
        plt.ylabel('Resistance [Ohm]')
        plt.title('R_CoilSection 1')

        plt.subplot(1, 2, 2)
        plt.plot(time, new_R2)
        leg = ["Interpolated-" + str(current_level)]
        for i in range(data_R2.shape[1]):
            plt.plot(time, data_R2[:, i])
            leg.append(x[i][0])
        plt.legend(leg)
        plt.xlabel('Time [s]')
        plt.ylabel('Resistance [Ohm]')
        plt.title('R_CoilSection 2')
        f.suptitle(str(current_level[0][0]) + 'A', fontsize=16)
        plt.show()
    return [time, new_R1, new_R2]


def writeStimuliFromInterpolation(current_level, Outputfile, type_stl, tShift, InterpolationType, sparseTimeStepping):
    R1 = np.array([])
    R2 = np.array([])
    print("Interpolating Coil-Resistances")
    for k in current_level:
        [time, data_R1, data_R2] = InterpolateResistance(k * 2, InterpolationType)
        if not R1.size > 0:
            # added by cobermai: [np.newaxis, ...]
            R1 = data_R1[np.newaxis, ...]
        else:
            R1 = np.vstack((R1, data_R1))
        if not R2.size > 0:
            # added by cobermai: [np.newaxis, ...]
            R2 = data_R2[np.newaxis, ...]
        else:
            R2 = np.vstack((R2, data_R2))
    stlString = ''

    for k in range(len(current_level)):
        timeShift = tShift[k]
        if timeShift < 0: timeShift = 0
        stlString = stlString + '\n .STIMULUS R_field_' + str(1) + '_stim_' + str(
            k + 1) + ' PWL \n + TIME_SCALE_FACTOR = 1 \n + VALUE_SCALE_FACTOR = 1 \n'
        stlString = stlString + "+ ( 0s, 0.0 )\n"
        count = 0
        for l in range(1, R1.shape[1] - 1):
            if np.isnan(R1[k, l]): continue
            if float(time[l]) < 0:  timeShift = timeShift + abs(float(time[l])) - 0.03
            if float(time[l]) + timeShift < 0:
                tt = 0
            else:
                tt = float(time[l]) + timeShift
            if count >= sparseTimeStepping:
                stlString = stlString + "+ ( " + str(tt) + "s, " + str(R1[k, l]) + " )\n"
                count = 0
            count = count + 1
        R1_last = R1[k]
        R1_last = R1_last[~np.isnan(R1_last)]
        stlString = stlString + "+ ( " + str(10000) + "s," + str(R1_last[-1]) + " ) \n"
        stlString = stlString + " \n"

        stlString = stlString + '\n .STIMULUS R_field_' + str(2) + '_stim_' + str(
            k + 1) + ' PWL \n + TIME_SCALE_FACTOR = 1 \n + VALUE_SCALE_FACTOR = 1 \n'
        stlString = stlString + "+ ( 0s, 0.0 )\n"
        count = 0
        for l in range(1, R2.shape[1] - 1):
            if np.isnan(R2[k, l]): continue
            if float(time[l]) < 0:  timeShift = timeShift + abs(float(time[l])) - 0.03
            if float(time[l]) + timeShift < 0:
                tt = 0
            else:
                tt = float(time[l]) + timeShift
            if count >= sparseTimeStepping:
                stlString = stlString + "+ ( " + str(tt) + "s, " + str(R2[k, l]) + " )\n"
                count = 0
            count = count + 1
        R2_last = R2[k]
        R2_last = R2_last[~np.isnan(R2_last)]
        stlString = stlString + "+ ( " + str(10000) + "s," + str(R2_last[-1]) + " ) \n"
        stlString = stlString + " \n"

    with open(Outputfile, type_stl) as ofile:
        ofile.write(stlString)