import sys
import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from src.utils.hdf_tools import df_to_hdf
from src.visualisation.visualisation import plot_hdf

from src.utils.hdf_tools import load_from_hdf_with_regex



if __name__ == "__main__":
    # define STEAM paths
    steam_dir = r'C:\Users\cobermai\cernbox\SWAN_projects\steam-models-dev'
    steam_analysis_dir = r'\analyses\analysis_RB_with_yaml'
    file_name_analysis = r'C:\Users\cobermai\cernbox\SWAN_projects\lhc-anomaly-detection\data\STEAM_context_data\analysisSTEAM_example_RB.yaml'
    sys.path.insert(0, steam_dir)
    os.chdir(steam_dir + steam_analysis_dir)
    from analyses.analysis_RB_with_yaml.utils_RB import *

    # store hdf in
    hdf_dir = r'\\eosproject-smb\eos\project\m\ml-for-alarm-system\private\RB_signals\20220707_simulation_ADD'
    Path(hdf_dir).mkdir(parents=True, exist_ok=True)
    # store plots in
    plot_dir = r'\\eosproject-smb\eos\project\m\ml-for-alarm-system\private\RB_signals\20220707_sim_plots'
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # load mp3 files
    mp3_excel_path = r'C:\Users\cobermai\cernbox\SWAN_projects\lhc-anomaly-detection\data\RB_TC_extract_2022_07_07_processed_filled.csv'
    mp3_fpa_df = pd.read_csv(mp3_excel_path)

    # secondary quenches have same timestamp as primary quenches
    mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(
        subset=['timestamp_fgc', 'Circuit Name'])

    # only events > 2014 (1388530800000000000), string to unix timestamp with
    # only events = 2021 (1608530800000000000), string to unix timestamp with
    # lhcsmapi.Time.to_unix_timestamp()
    lower_threshold_unix = 1388530800000000000 #1628530800000000000
    mp3_fpa_df_period = mp3_fpa_df_unique[(
        mp3_fpa_df_unique['timestamp_fgc'] >= lower_threshold_unix)].reset_index(drop=True)

    for index, row in mp3_fpa_df_period.iterrows():
        fpa_identifier = f"{row['Circuit Family']}_{row['Circuit Name']}_{int(row['timestamp_fgc'])}"

        if not os.path.isfile(Path(plot_dir) / (fpa_identifier + ".png")):

            mp3_fpa_df_subset = mp3_fpa_df[(mp3_fpa_df['Circuit Name'] == row['Circuit Name']) &
                                           (mp3_fpa_df['timestamp_fgc'] == int(row['timestamp_fgc']))]

            ################################ Define simulation inputs ################################
            # Define flags and settings
            verbose = False
            N_MAGNETS = 154
            flag_run_simulation = True
            parameters_set = 0

            context_data = {
                "selected_circuit": row['Circuit Name'],
                "I_end_2_from_data": mp3_fpa_df_subset['I_Q_M'].max(),
                "dI_dt_from_data": 0,
                "EE_quantities": {'R_EE_odd': float(mp3_fpa_df_subset['U_EE_max_ODD'].values[0]) 
                                               / float(mp3_fpa_df_subset['I_Q_M'].max()),
                                  'R_EE_even': float(mp3_fpa_df_subset['U_EE_max_ODD'].values[0])
                                                / float(mp3_fpa_df_subset['I_Q_M'].max()),
                                  't_EE_odd': float(mp3_fpa_df_subset['Delta_t(EE_odd-PIC)'].values[0]) / 1000,
                                  't_EE_even': float(mp3_fpa_df_subset['Delta_t(EE_even-PIC)'].values[0]) / 1000},
                "current_level_quenches": list(mp3_fpa_df_subset['I_Q_M'].values),
                "t_shifts": list(mp3_fpa_df_subset['Delta_t(iQPS-PIC)'].dropna() / 1000),
                "quenching_magnets": mp3_fpa_df_subset['Position'].tolist()[:len(mp3_fpa_df_subset['Delta_t(iQPS-PIC)'].dropna())]
            }
            print(context_data)
            # Check if any value in context data is nan
            if not pd.json_normalize(context_data, sep='_').isnull().values.any():
                #try:
                time_sim, data_sim, signals_sim, simulation_name, sim_number = \
                    wrapper_RB_analysis(file_name_analysis=file_name_analysis,
                                        parameters_set=parameters_set,
                                        flag_run_simulation=flag_run_simulation,
                                        verbose=verbose,
                                        **context_data)
                df = pd.DataFrame(data_sim, index=time_sim, columns=signals_sim)
                df_to_hdf(file_path=Path(hdf_dir) / (fpa_identifier + ".hdf"), df=df)

                column_regex = ['r1_warm', "0v_magf"]
                data = load_from_hdf_with_regex(file_path=Path(hdf_dir) / (fpa_identifier + ".hdf"), regex_list=column_regex)
                plot_hdf(data=data, column_regex=column_regex, fig_path=Path(plot_dir) / (fpa_identifier + ".png"))
                #except:
                plt.plot([0,1])
                plt.savefig(Path(plot_dir) / ("failed_" + fpa_identifier + ".png"))
 
