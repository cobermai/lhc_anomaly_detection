import os

import pandas as pd
from src.simulation import simulate_RB_circuit

# Execution only on Windows, as PSpice is utilized
if __name__ == "__main__":
    steam_notebooks_dir = 'C:\\Users\\cobermai\\cernbox\\SWAN_projects\\steam-notebooks\\steam-sing-input\\STEAMLibrary_simulations'
    rb_event_files_dir = os.path.abspath(os.path.join(os.pardir, "data", "STEAM_context_data"))
    os.chdir(steam_notebooks_dir)

    RB_event_file = r"C:\Users\cobermai\cernbox\SWAN_projects\lhc-anomaly-detection\data\STEAM_context_data\RB.A78_FPA-2021-03-28-22h09.csv"
    RB_event_data = pd.read_csv(RB_event_file)

    mp3_excel_path = r'C:\Users\cobermai\cernbox\SWAN_projects\lhc-anomaly-detection\data\RB_TC_extract_2021_11_22_processed_filled'
    mp3_fpa_df = pd.read_csv(mp3_excel_path)

    # secondary quenches have same timestamp as primary quenches
    mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['timestamp_fgc', 'Circuit Name'])

    # only events > 2014, string to unix timestamp with lhcsmapi.Time.to_unix_timestamp()
    lower_threshold_unix = 1388530800000000000
    mp3_fpa_df_period = mp3_fpa_df_unique[(mp3_fpa_df_unique['timestamp_fgc'] >= lower_threshold_unix)].reset_index(drop=True)

    for index, row in mp3_fpa_df_period.iterrows():
        circuit_type = row['Circuit Family']
        circuit_name = row['Circuit Name']
        timestamp_fgc = row['timestamp_fgc']

        foldername = f"\{circuit_type}_{circuit_name}_{int(timestamp_fgc)}"
        simulation_path = r'\\eosproject-smb\eos\project\m\ml-for-alarm-system\private\RB_signals\PSPICE' + foldername


        mp3_fpa_df_subset = mp3_fpa_df[(mp3_fpa_df['Circuit Name'] == circuit_name) &
                                       (mp3_fpa_df['timestamp_fgc'] == timestamp_fgc)]

        if mp3_fpa_df_subset['I_Q_circ'].isna().any():
            mp3_fpa_df_subset['I_Q_circ'] = mp3_fpa_df_subset['I_Q_M'].max()

        missing_columns = ['Delta_t(EE_odd-PIC)', 'Delta_t(EE_even-PIC)', 'Delta_t(iQPS-PIC)', 'U_EE_max_ODD', 'U_EE_max_EVEN', 'I_Q_circ']
        mp3_fpa_df_subset_missing = mp3_fpa_df_subset[missing_columns]

        # allways nan before 2021: 'Delta_t(EE_odd-PIC)', 'Delta_t(EE_even-PIC)', 'Delta_t(iQPS-PIC)', 'U_EE_max_ODD', 'U_EE_max_EVEN', 'I_Q_circ' (take I_Q_M of first quench)
        # Nan dependent on 'Type of Quench': 'Quench origin'
        #if not os.path.isdir(simulation_path):
        #    simulate_RB_circuit(mp3_fpa_df_subset, final_dir=simulation_path)
        #    mp3_fpa_df_subset.to_csv(simulation_path + "\input.csv")

        try:
            if not os.path.isdir(simulation_path):
                simulate_RB_circuit(mp3_fpa_df_subset, final_dir=simulation_path)
                mp3_fpa_df_subset.to_csv(simulation_path + "\input.csv")
        except Exception as e:
            failed_dir = simulation_path + '_failed'
            os.mkdir(failed_dir)
            mp3_fpa_df_subset.to_csv(failed_dir + "\input.csv")

            print(e)
            print('failed')


