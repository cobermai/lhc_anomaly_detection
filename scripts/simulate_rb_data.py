import os

import pandas as pd

from src.simulation import simulate_RB_circuit

# Execution only on Windows, as PSpice is utilized
if __name__ == "__main__":
    steam_notebooks_dir = 'C:\\Users\\cobermai\\cernbox\\SWAN_projects\\steam-notebooks\\steam-sing-input\\STEAMLibrary_simulations'
    rb_event_files_dir = os.path.abspath(
        os.path.join(os.pardir, "data", "STEAM_context_data"))
    os.chdir(steam_notebooks_dir)

    mp3_excel_path = r'C:\Users\cobermai\cernbox\SWAN_projects\lhc-anomaly-detection\data\RB_TC_extract_2021_11_22_processed_filled'
    mp3_fpa_df = pd.read_csv(mp3_excel_path)

    output_path = r'\\eosproject-smb\eos\project\m\ml-for-alarm-system\private\RB_signals\PSPICE'

    # secondary quenches have same timestamp as primary quenches
    mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(
        subset=['timestamp_fgc', 'Circuit Name'])

    # only events > 2014, string to unix timestamp with
    # lhcsmapi.Time.to_unix_timestamp()
    lower_threshold_unix = 1388530800000000000
    mp3_fpa_df_period = mp3_fpa_df_unique[(
        mp3_fpa_df_unique['timestamp_fgc'] >= lower_threshold_unix)].reset_index(drop=True)

    for index, row in mp3_fpa_df_period.iterrows():
        fpa_identifier = f"{row['Circuit Family']}_{row['Circuit Name']}_{int(row['timestamp_fgc'])}"

        simulation_path = output_path + fpa_identifier
        mp3_fpa_df_subset = mp3_fpa_df[(mp3_fpa_df['Circuit Name'] == row['Circuit Name']) & (
            mp3_fpa_df['timestamp_fgc'] == int(row['timestamp_fgc']))]

        # Ensure all columns necessary for simulation are available
        if mp3_fpa_df_subset['I_Q_circ'].isna().any():
            mp3_fpa_df_subset['I_Q_circ'] = mp3_fpa_df_subset['I_Q_M'].max()
        used_columns = [
            'Circuit Name',
            'Type of Quench',
            'I_Q_circ',
            'I_Q_M',
            'Position',
            'Quench origin',
            'Delta_t(EE_odd-PIC)',
            'Delta_t(EE_even-PIC)',
            'Delta_t(iQPS-PIC)',
            'U_EE_max_ODD',
            'U_EE_max_EVEN',
            'I_Q_circ']
        mp3_fpa_df_subset_used = mp3_fpa_df_subset[used_columns]
        mp3_fpa_df_subset.to_csv(simulation_path + "\\input.csv")

        simulate_RB_circuit(mp3_fpa_df_subset, final_dir=simulation_path)
