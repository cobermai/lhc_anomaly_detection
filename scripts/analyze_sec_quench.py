import os.path
from pathlib import Path

import pandas as pd

from src.modeling.sec_quench import get_sec_quench_frame_exclude_quench, get_sec_quench_features
from src.utils.hdf_tools import load_from_hdf_with_regex
from src.utils.dataset_utils import u_diode_data_to_df

if __name__ == "__main__":
    # define paths
    data_path = Path('/eos/project/m/ml-for-alarm-system/private/RB_signals/20220707_data')
    if not os.path.isdir(data_path):
        data_path = Path('/mnt/d/datasets/20220707_data')
    output_path = Path("../data/sec_quench_feature.csv")
    mp3_excel_path = Path("../data/RB_TC_extract_2022_07_07_processed_filled.csv")
    acquisition_summary_path = Path("../data/20220707_acquisition_summary.xlsx")

    # load mp3 files
    mp3_fpa_df = pd.read_csv(mp3_excel_path)

    # secondary quenches have same timestamp as primary quenches
    mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['timestamp_fgc', 'Circuit Name'])

    # only events > 2014 (1388530800000000000), string to unix timestamp with
    # only events = 2021 (1608530800000000000), string to unix timestamp with
    lower_threshold_unix = 1388530800000000000
    mp3_fpa_df_period = mp3_fpa_df_unique[(
        mp3_fpa_df_unique['timestamp_fgc'] >= lower_threshold_unix)].reset_index(drop=True)
    mp3_fpa_df_period_all = mp3_fpa_df[mp3_fpa_df['timestamp_fgc'] >= lower_threshold_unix].reset_index(drop=True)

    # Add information, whether VoltageNXCALS.*U_DIODE download was successfull
    df_acquisition = pd.read_excel(acquisition_summary_path)
    mp3_fpa_df_period_merged = mp3_fpa_df_period.merge(df_acquisition, left_on=['Circuit Name', 'timestamp_fgc'],
                                                       right_on=['Circuit Name', 'timestamp_fgc'], how="left")
    df_to_analyze = mp3_fpa_df_period_merged[mp3_fpa_df_period_merged["VoltageNXCALS.*U_DIODE"] == 1]

    # define period to analyze
    time_frame_after_quench = [0, 2]

    df_results = pd.DataFrame()
    for k, row in df_to_analyze.iterrows():
        circuit_name = row['Circuit Name']
        timestamp_fgc = int(row['timestamp_fgc'])
        fpa_identifier = f"{row['Circuit Family']}_{row['Circuit Name']}_{int(row['timestamp_fgc'])}"

        data_dir = data_path / (fpa_identifier + ".hdf5")
        data = load_from_hdf_with_regex(file_path=data_dir, regex_list=["VoltageNXCALS.*U_DIODE"])

        df_data_nxcals = u_diode_data_to_df(data)
        df_subset = mp3_fpa_df[(mp3_fpa_df.timestamp_fgc == timestamp_fgc) &
                               (mp3_fpa_df["Circuit Name"] == circuit_name)]

        print(f"{k}/{len(df_to_analyze)} {fpa_identifier} n_quenches:{len(df_subset)} len_data:{len(df_data_nxcals)}")

        quench_times = df_subset["Delta_t(iQPS-PIC)"].values / 1e3
        sec_quenches = get_sec_quench_frame_exclude_quench(df_data=df_data_nxcals,
                                                           all_quenched_magnets=df_subset.Position.values,
                                                           quench_times=quench_times,
                                                           time_frame=time_frame_after_quench)

        for sec_quench_number, df_quench_frame in enumerate(sec_quenches):
            if not df_quench_frame.empty:
                df_results_new = get_sec_quench_features(df_quench_frame=df_quench_frame,
                                                         df_mp3_subset=df_subset,
                                                         time_frame_after_quench=time_frame_after_quench,
                                                         sec_quench_number=sec_quench_number)
                df_results = pd.concat([df_results, df_results_new])

    df_results.dropna().to_csv(output_path)


