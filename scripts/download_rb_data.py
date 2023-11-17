from pathlib import Path

import pandas as pd
from nxcals.spark_session_builder import get_or_create, Flavor

from src.acquisition import download_data
from src.acquisitions.ee_u_dump_res_pm import EEUDumpResPM
from src.acquisitions.pc_pm import PCPM
from src.acquisitions.qh_pm import QHPM
from src.acquisitions.voltage_logic_iqps import VoltageLogicIQPS
from src.acquisitions.voltage_nqps import VoltageNQPS
from src.acquisitions.voltage_nxcals import VoltageNXCALS
from src.utils.fgc_timestamp_utils import select_fgc_period

if __name__ == "__main__":
    spark = get_or_create(flavor=Flavor.YARN_MEDIUM)  # only required to download NXCALS data

    # get list of fpa_identifiers
    mp3_excel_path = Path("../data/processed/MP3_context/RB_TC_extract_2023_11_17_processed.csv")
    mp3_fpa_df = pd.read_csv(mp3_excel_path)

    # secondary quenches have same timestamp as primary quenches
    mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['fpa_identifier']).dropna(subset=['fpa_identifier'])
    mp3_fpa_df_period = select_fgc_period(mp3_fpa_df_unique,
                                          lower_threshold='2014-01-01 00:00:00+01:00',
                                          upper_threshold='2024-01-01 00:00:00+01:00')

    output_dir = Path('/eos/project/m/ml-for-alarm-system/private/RB_signals/raw')
    signal_groups = [PCPM, VoltageNXCALS, VoltageNQPS, VoltageLogicIQPS, EEUDumpResPM,
                     QHPM]  # select all signals to download from src/acquisition
    plot_regex = ['I_MEAS', 'VoltageNQPS.*U_DIODE', 'VoltageNXCALS.*U_DIODE', 'I_EARTH_PCNT', 'IEARTH.I_EARTH', 'U_QS0',
                  'U_1', 'U_2', 'I_HDS', 'U_HDS',
                  'EEUDumpResPM']  # Regex of signals can be found in src/acquisition/acquisition_example.png

    download_data(fpa_identifiers=mp3_fpa_df_period['fpa_identifier'].to_list(),
                  signal_groups=signal_groups,
                  output_dir=output_dir,
                  plot_regex=plot_regex,
                  spark=spark)
