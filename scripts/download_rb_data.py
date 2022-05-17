from pathlib import Path

import pandas as pd
from nxcals.spark_session_builder import get_or_create, Flavor

from src.acquisitions.current_voltage_diode_leads_nxcals import CurrentVoltageDiodeLeadsNXCALS
from src.acquisitions.current_voltage_diode_leads_pm import CurrentVoltageDiodeLeadsPM
from src.acquisitions.ee_t_res_pm import EETResPM
from src.acquisitions.ee_u_dump_res_pm import EEUDumpResPM
from src.acquisitions.leads import Leads
from src.acquisitions.pc_pm import PCPM
from src.acquisitions.qh_pm import QHPM
from src.acquisitions.voltage_logic_iqps import VoltageLogicIQPS
from src.acquisitions.voltage_nqps import VoltageNQPS
from src.acquisitions.voltage_nxcals import VoltageNXCALS
from src.utils.hdf_tools import acquisition_to_hdf5
from src.utils.mp3_excel_processing import select_fgc_period, select_fgc_not_downloaded
from src.utils.utils import log_acquisition

if __name__ == "__main__":
    spark = get_or_create(flavor=Flavor.YARN_MEDIUM)
    file_dir = Path('/eos/project/m/ml-for-alarm-system/private/RB_signals')
    #signal_groups = [PCPM, VoltageNQPS, VoltageNXCALS]
    signal_groups = [VoltageNQPS]

    mp3_excel_path = "../data/RB_TC_extract_2021_11_22_processed.csv"
    mp3_fpa_df = pd.read_csv(mp3_excel_path)

    # secondary quenches have same timestamp as primary quenches
    mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['timestamp_fgc', 'Circuit Name'])

    mp3_fpa_df_period = select_fgc_period(mp3_fpa_df_unique,
                                          lower_threshold='2014-01-01 00:00:00+01:00',
                                          upper_threshold='2023-01-01 00:00:00+01:00')

    #mp3_fpa_df_to_download = select_fgc_not_downloaded(context_path=file_dir / "context", mp3_df=mp3_fpa_df_period) # loading of context data takes to long
    #mp3_fpa_df_to_download = mp3_fpa_df_period

    mp3_fpa_df_to_download = pd.Dataframe({'circuit_type': 'RB',
                                           'circuit_name': 'RB.A12',
                                           'timestamp_fgc': 1436908751420000000}, index=[0])

    for index, row in mp3_fpa_df_to_download.iterrows():
        fpa_identifier = {'circuit_type': row['Circuit Family'],
                          'circuit_name': row['Circuit Name'],
                          'timestamp_fgc': int(row['timestamp_fgc'])}

        for signal_group in signal_groups:
            group = signal_group(**fpa_identifier, spark=spark)
            acquisition_to_hdf5(acquisition=group, file_dir=file_dir)

        log_acquisition(identifier=fpa_identifier, log_data={"download_complete": True}, log_path=file_dir / "context")