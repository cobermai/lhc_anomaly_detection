import os
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
from src.utils.hdf_tools import acquisition_to_hdf5, load_from_hdf_with_regex
from src.utils.mp3_excel_processing import select_fgc_period, select_fgc_not_downloaded
from src.utils.utils import log_acquisition
from src.visualisation.visualisation import plot_hdf

if __name__ == "__main__":
    spark = get_or_create(flavor=Flavor.YARN_MEDIUM)
    file_dir = Path('/eos/project/m/ml-for-alarm-system/private/RB_signals')
    signal_groups = signal_groups = [PCPM, VoltageNXCALS, VoltageNQPS, VoltageLogicIQPS, EEUDumpResPM, QHPM]

    mp3_excel_path = "../data/RB_TC_extract_2022_07_07_processed.csv"
    mp3_fpa_df = pd.read_csv(mp3_excel_path)

    # secondary quenches have same timestamp as primary quenches
    mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['timestamp_fgc', 'Circuit Name'])

    mp3_fpa_df_period = select_fgc_period(mp3_fpa_df_unique,
                                          lower_threshold='2014-01-01 00:00:00+01:00',
                                          upper_threshold='2023-01-01 00:00:00+01:00')

    #mp3_fpa_df_to_download = select_fgc_not_downloaded(context_path=file_dir / "context", mp3_df=mp3_fpa_df_period) # loading of context data takes to long
    mp3_fpa_df_to_download = mp3_fpa_df_period

    for index, row in mp3_fpa_df_to_download.iterrows():

        fpa_identifier = {'circuit_type': row['Circuit Family'],
                          'circuit_name': row['Circuit Name'],
                          'timestamp_fgc': int(row['timestamp_fgc'])}
        file_name = f"{fpa_identifier['circuit_type']}_{fpa_identifier['circuit_name']}_{fpa_identifier['timestamp_fgc']}"
        plot_dir = file_dir / Path('20220707_data_plots')
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / (file_name + ".png")
        file_path = file_dir / Path('20220707_data') / (file_name + ".hdf5")

        if not os.path.isfile(file_path):
            for signal_group in signal_groups:
                group = signal_group(**fpa_identifier, spark=spark)
                acquisition_to_hdf5(acquisition=group,
                                    file_dir=file_dir,
                                    context_dir_name="20220707_context",
                                    failed_queries_dir_name="20220707_failed",
                                    data_dir_name="20220707_data")
            log_acquisition(identifier=fpa_identifier, log_data={"download_complete": True}, log_path=file_dir / "20220707_context")

            signals = ['I_MEAS', 'VoltageNQPS.*U_DIODE', 'VoltageNXCALS.*U_DIODE', 'I_EARTH_PCNT', 'IEARTH.I_EARTH',
                       'U_QS0', 'U_1', 'U_2', 'I_HDS', 'U_HDS', 'EEUDumpResPM']
            data = load_from_hdf_with_regex(file_path)
            plot_hdf(data=data, column_regex=signals, fig_path=plot_path)

