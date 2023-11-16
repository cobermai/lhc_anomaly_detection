from datetime import datetime
import os
from pathlib import Path

import pandas as pd
from nxcals.spark_session_builder import get_or_create, Flavor

from src.acquisitions.voltage_nqps import VoltageNQPS
from src.utils.hdf_tools import acquisition_to_hdf5, load_from_hdf_with_regex
from src.utils.fgc_timestamp_utils import select_fgc_period
from src.utils.utils import log_acquisition
from src.visualisation.visualisation import plot_hdf

if __name__ == "__main__":
    spark = get_or_create(flavor=Flavor.YARN_MEDIUM)
    file_dir = Path('/eos/project/m/ml-for-alarm-system/private/RB_signals')
    date_suffix = datetime.now().strftime('%Y%m%d')
    signal_groups = [VoltageNQPS]  # select all signals to download from src/acquisition

    # get list of fpa_identifiers
    mp3_excel_path = Path("../data/processed/MP3_context/RB_TC_extract_2023_03_13_processed.csv")
    mp3_fpa_df = pd.read_csv(mp3_excel_path)

    # secondary quenches have same timestamp as primary quenches
    mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['fpa_identifier']).dropna(subset=['fpa_identifier'])
    mp3_fpa_df_period = select_fgc_period(mp3_fpa_df_unique,
                                          lower_threshold='2014-01-01 00:00:00+01:00',
                                          upper_threshold='2024-01-01 00:00:00+01:00')


    for index, row in mp3_fpa_df_period.iterrows():

        fpa_identifier = {'circuit_type': row['Circuit Family'],
                          'circuit_name': row['Circuit Name'],
                          'timestamp_fgc': int(row['timestamp_fgc'])}
        plot_dir = file_dir / Path(f'{date_suffix}_data_plots')
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / (row['fpa_identifier'] + ".png")
        file_path = file_dir / Path(f'{date_suffix}_data') / (row['fpa_identifier'] + ".hdf5")

        if not os.path.isfile(file_path):
            for signal_group in signal_groups:
                group = signal_group(**fpa_identifier, spark=spark)
                acquisition_to_hdf5(acquisition=group,
                                    file_dir=file_dir,
                                    context_dir_name=f"{date_suffix}_context",
                                    failed_queries_dir_name=f"{date_suffix}_failed",
                                    data_dir_name=f"{date_suffix}_data")
            log_acquisition(identifier=fpa_identifier, log_data={"download_complete": True},
                            log_path=file_dir / f"{date_suffix}_context")

            if os.path.isfile(file_path):
                signals = ['VoltageNQPS.*U_DIODE']
                data = load_from_hdf_with_regex(file_path)
                plot_hdf(data=data, column_regex=signals, fig_path=plot_path)



