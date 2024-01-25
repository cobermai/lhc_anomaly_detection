import sys
sys.path.insert(0,'..')

from pathlib import Path

from nxcals.spark_session_builder import get_or_create, Flavor

from data.processed.MP3_context.process_MP3_excel import process_mp3_excel
from data.processed.RB_position_context.generate_RB_position_context import generate_RB_position_context
from data.processed.RB_snapshot_context.generate_RB_snapshot_context import generate_RB_snapshot_context
from data.processed.position_mapping.generate_position_mapping import generate_position_mapping

if __name__ == "__main__":
    spark = get_or_create(flavor=Flavor.YARN_MEDIUM)

    data_dir = Path("../data/raw/MP3_context/")
    output_dir = Path("../data/processed/MP3_context/")
    file_name = "RB_TC_extract_2023_03_13"
    process_mp3_excel(input_dir=data_dir, output_dir=output_dir, mp3_file_name=file_name)

    data_dir = Path("../data/raw/MP3_context/")
    output_dir = Path("../data/processed/RB_snapshot_context/")
    file_name = "RB_snapshot_context"
    generate_RB_snapshot_context(input_dir=data_dir, output_dir=output_dir, file_name=file_name)

    year = 2021
    data_dir = Path("../data/raw")
    RB_LayoutDetails_path = data_dir / "SIGMON_context/RB_LayoutDetails.csv"
    nQPS_RB_busBarProtectionDetails_path = data_dir / "SIGMON_context/nQPS_RB_busBarProtectionDetails.csv"
    RB_Beamscreen_Resistances_path = data_dir / "MB_feature_analysis/RB_Beamscreen_Resistances.csv"
    RB_Magnet_metadata_path = data_dir / "MP3_context/RB_TC_extract_2023_03_13.xlsx"
    output_dir = Path("../data/processed/RB_position_context/")
    generate_RB_position_context(year=year,
                                 RB_LayoutDetails_path=RB_LayoutDetails_path,
                                 nQPS_RB_busBarProtectionDetails_path=nQPS_RB_busBarProtectionDetails_path,
                                 RB_Beamscreen_Resistances_path=RB_Beamscreen_Resistances_path,
                                 RB_Magnet_metadata_path=RB_Magnet_metadata_path,
                                 output_dir=output_dir)

    data_dir = Path("../data/processed/RB_position_context/")
    file_name = "RB_position_context"
    generate_position_mapping(input_dir=data_dir, output_dir=data_dir, file_name=file_name)