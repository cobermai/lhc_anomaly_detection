import warnings
from datetime import datetime
from pathlib import Path

from src.dataset import load_dataset
from src.datasets.rb_fpa_prim_quench_ee_plateau1 import RBFPAPrimQuenchEEPlateau1
from src.datasets.rb_fpa_prim_quench_ee_plateau2 import RBFPAPrimQuenchEEPlateau2

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # select datasets to create
    dataset_creators = [RBFPAPrimQuenchEEPlateau1, RBFPAPrimQuenchEEPlateau2]

    # file paths in this repo
    mp3_excel_path = Path("../data/processed/MP3_context/RB_TC_extract_2023_03_13_processed.csv")
    RB_snapshot_context_path = Path("../data/processed/RB_snapshot_context/RB_snapshot_context.csv")
    RB_position_context_path = Path("../data/processed/RB_position_context/RB_position_context.csv")

    # external file paths
    root_dir = Path(r"D:\RB_data")  # data available at "/eos/project/m/ml-for-alarm-system/private/RB_signals/"

    # path to raw data
    quench_data_path = root_dir / Path(r"raw\20230313_data")
    snapshot_data_path = root_dir / Path(r"raw\20221123_snapshots_data")

    # approach: manually move all signals not to use from this directory to raw\data_bad_plots
    quench_data_filtered_plots = root_dir / Path(r"raw\20230313_data_plots")
    snapshot_data_filtered_plots = root_dir / Path(r"raw\20221123_snapshots_data_plots")

    for dataset_creator in dataset_creators:
        # output file paths
        dataset_name = f"{datetime.now().strftime('%Y%m%d')}_{dataset_creator.__name__}"
        dataset_path = root_dir / Path(f"processed/{dataset_name}")
        plot_dataset_path = root_dir / Path(f"processed/{dataset_name}_plots")

        # load quench data
        load_dataset(creator=dataset_creator,
                     dataset_path=dataset_path,
                     context_path=mp3_excel_path,
                     metadata_path=RB_position_context_path,
                     acquisition_summary_path=quench_data_filtered_plots,
                     data_path=quench_data_path,
                     plot_dataset_path=plot_dataset_path,
                     generate_dataset=True,
                     add_exp_trend_coeff=True)

        # load snapshot data
        load_dataset(creator=dataset_creator,
                     dataset_path=dataset_path,
                     context_path=RB_snapshot_context_path,
                     metadata_path=RB_position_context_path,
                     acquisition_summary_path=snapshot_data_filtered_plots,
                     data_path=snapshot_data_path,
                     plot_dataset_path=plot_dataset_path,
                     generate_dataset=True,
                     add_exp_trend_coeff=True)
