import warnings
from datetime import datetime
from pathlib import Path

from src.dataset import load_dataset
from src.datasets.rb_fpa_sec_quench import RBFPASecQuench


warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # select datasets to create
    dataset_creator = RBFPASecQuench

    # file paths in this repo
    mp3_excel_path = Path("../data/processed/MP3_context/RB_TC_extract_2023_11_17_processed.csv")
    RB_position_context_path = Path("../data/processed/RB_position_context/RB_position_context.csv")

    # external file paths
    root_dir = Path(r"D:\RB_data")  # data available at "/eos/project/m/ml-for-alarm-system/private/RB_signals/"

    # path to raw data
    quench_data_path = root_dir / Path(r"raw\20231117_data")
    # approach: manually move all signals not to use from this directory to raw\data_bad_plots
    quench_data_filtered_plots = root_dir / Path(r"raw\20231117_data_plots")

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
                 add_exp_trend_coeff=False)

