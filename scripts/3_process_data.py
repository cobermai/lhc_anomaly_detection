import os
import warnings
from pathlib import Path


from src.dataset import load_dataset
#from src.datasets.rb_fpa_full_quench_V2 import RBFPAFullQuench
from src.datasets.rb_fpa_prim_quench_ee_plateau2 import RBFPAPrimQuenchEEPlateau2
from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # define paths to read
    context_path = Path("../data/MP3_context/RB_TC_extract_2023_03_13_processed.csv")
    acquisition_summary_path = Path("../data/20230313_acquisition_summary.csv")
    metadata_path = Path("../data/RB_position_context.csv")
    data_path = Path(r"D:\datasets\20230313_data")

    # define paths to read + write
    dataset_path = Path(r"D:\datasets\20230313_RBFPAPrimQuenchEEPlateau")
    plot_dataset_path = Path(r"D:\datasets\20230313_RBFPAPrimQuenchEEPlateau_plots")
    output_path = Path(f"../output/{os.path.basename(__file__)}")  # datetime.now().strftime("%Y-%m-%dT%H.%M.%S.%f")
    output_path.mkdir(parents=True, exist_ok=True)


    # load dataset
    dataset = load_dataset(creator=RBFPAPrimQuenchEEPlateau,
                           dataset_path=dataset_path,
                           context_path=context_path,
                           metadata_path=metadata_path,
                           acquisition_summary_path=acquisition_summary_path,
                           data_path=data_path,
                           simulation_path=None,
                           plot_dataset_path=plot_dataset_path,
                           generate_dataset=True,
                           drop_data_vars=None)


    # define paths to read + write
    dataset_path = Path(r"D:\datasets\20230313_RBFPAPrimQuenchEEPlateau2_V2")
    plot_dataset_path = Path(r"D:\datasets\20230313_RBFPAPrimQuenchEEPlateau2_V2_plots")
    output_path = Path(f"../output/{os.path.basename(__file__)}")  # datetime.now().strftime("%Y-%m-%dT%H.%M.%S.%f")
    output_path.mkdir(parents=True, exist_ok=True)

    # load dataset
    dataset = load_dataset(creator=RBFPAPrimQuenchEEPlateau2_V2,
                           dataset_path=dataset_path,
                           context_path=context_path,
                           metadata_path=metadata_path,
                           acquisition_summary_path=acquisition_summary_path,
                           data_path=data_path,
                           simulation_path=None,
                           plot_dataset_path=plot_dataset_path,
                           generate_dataset=True,
                           drop_data_vars=None)





