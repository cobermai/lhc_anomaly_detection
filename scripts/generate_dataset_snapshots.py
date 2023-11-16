import os
import warnings
from pathlib import Path


from src.dataset import load_dataset
from src.datasets.rb_fpa_full_quench import RBFPAFullQuench_V2
from src.datasets.rb_fpa_snapshots_ee_plateau import RBFPASnapshotsEEPlateau
from src.datasets.rb_fpa_snapshots_ee_plateau2 import RBFPASnapshotsEEPlateau2

warnings.filterwarnings('ignore')



if __name__ == "__main__":
    # define paths to read
    context_path = Path("../data/RB_snapshot_context.csv")
    metadata_path = Path("../data/RB_position_context.csv")
    data_path = Path(r"D:\datasets\20230220_snapshot_data")

    # define paths to read + write
    dataset_path = Path(r"D:\datasets\20230313_RBFPAPrimQuenchEEPlateau_V2")
    plot_dataset_path = Path(r"D:\datasets\20230313_RBFPAPrimQuenchEEPlateau_V2_plots")
    output_path = Path(f"../output/{os.path.basename(__file__)}")  # datetime.now().strftime("%Y-%m-%dT%H.%M.%S.%f")
    output_path.mkdir(parents=True, exist_ok=True)

    # load dataset
    dataset = load_dataset(creator=RBFPASnapshotsEEPlateau,
                           dataset_path=dataset_path,
                           context_path=context_path,
                           metadata_path=metadata_path,
                           acquisition_summary_path=None,
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
    dataset = load_dataset(creator=RBFPASnapshotsEEPlateau2,
                           dataset_path=dataset_path,
                           context_path=context_path,
                           metadata_path=metadata_path,
                           acquisition_summary_path=None,
                           data_path=data_path,
                           simulation_path=None,
                           plot_dataset_path=plot_dataset_path,
                           generate_dataset=True,
                           drop_data_vars=None)






