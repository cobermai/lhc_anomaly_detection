import os
import warnings
from pathlib import Path


from src.dataset import load_dataset
# from src.datasets.rb_fpa_full_quench import RBFPAFullQuench
# from src.datasets.rb_fpa_prim_quench import RBFPAPrimQuench
from src.datasets.rb_fpa_sec_quench import RBFPASecQuench
from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau
from src.datasets.rb_fpa_prim_quench_ee_plateau2 import RBFPAPrimQuenchEEPlateau2
from src.datasets.rb_fpa_udiode import RBFPAUDiode
from src.datasets.rb_fpa_snapshots_ee_plateau import RBFPASnapshotsEEPlateau
from src.datasets.rb_fpa_snapshots_ee_plateau2 import RBFPASnapshotsEEPlateau2
from src.datasets.rb_fpa_snapshots_uqs0 import RBFPASnapshotsUQS0

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # define paths to read
    context_path = Path("../data/RB_snapshot_context.csv")
    metadata_path = Path("../data/RB_metadata.csv")
    data_path = Path("D:\\datasets\\20221123_snapshots_data")
    simulation_path = Path("/mnt/d/datasets/20221123_snapshot_simulation")

    # define paths to read + write
    dataset_path = Path('D:\\datasets\\20230217_UQS0') #Path("/mnt/d/datasets/20230217_UQS0")
    plot_dataset_path = Path('D:\\datasets\\20230217_UQS0')
    output_path = Path(f"../output/{os.path.basename(__file__)}")
    output_path.mkdir(parents=True, exist_ok=True)

    # generate dataset
    dataset = load_dataset(creator=RBFPASnapshotsUQS0,
                           dataset_path=dataset_path,
                           context_path=context_path,
                           metadata_path=metadata_path,
                           data_path=data_path,
                           simulation_path=simulation_path,
                           plot_dataset_path=plot_dataset_path,
                           generate_dataset=True,
                           drop_data_vars=None
                           )





