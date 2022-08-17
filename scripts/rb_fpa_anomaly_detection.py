import warnings

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from lhcsmapi.Timer import Timer

from src.dataset import load_dataset
from src.datasets.rb_fpa_full_quench import RBFPAFullQuench
from src.datasets.rb_fpa_prim_quench import RBFPAPrimQuench
from src.datasets.rb_fpa_sec_quench import RBFPASecQuench
from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # define paths
    context_path = Path("../data/RB_TC_extract_2022_07_07_processed_filled.csv")
    acquisition_summary_path = Path("../data/20220707_acquisition_summary.xlsx")
    data_path = Path("/mnt/d/datasets/20220707_data")
    simulation_path = Path("/mnt/d/datasets/20220707_simulation")

    # define dataset paths
    dataset_path = Path("/mnt/d/datasets/20220707_full_dataset")
    plot_dataset_path = Path("/mnt/d/datasets/20220707_full_plots")

    # load dataset
    with Timer():
        dataset = load_dataset(creator=RBFPAFullQuench,
                               dataset_path=dataset_path,
                               context_path=context_path,
                               acquisition_summary_path=acquisition_summary_path,
                               data_path=data_path,
                               simulation_path=simulation_path,
                               plot_dataset_path=plot_dataset_path,
                               generate_dataset=True)


    print("asd")
