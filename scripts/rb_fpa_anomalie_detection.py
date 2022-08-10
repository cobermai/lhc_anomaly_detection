import warnings

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from src.dataset import load_dataset
from src.datasets.rb_fpa_prim_quench import RBFPAPrimQuench


warnings.filterwarnings('ignore')

if __name__ == "__main__":

    # define paths
    dataset_path = Path("/mnt/d/datasets/20220707_dataset")
    context_path = Path("../data/RB_TC_extract_2022_07_07_processed_filled_inspected.csv")
    data_path = Path("/mnt/d/datasets/20220707_data")
    simulation_path = Path("/mnt/d/datasets/20220707_simulation")

    # load dataset
    dataset = load_dataset(creator=RBFPAPrimQuench,
                           dataset_path=dataset_path,
                           context_path=context_path,
                           data_path=data_path,
                           simulation_path=simulation_path,
                           generate_dateset=True)
    print("")
