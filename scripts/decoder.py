import os
import warnings
import sys

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# from lhcsmapi.Timer import Timer
import tensorflow as tf
from matplotlib import pyplot as plt

from src.dataset import load_dataset
from src.datasets.rb_fpa_full_quench import RBFPAFullQuench
# from src.datasets.rb_fpa_prim_quench import RBFPAPrimQuench
# from src.datasets.rb_fpa_sec_quench import RBFPASecQuench
# from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau
#from src.model import Model
from src.decoder import Model
from src.models import ae1d_3e_3d

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # define paths to read
    context_path = Path("../data/RB_TC_extract_2022_07_07_processed_filled.csv")
    acquisition_summary_path = Path("../data/20220707_acquisition_summary.xlsx")
    data_path = Path("/mnt/d/datasets/20220707_data")
    simulation_path = Path("/mnt/d/datasets/20220707_simulation")

    # define paths to read + write
    dataset_path = Path("/mnt/d/datasets/20220707_full_dataset_subsampled")
    plot_dataset_path = Path("/mnt/d/datasets/20220707_full_plots_subsampled")
    output_path = Path(f"../output/{os.path.basename(__file__)}")  # datetime.now().strftime("%Y-%m-%dT%H.%M.%S.%f")
    output_path.mkdir(parents=True, exist_ok=True)

    # load dataset
    train, valid, test = load_dataset(creator=RBFPAFullQuench,
                                      dataset_path=dataset_path,
                                      context_path=context_path,
                                      acquisition_summary_path=acquisition_summary_path,
                                      data_path=data_path,
                                      simulation_path=simulation_path,
                                      plot_dataset_path=plot_dataset_path,
                                      generate_dataset=False)

    X = np.nan_to_num(train.X[:, 0, :, ::5].values)
    X_sim = np.nan_to_num(train.X[:, 1, :, ::5].values)
    decoder = Model(input_shape=np.shape(X[0]),
                    output_directory=output_path,
                    model=ae1d_3e_3d,
                    epochs=30,
                    batch_size=32)

    fit_model = True #not Path(output_path / "ae_weights.h5").exists()
    if fit_model:
        decoder.fit_model(X=X, context=train.context)
    else:
        decoder.model.build(input_shape=np.shape(X[0]))
        decoder.model.load_weights(str(output_path / 'ae_weights.h5'))

    plot_path = output_path / "reconstructions"
    plot_path.mkdir(parents=True, exist_ok=True)
    X_rec = decoder.model(train.context[:, :13]).numpy()
    for j in range(5):
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].plot(X[j].T)
        ax[0].set_title("Data")
        ax[0].set_xlabel("Samples")
        ax[0].set_ylabel("Normalized Voltage ")
        ax[1].plot(X_sim[j].T)
        ax[1].set_title("Simulation")
        ax[1].set_xlabel("Samples")
        ax[1].set_ylabel("Normalized Voltage ")
        ax[2].plot(X_rec[j].T)
        ax[2].set_title("Reconstruction")
        ax[2].set_xlabel("Samples")
        ax[2].set_ylabel("Normalized Voltage ")

        plt.tight_layout()
        plt.savefig(str(plot_path / f'reconstruction_{j}.png'))

        error_sim = (X[j] - X_sim[j]).sum(axis=0)
        error_rec = (X[j] - X_rec[j]).sum(axis=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(error_sim, label=f"error simulation: {int(np.abs(error_sim).sum())}")
        plt.plot(error_rec, label=f"error reconstruction: {int(np.abs(error_rec).sum())}")
        plt.legend()
        plt.grid()
        plt.title("Error")
        plt.xlabel("Samples")
        plt.ylabel("Normalized Voltage ")
        plt.tight_layout()
        plt.savefig(str(plot_path / f'error_{j}.png'))
