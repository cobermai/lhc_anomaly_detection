# Authors: Vlad Niculae, Alexandre Gramfort
# License: BSD 3 clause
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau
from src.models import nmf_missing as my_nmf
from src.utils.frequency_utils import get_fft_of_DataArray
from src.visualisation.fft_visualisation import plot_NMF_components

if __name__ == "__main__":
    # define paths to read
    context_path = Path("../data/RB_TC_extract_2022_07_07_processed_filled.csv")
    metadata_path = Path("../data/RB_metadata.csv")
    acquisition_summary_path = Path("../data/20220707_acquisition_summary.xlsx")
    data_path = Path("/mnt/d/datasets/20220707_data")
    simulation_path = Path("/mnt/d/datasets/20220707_simulation")

    # define paths to read + write
    dataset_path = Path('D:\\datasets\\20220707_prim_ee_plateau_dataset')
    plot_dataset_path = Path("/mnt/d/datasets/1EE_plateau_test")
    #dataset_path = Path("/mnt/d/datasets/20220707_RBFPAPrimQuenchEEPlateau2")
    #plot_dataset_path = Path("/mnt/d/datasets/20220707_RBFPAPrimQuenchEEPlateau2_plots")
    output_path = Path(f"../output/{os.path.basename(__file__)}")  # datetime.now().strftime("%Y-%m-%dT%H.%M.%S.%f")
    output_path.mkdir(parents=True, exist_ok=True)

    # initialize dataset
    dataset_creator = RBFPAPrimQuenchEEPlateau
    ds = dataset_creator(dataset_path=dataset_path,
                         context_path=context_path,
                         metadata_path=metadata_path,
                         data_path=data_path,
                         simulation_path=simulation_path,
                         acquisition_summary_path=acquisition_summary_path,
                         plot_dataset_path=plot_dataset_path)

    # load desired fpa_identifiers
    mp3_fpa_df = pd.read_csv(context_path)
    sec_after_prim_quench = 2
    fpa_identifiers = mp3_fpa_df[(mp3_fpa_df['Delta_t(iQPS-PIC)'] / 1000 > sec_after_prim_quench) &
                                 (mp3_fpa_df['timestamp_fgc'] > 1611836512820000000)
                                 ].fpa_identifier.unique()
    dataset = ds.load_dataset(fpa_identifiers=fpa_identifiers,
                              dataset_path=dataset_path,
                              drop_data_vars=['simulation', 'el_position_feature', 'event_feature'])

    # calculate fft
    max_freq = 360
    dataset_fft = get_fft_of_DataArray(data=dataset.data,
                                       cutoff_frequency=max_freq)

    # postprocess fft data
    data_scaled = np.array([RBFPAPrimQuenchEEPlateau.log_scale_data(x) for x in dataset_fft.data])
    data = np.nan_to_num(data_scaled.reshape(-1, np.shape(data_scaled)[2]))

    # Calculate Non-Negative Matrix Factorization (NMF)
    # W (n_samples, n_components) -> (m=3360, r=2) not (n=32, r=2)
    # H (n_components, n_features) -> (r=2, n=32) not (r=2, m=3360)
    # -> W and H are switched in scikit learn
    n_components = 2
    W, H, n_iter, violation = my_nmf.non_negative_factorization(X=data,
                                                                n_components=n_components,
                                                                init='nndsvda',
                                                                tol=1e-4,
                                                                max_iter=1000)
    event_idex = 1
    mp3_fpa_df_subset = mp3_fpa_df[mp3_fpa_df['fpa_identifier'] == fpa_identifiers[event_idex]]
    plot_NMF_components(dataset_fft, dataset_fft.frequency, W, H, event_idex, mp3_fpa_df_subset)
    plt.show()
    print(f"loss: {violation}")

