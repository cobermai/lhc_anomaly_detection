import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau
from src.models import nmf as my_nmf
from src.utils.frequency_utils import get_fft_of_DataArray
from src.utils.utils import dict_to_df_meshgrid

from src.visualisation.fft_visualisation import plot_NMF


def train_NMF(data: np.array, param_grid: dict, out_path: Path, output_name: str) -> pd.DataFrame:
    """
    train non-negative matrix factorization for all values in param_grid, stores results in output path
    :param data: input data for NMF (V), dim: (events * n_magnets, n_frequencies)
    :param param_grid: dict with parameters to iterate over in this experiment,
    values must be lists containing the same datatype
    :param out_path: path to output folder
    :param output_name: name of experiment
    :return: dataframe with nmf performance measures
    """
    plot_path = out_path / f'plots_{output_name}'
    plot_path.mkdir(parents=True, exist_ok=True)

    df_meshgrid = dict_to_df_meshgrid(param_grid)
    df_results = df_meshgrid
    im_paths = []
    for index, row in df_meshgrid.iterrows():
        print(row)
        # Calculate Non-Negative Matrix Factorization (NMF)
        # W (n_samples, n_components) -> (m=3360, r=2) not (n=32, r=2)
        # H (n_components, n_features) -> (r=2, n=32) not (r=2, m=3360)
        # -> W and H are switched in scikit learn
        start_time = time.time()

        #nmf_model = NMF(**row.to_dict())
        #W, H, n_iter = nmf_model.train(X=data)

        W, H, n_iter = my_nmf.non_negative_factorization(X=data, **row.to_dict())
        passed_time = time.time() - start_time

        # log results
        df_results.loc[index, 'duration'] = passed_time
        df_results.loc[index, 'violation'] = np.linalg.norm(data - W @ H)
        df_results.loc[index, 'n_iter'] = n_iter
        df_results.loc[index, 'l2_transformation'] = np.linalg.norm(W) / row["n_components"]
        df_results.loc[index, 'l2_components'] = np.linalg.norm(H) / row["n_components"]
        df_results.loc[index, 'l2_sum'] = (np.linalg.norm(H) + np.linalg.norm(W)) / row["n_components"]
        df_results.loc[index, 'l1_transformation'] = (np.sum(W)) / row["n_components"]
        df_results.loc[index, 'l1_components'] = (np.sum(H)) / row["n_components"]
        df_results.loc[index, 'l1_sum'] = (np.sum(H) + np.sum(W)) / row["n_components"]
        df_results.loc[index, 'ortho_components'] = (np.linalg.norm(H.T @ H - np.eye(len(H[0])))) / row[
            "n_components"]

        # plot example
        event_idex = 1
        #mp3_fpa_df_subset = mp3_fpa_df[mp3_fpa_df['fpa_identifier'] == fpa_identifiers[event_idex]]
        ax = plot_NMF(data, W, H, dataset_fft.frequency, event_idex,
                      mp3_fpa_df_subset=None, hyperparameters=row.to_dict())
        ax[1, 1].set_title(f"reconstructed image \nloss: {np.linalg.norm(data - W @ H):.2f}")
        plt.tight_layout()
        im_path = plot_path / (str(index) + "_" + '_'.join(row.astype(str).values) + '.png')
        im_paths.append(im_path)
        plt.savefig(im_path)  # (str(row["n_components"]) + f"_{index}"))

        print(f"{index}/{len(df_meshgrid)} {'_'.join(row.astype(str).values)} "
              f"Loss: {np.linalg.norm(data - W @ H)} "
              f"Time: {passed_time}")

    df_results.to_csv(out_path / f'results_{output_name}.csv')

    imgs = (Image.open(f) for f in im_paths)
    img = next(imgs)  # extract first image from iterator
    img.save(fp=plot_path / "summary.gif", format='GIF', append_images=imgs, save_all=True, duration=400, loop=0)

    return df_results


if __name__ == "__main__":

    # define paths to read
    context_path = Path("../data/RB_TC_extract_2022_07_07_processed_filled.csv")

    # define paths to read + write
    dataset_path = Path('D:\\datasets\\20220707_prim_ee_plateau_dataset')
    output_path = Path(f"../output/{os.path.basename(__file__)}")  # datetime.now().strftime("%Y-%m-%dT%H.%M.%S.%f")
    output_path.mkdir(parents=True, exist_ok=True)

    # load desired fpa_identifiers
    sec_after_prim_quench = 2
    mp3_fpa_df = pd.read_csv(context_path)
    fpa_identifiers = mp3_fpa_df[(mp3_fpa_df['Delta_t(iQPS-PIC)'] / 1000 > sec_after_prim_quench) &
                                 (mp3_fpa_df['timestamp_fgc'] > 1611836512820000000)
                                 ].fpa_identifier.unique()
    dataset_creator = RBFPAPrimQuenchEEPlateau()
    dataset = dataset_creator.load_dataset(fpa_identifiers=fpa_identifiers,
                                           dataset_path=dataset_path,
                                           drop_data_vars=['simulation', 'el_position_feature', 'event_feature'])

    # calculate fft
    max_freq = 360
    dataset_fft = get_fft_of_DataArray(data=dataset.data,
                                       cutoff_frequency=max_freq)

    # postprocess fft data
    data_scaled = np.array([dataset_creator.log_scale_data(x) for x in dataset_fft.data])
    data_processed = np.nan_to_num(data_scaled.reshape(-1, np.shape(data_scaled)[2]))

    # define parameters to iterate over and start training
    experiment_name = "ortho_weight"
    experiment_param_grid = {
        "n_components": [10],
        "solver": ["mu"],
        "beta_loss": ['frobenius'],
        "init": ["nndsvda"],
        "tol": [1e-5],
        "max_iter": [1000],
        "regularization": ["both"],
        "l1_ratio": [0],
        "alpha": [0],
        "shuffle": ["True"],
        "ortho_reg": list(np.round(np.arange(1, 30)/10, 2))
    }
    df_results = train_NMF(data=data_processed,
                           param_grid=experiment_param_grid,
                           out_path=output_path,
                           output_name=experiment_name)

