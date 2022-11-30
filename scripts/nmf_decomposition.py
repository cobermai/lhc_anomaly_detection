import os
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau
from src.models.nmf import NMF
from src.utils.frequency_utils import get_fft_of_DataArray
from src.utils.utils import dict_to_df_meshgrid, merge_array

from src.visualisation.fft_visualisation import plot_NMF
from src.visualisation.visualisation import make_gif


def NMF_sensitivity_analysis(data: np.array,
                             param_grid: dict,
                             out_path: Path,
                             output_name: str,
                             frequency: Optional[pd.DataFrame] = None,
                             event_context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    train non-negative matrix factorization for all values in param_grid, stores results in output path
    In the used code the variables W, H are denoted inversely,
    i.e. H are the components, and W are the component weights
    :param data: input data for NMF (V), dim: (events * n_magnets, n_frequencies)
    :param param_grid: dict with parameters to iterate over in this experiment,
    values must be lists containing the same datatype
    :param out_path: path to output folder
    :param frequency: array with n_frequencies frequency values
    :param event_context: dataframe with context for events
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
        im_path = plot_path / (str(index) + "_" + '_'.join(row.astype(str).values) + '.png')

        # fit and transform NMF, fit both W and H
        nmf_model = NMF(**row.to_dict())
        W = nmf_model.fit_transform(X=data)
        H = nmf_model.components_
        plot_NMF(data, W, H, frequency=frequency, event_context=event_context, hyperparameters=row.to_dict())
        plt.title("normal")
        plt.show()

        merge_component_index = [0, [2, 6], [1, 3, 4, 7, 8], 5, 9]
        H_merged = merge_array(H.T, merge_component_index, axis=-1, func=np.sum).T
        W_merged = merge_array(W, merge_component_index, axis=-1, func=np.mean)
        plot_NMF(data, W_merged, H_merged, frequency=frequency, event_context=event_context, hyperparameters=row.to_dict())
        plt.title("merged components")
        plt.show()

        # fit and transform NMF, fit W, init with part of H

        H_new = np.zeros_like(H)

        H_new[:len(H_merged)] = H_merged
        nmf_model = NMF(**row.to_dict())
        nmf_model.fit(X=data, H=H_new, not_init_H_idx=[0, 1, 2, 3, 4]) # init with part of existing H, everything is still trained
        W = nmf_model.transform(X=data)
        H = nmf_model.components_
        plot_NMF(data, W, H, frequency=frequency, event_context=event_context, hyperparameters=row.to_dict())
        plt.title("init with part of H")
        plt.show()

        # fit and transform NMF, fit W, not train part of H, cd
        nmf_model = NMF(**row.to_dict())
        nmf_model.fit(X=data, H=H_new, not_fit_H_idx=[0, 1, 2, 3, 4]) # init with part of existing H, everything is still trained
        W = nmf_model.transform(X=data)
        H = nmf_model.components_
        plot_NMF(data, W, H, frequency=frequency, event_context=event_context, hyperparameters=row.to_dict())
        plt.title("not train part of H, cd")
        plt.show()

        # log results
        #results = nmf_model.evaluate(X=data, W=W)
        #df_results.loc[index, list(results.keys())] = results.values()

        # plot example
        plot_NMF(data, W, H, frequency=frequency, event_context=event_context, hyperparameters=row.to_dict())
        im_path = plot_path / (str(index) + "_" + '_'.join(row.astype(str).values) + '.png')
        im_paths.append(im_path)
        plt.savefig(im_path)

    df_results.to_csv(out_path / f'results_{output_name}.csv')
    make_gif(im_paths=im_paths, output_path=plot_path)

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
    all_fpa_identifiers_mp3 = mp3_fpa_df[(mp3_fpa_df['timestamp_fgc'] > 1611836512820000000)].fpa_identifier.unique()
    fpa_identifiers_fast_sec_quench = mp3_fpa_df[(mp3_fpa_df['Delta_t(iQPS-PIC)'] / 1000 < 5) & (
                mp3_fpa_df['Nr in Q event'].astype(str) != '1')].fpa_identifier.unique()
    all_fpa_identifiers = all_fpa_identifiers_mp3[~np.isin(all_fpa_identifiers_mp3, fpa_identifiers_fast_sec_quench)]

    dataset_creator = RBFPAPrimQuenchEEPlateau()
    dataset = dataset_creator.load_dataset(fpa_identifiers=all_fpa_identifiers,
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
    experiment_name = "test"
    experiment_param_grid = {
        "n_components": [10],
        "solver": ["cd"],
        "beta_loss": ['frobenius'],
        "init": ["nndsvd"],
        "tol": [1e-5],
        "max_iter": [200],
        "l1_ratio": [0.5],
        "alpha": [1],
        "shuffle": ["True"],
        "ortho_reg": [0]  # list(np.round(np.arange(1, 30)/10, 2))
    }
    df_results = NMF_sensitivity_analysis(data=data_processed,
                                          param_grid=experiment_param_grid,
                                          frequency=dataset_fft.frequency,
                                          out_path=output_path,
                                          output_name=experiment_name)

