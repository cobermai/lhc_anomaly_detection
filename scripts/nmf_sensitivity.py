import datetime
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from src.datasets.rb_fpa_prim_quench_ee_plateau import RBFPAPrimQuenchEEPlateau
from src.models.nmf import NMF
from src.utils.frequency_utils import get_fft_of_DataArray, scale_fft_amplitude
from src.utils.utils import dict_to_df_meshgrid, merge_array

from src.visualisation.fft_visualisation import plot_NMF, plot_nmf_components, plot_NMF_loss
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

        # fit and transform NMF, fit both W and H
        nmf_model = NMF(**row.to_dict())
        nmf_model.fit(X=data[bool_train_flattened])
        W = nmf_model.transform(X=data)
        H = nmf_model.components_
        event_loss = np.linalg.norm(np.nan_to_num(data_scaled) - (W @ H).reshape(data_scaled.shape), axis=(1, 2))

        # log results
        results = nmf_model.evaluate(X=data, W=W)
        df_results.loc[index, list(results.keys())] = results.values()


        H_norm, W_norm = nmf_model.normalize_H(H=H, W=W)
        df_components = pd.DataFrame(H_norm.T,
                                     index=frequency,
                                     columns=[f"component_{i}" for i in range(len(H_norm))])
        component_path = plot_path / ("component_" + str(index) + "_" + '_'.join(row.astype(str).values))
        df_components.to_csv(f'{component_path}.csv')

        # plot components only
        plot_nmf_components(H, dataarray_fft_amplitude,  W, loss=event_loss,
                            component_indexes=None, vmin=lower_bound, vmax=upper_bound, hyperparameters=row.to_dict())
        im_path = plot_path / (str(index) + "_" + '_'.join(row.astype(str).values) + '_scaled.png')
        im_paths.append(im_path)
        plt.tight_layout()
        plt.savefig(im_path)

        """
        merge_component_index = [[4,2], # EM perturbation
                                 [6], # 20Hz
                                 [3, 1], # 40Hz
                                 [0,5,8,9, 7, 6 ], # 65 Hz
                                 [7], # 143 Hz
                                 [1]] #noise

        H_merged = merge_array(H.T, merge_component_index, axis=-1, func=np.sum).T
        W_merged = merge_array(W, merge_component_index, axis=-1, func=np.mean)

        H_norm, W_norm = nmf_model.normalize_H(H=H_merged, W=W_merged)
        plot_nmf_components(H_merged, dataarray_fft_amplitude, W_merged, hyperparameters=row.to_dict())
        im_path = plot_path / (str(index) + "_" + '_'.join(row.astype(str).values) + '_scaled_merged.png')
        plt.tight_layout()
        plt.savefig(im_path)

        df_components = pd.DataFrame(H_norm.T,
                                     index=frequency,
                                     columns=[f"component_{i}" for i in range(len(H_norm))])
        component_path = plot_path / ("component_" + str(index) + "_" + '_'.join(row.astype(str).values))
        df_components.to_csv(f'{component_path}.csv')
        """
        # plot loss
        outlier_events = ["RB_RB.A78_1619330143440000000",
                          "RB_RB.A12_1621014819920000000",
                          "RB_RB.A45_1620797547820000000"]  # "RB_RB.A34_1620105483360000000"
        plot_NMF_loss(loss=event_loss[bool_train],
                      mp3_fpa_df_subset=mp3_fpa_df[mp3_fpa_df.fpa_identifier.isin(fpa_identifiers_train)]
                      .drop_duplicates(subset=['fpa_identifier']),
                      outlier_events=outlier_events)
        loss_im_path = plot_path / ("loss_" + str(index) + "_" + '_'.join(row.astype(str).values) + '.png')
        plt.savefig(loss_im_path)




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
    mp3_fpa_df = pd.read_csv(context_path)
    all_fpa_identifiers = mp3_fpa_df.fpa_identifier.unique()

    dataset_creator = RBFPAPrimQuenchEEPlateau()
    dataset = dataset_creator.load_dataset(fpa_identifiers=all_fpa_identifiers,
                                           dataset_path=dataset_path,
                                           drop_data_vars=['simulation', 'el_position_feature', 'event_feature'])
    # postprocess timeseries data
    dataset_detrend = dataset_creator.detrend_dim(dataset)

    # calculate fft
    f_window = np.hamming
    f_lim = (0, 360)
    dataarray_fft = get_fft_of_DataArray(data=dataset_detrend.data, f_window=f_window, f_lim=f_lim)
    dataarray_fft_amplitude = scale_fft_amplitude(data=dataarray_fft, f_window=f_window)
    dataarray_fft_amplitude = dataarray_fft_amplitude[:, :, dataarray_fft_amplitude.frequency < f_lim[1]]

    # postprocess fft data
    lower_bound = 1e-3
    upper_bound = 1
    data_scaled = np.array([dataset_creator.log_scale_data(x, vmin=lower_bound, vmax=upper_bound)
                            for x in dataarray_fft_amplitude.data])
    data_processed = np.nan_to_num(data_scaled.reshape(-1, np.shape(data_scaled)[2]))

    # all events with successfully loaded data
    fpa_identifiers = all_fpa_identifiers[np.isin(all_fpa_identifiers, dataset.event.values)]

    # model is not trained on data before 2021 and events with fast secondary quenches
    bool_fast = ((mp3_fpa_df['Delta_t(iQPS-PIC)'] / 1000 < 5) &
                 (mp3_fpa_df['Nr in Q event'].astype(str) != '1'))
    bool_R2 = (mp3_fpa_df['timestamp_fgc'] < 1611836512820000000)
    bool_test = bool_R2 | bool_fast

    fpa_identifiers_test = fpa_identifiers[np.isin(fpa_identifiers, mp3_fpa_df[bool_test].fpa_identifier.unique())]
    bool_train = ~np.isin(fpa_identifiers, fpa_identifiers_test)
    fpa_identifiers_train = fpa_identifiers[bool_train]

    # add dims for indexing flattended data
    bool_train_flattened = np.stack([bool_train for l in range(data_scaled.shape[1])]).T.reshape(-1)

    experiment_name = "12c_360Hz_1e-3"
    experiment_param_grid = {
        "n_components": [20],
        "solver": ["mu"],
        "beta_loss": ['frobenius'],
        "init": ["nndsvda"],
        "tol": [1e-5],
        "max_iter": [500],
        "l1_ratio": [0.5],
        "alpha": [0],
        "shuffle": ["False"],
        "ortho_reg": [0]
    }


    df_results = NMF_sensitivity_analysis(data=data_processed,
                                          param_grid=experiment_param_grid,
                                          frequency=dataarray_fft_amplitude.frequency,
                                          out_path=output_path,
                                          output_name=experiment_name)
