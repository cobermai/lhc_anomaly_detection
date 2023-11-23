import gc
import os
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal

from src.datasets.rb_fpa_prim_quench_ee_plateau1 import RBFPAPrimQuenchEEPlateau1
from src.models.nmf import NMF
from src.result import SensitivityAnalysis
from src.results.nmf_result import NMFResult
from src.utils.frequency_utils import get_fft_of_DataArray, scale_fft_amplitude
from src.utils.utils import dict_to_df_meshgrid
from src.visualisation.NMF_visualization import plot_outlier_events, plot_NMF_components


def NMF_sensitivity_analysis(hyperparameter, out_path):
    df_meshgrid = dict_to_df_meshgrid(hyperparameter)
    analysis = SensitivityAnalysis(result_path=out_path,
                                   event_identifiers=ds.event.values[bool_train])
    for index, row in df_meshgrid.iterrows():
        start_time = time.time()

        experiment_name = '_'.join(row.astype(str).values)
        print(f"{index}/{len(df_meshgrid)} {experiment_name}")

        # Preprocessing
        ds_detrend = dataset_creator.detrend_dim(ds, deg=row["trend_deg"])
        f_window = window_functions[row["f_window"]]
        da_processed = ds_detrend.data * f_window(len(ds.time))

        # Calculate fft
        f_lim = (0, 534)
        da_fft = get_fft_of_DataArray(data=da_processed, f_lim=f_lim)
        da_fft_amp = scale_fft_amplitude(data=da_fft, f_window=f_window)
        da_fft_amp = da_fft_amp[:, :, da_fft_amp.frequency < f_lim[1]]

        # Flatten data for nmf
        na_fft_flat = np.nan_to_num(da_fft_amp.data.reshape(-1, np.shape(da_fft_amp.data)[2]))
        print(f"Time till postprocessing: {time.time() - start_time}") #6s

        # Train NMF
        fit_model = False
        nmf_model = NMF(**row.drop(["trend_deg", "f_window"]).to_dict())
        nmf_result = NMFResult(out_path=out_path,
                               name=experiment_name,
                               **row.to_dict())
        if fit_model:
            component_weights = nmf_model.fit_transform(X=na_fft_flat[bool_train_flat])
            nmf_result.set_result(components=nmf_model.components_)
        else:
            input_path = Path(f"../output/{os.path.basename(__file__)}/2023-11-22_refitted") / experiment_name
            nmf_result.load(input_path=input_path)
            nmf_model.components_ = nmf_result.components
            component_weights = nmf_model.transform(X=na_fft_flat[bool_train_flat])

        # store results
        #nmf_result.update_results(component_weights=component_weights)
        nmf_result.set_result(component_weights=component_weights)
        nmf_result.calculate_p_values(da_fft_amp.data[bool_train], plot_fit=True)
        nmf_result.save()
        analysis.add_class_result(nmf_result)

        # plot components
        plot_NMF_components(da_fft_amp.frequency, nmf_model.components_, out_path / experiment_name / 'components.png')

        # free allocated storage
        plt.close('all')
        del ds_detrend, da_processed, da_fft, da_fft_amp, na_fft_flat
        gc.collect()

        print(f"Time till fit & transform NMF: {time.time() - start_time}") #11s

    df_outliers = analysis.get_outlier_events(n_outliers=10, plot_outliers=True)
    plot_outlier_events(ds.data, df_outliers, out_path)  # ds.data


if __name__ == "__main__":
    # define paths to read
    mp3_excel_path = Path("../data/processed/MP3_context/RB_TC_extract_2023_03_13_processed.csv")
    snapshot_context_path = Path("../data/processed/RB_snapshot_context/RB_snapshot_context.csv")

    # external file paths
    root_dir = Path(r"D:\RB_data")  # data available at "/eos/project/m/ml-for-alarm-system/private/RB_signals/"

    # data available at "/eos/project/m/ml-for-alarm-system/private/RB_signals/
    dataset_path = root_dir / Path(r'processed\20230313_RBFPAPrimQuenchEEPlateau1')

    # approach: manually move all signals not to use from this directory to raw\data_bad_plots
    quench_data_filtered_plots = root_dir / Path(r"raw\20230313_data_plots")

    # define paths to read + write %Y-%m-%dT%H.%M.%S.%f
    output_path = Path(f"../output/{os.path.basename(__file__)}/{datetime.now().strftime('%Y-%m-%dT%H.%M.%S.%f')}")
    #output_path = Path(f"../output/{os.path.basename(__file__)}/2023-11-22_refitted")
    output_path.mkdir(parents=True, exist_ok=True)

    # load desired fpa_identifiers
    mp3_fpa_df = pd.read_csv(mp3_excel_path)
    mp3_fpa_df_unique = mp3_fpa_df[mp3_fpa_df['timestamp_fgc'] >= 1526582397220000000].drop_duplicates(
        subset=['fpa_identifier']).dropna(
        subset=['fpa_identifier'])
    dataset_creator = RBFPAPrimQuenchEEPlateau1()
    ds = dataset_creator.load_dataset(fpa_identifiers=mp3_fpa_df_unique.fpa_identifier.values,
                                      dataset_path=dataset_path)

    # model is not trained on data before 2021 and events with fast secondary quenches
    test_conditions = ((mp3_fpa_df['Delta_t(iQPS-PIC)'] / 1000 < 5) &
                       (mp3_fpa_df['Nr in Q event'].astype(str) != '1'))
    bool_train = ~np.isin(ds.event.values, mp3_fpa_df[test_conditions].fpa_identifier.unique())
    train_events = ds.event.values[bool_train]

    # add dims for indexing flattened data
    bool_train_flat = np.stack([bool_train for l in range(len(ds.el_position))]).T.reshape(-1)

    window_functions = {"hamming": signal.windows.hamming}

    hyperparameter = {
        "trend_deg": [1],
        "f_window": list(window_functions.keys()),
        "n_components": [6, 7],
        "solver": ["mu"],
        "beta_loss": ['frobenius', 'kullback-leibler'],
        "init": ["nndsvda"],
        "shuffle": ["False"]
    }
    NMF_sensitivity_analysis(hyperparameter, output_path)

"""
    window_functions = {"ones": np.ones,
                        "hanning": np.hanning,
                        "bartlett": signal.windows.bartlett,
                        "blackman": signal.windows.blackman,
                        "flattop": signal.windows.flattop,
                        "hamming": signal.windows.hamming,
                        "tukey": signal.windows.tukey}

    hyperparameter = {
        "trend_deg": [0, 1],
        "f_window": list(window_functions.keys()),
        "n_components": [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "solver": ["mu"],
        "beta_loss": ['frobenius', 'kullback-leibler'],
        "init": ["nndsvda"],
        "shuffle": ["False"]
    }
    NMF_sensitivity_analysis(hyperparameter, output_path)
"""
