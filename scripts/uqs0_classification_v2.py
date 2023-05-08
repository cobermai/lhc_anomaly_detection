import os
from datetime import datetime
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from src.utils.dataset_utils import u_diode_data_to_df
from src.utils.hdf_tools import load_from_hdf_with_regex
from src.utils.uqs0_classification_utils import plot_confusion_signals, plot_confusion_matrix, WindowSlice, write_excel


def load_uqs0_data_from_hdf5(df_events: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """
    loads uqs0 data from hdf5 given DataFrame with columns 'Circuit' and 'timestamp_fgc'
    :param df_events: dataframe with columns 'Circuit' and 'timestamp_fgc'
    :param file_path: path to hdf5 files
    :return: DataFrame with magnets as columns and time as index
    """
    data_list = []
    data_columns = []
    for j, row in df_events.iterrows():
        file_name = f"RB_{row['Circuit']}_{row['timestamp_fgc']}"
        data = load_from_hdf_with_regex(file_path / (file_name + ".hdf5"), regex_list=['U_QS0'])

        df = u_diode_data_to_df(data, len_data=len(data[0]))

        df = df.iloc[96:]
        magnets = [x.split("_")[0] for x in df.filter(regex="_A").columns.values]
        data_columns.append(magnets)

        data_list.append(np.vstack([df.filter(regex="_A"), df.filter(regex="_B")]).T)
    na_data = np.stack(data_list)
    data = na_data.reshape(-1, na_data.shape[-1])

    df_data = pd.DataFrame(data.T,
                           columns=np.array(data_columns).reshape(-1),
                           index=np.hstack([df.index.values, df.index.values]))

    print(f"Length of one Signal: {len(df)}")
    return df_data


def calc_features(X: np.array, window_size: int) -> np.array:
    """
    calculate features for each window in X
    :param X: Input data
    :param window_size: size of window, if 1 the dataset is not divided into subsets
    :return: array with features
    """
    n_features = 6
    X_features = np.zeros((len(X), window_size * n_features))
    for i, X_split in enumerate(np.array_split(X.T, window_size)):
        X_features[:, i] = np.nanquantile(X_split, 0.95, axis=0)
        X_features[:, 1 * window_size + i] = np.nanquantile(X_split, 0.75, axis=0)
        X_features[:, 2 * window_size + i] = np.nanquantile(X_split, 0.5, axis=0)
        X_features[:, 3 * window_size + i] = np.nanquantile(X_split, 0.25, axis=0)
        X_features[:, 4 * window_size + i] = np.nanquantile(X_split, 0.05, axis=0)

        linreg = [stats.linregress(np.arange(0, len(signal)), signal) for signal in X_split.T]
        X_features[:, 5 * window_size + i] = [l[0] for l in linreg]

    # X_features = (X_features - X_features.mean(axis=0)) / X_features.std(axis=0)
    return X_features


if __name__ == "__main__":
    # Define output path
    output_path = Path(f"../output/{os.path.basename(__file__)}/{datetime.now().strftime('%Y-%m-%dT%H.%M.%S.%f')}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Load context data
    df_context_path = Path("../data/RB_snapshot_context.csv")
    df_context = pd.read_csv(df_context_path)
    current_level = 11000
    dI_dt = 10
    df_context_subset = df_context[(df_context["I_Q_M"] > current_level - 500) &
                                   (df_context["I_Q_M"] < current_level + 500) &
                                   (df_context["dI_dt_from_data"] > dI_dt - 2) &
                                   (df_context["dI_dt_from_data"] < dI_dt + 2)]

    # Load labels
    true_label_name = "manual classification"
    pred_label_name = "ML classification"
    label_names = ["normal",
                   "long",
                   "very_long",
                   "boardA_boardB"]

    labels_path = Path("../data/UQS0_labels/dest_file_v02.xlsx")
    df_event_labels = pd.read_excel(labels_path)
    df_event_labels["magnet"] = df_event_labels["event"].apply(lambda x: x.split("_")[0])
    df_magnet_labels = df_event_labels.set_index('magnet')[
        label_names]  # generate_magnet_labels_from_event(df_event_labels, label_names)

    experiment_path = Path("../data/UQS0_labels/Outliers_summary_Meas_2021_2022_NoZero.xlsx")
    df_all_experiments = pd.read_excel(experiment_path)
    df_experiment = df_all_experiments[(df_all_experiments["Current_level"] == current_level) &
                                       (df_all_experiments["Expected_dI_dt"] == dI_dt)]

    # Load data
    data_path = Path("D:/datasets/20230220_snapshot_data")  # Path('/eos/project/s/steam/measurement_database/RB/data')
    df_data = load_uqs0_data_from_hdf5(df_context_subset, data_path)[df_magnet_labels.index.values]
    # Process data
    clip = 0.15
    df_data[(df_data > clip)] = clip
    df_data[df_data < -clip] = -clip

    data = df_data.T.values
    X = np.nan_to_num(data)
    target_names = label_names
    y = df_magnet_labels.loc[df_magnet_labels.index.values, target_names].values

    # Select classification parameter
    augment_data = True
    use_raw_ts = False
    n_signals = 2  # board A board B
    classifiers = {
        # "Neural_Net": MLPClassifier(hidden_layer_sizes=(32, 3), max_iter=10000),
        "Nearest_Neighbors": KNeighborsClassifier(5, weights="distance", p=3),
        # "RBF_SVM": SVC(gamma=2, C=1),
        # "Gaussian_Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
        # "Decision_Tree": DecisionTreeClassifier(max_depth=5),
        #"Random_Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # "AdaBoost": AdaBoostClassifier(),
        # "Naive_Bayes": GaussianNB(),
        # "QDA": QuadraticDiscriminantAnalysis(),
    }

    # Leave one out cross validation
    skf = StratifiedKFold(n_splits=8, random_state=0, shuffle=True)

    df_accuracy = pd.DataFrame(columns=label_names, index=list(classifiers.keys()))
    # Iterate over classifiers
    for name, clf in classifiers.items():
        print(f"{name}:")
        clf_path = output_path / name
        df_result = pd.DataFrame(index=df_experiment.event.values)

        # Iterate over labels - one vs. rest classification
        for l, label_name in enumerate(label_names):
            target_names = [f"other label", label_name]

            df_kneighbors = pd.DataFrame(index=df_experiment.event.values)

            # Iterate over data splits
            for i, (train_index, test_index) in enumerate(skf.split(X, y[:, l])):
                fold_path = clf_path / Path(f"fold_{i}")
                fold_path.mkdir(parents=True, exist_ok=True)

                # Define data in this iteration
                if augment_data:
                    ws = WindowSlice(labels=np.eye(len(target_names)),
                                     none_flat_shape=(len(X[train_index]), n_signals, int(X.shape[-1] / 2)),
                                     reduce_ratio=0.9)
                    X_train_ts, y_train, source_index = ws.augment(X[train_index], y[train_index, l])
                else:
                    X_train_ts = X[train_index]
                    y_train = y[train_index, l]

                if use_raw_ts:
                    X_train = X_train_ts
                    X_test = X[test_index]
                else:
                    window_size = 4  # 40
                    X_train = calc_features(X_train_ts, window_size=window_size)
                    X_test = calc_features(X[test_index], window_size=window_size)
                y_test = y[test_index, l]

                # n_features = 3
                # event = -1
                # X_features = X_train.reshape(X_train.shape[0], n_features, -1)
                # plt.plot(X_train_ts[event])
                # feature_range = np.linspace(500, 7500, 8)
                # plt.plot(feature_range, X_features[event, 0], ".", label="min")
                # plt.plot(feature_range, X_features[event, 1], ".", label="median")
                # plt.plot(feature_range, X_features[event, 2], ".", label="max")
                # plt.legend()
                # plt.show()

                # Train classifier
                clf.fit(X_train, y_train)
                y_pred_train = clf.predict(X_train)

                # Evaluate classifier
                y_pred = clf.predict(X_test)
                y_prob = np.array(clf.predict_proba(X_test))
                result = classification_report(y_test, y_pred, target_names=target_names)

                if hasattr(clf, "feature_importances_"):
                    f_names = [[f"{i}_0.95", f"{i}_0.75", f"{i}_0.5", f"{i}_0.25", f"{i}_0.05", f"{i}_k"] for i in range(window_size)]
                    fimp = pd.Series(clf.feature_importances_, index=np.array(f_names).flatten())
                    fimp.sort_values().to_csv(fold_path / f"{name}_{label_name}_feature_importances.csv")

                # Add result to table
                df_result.loc[df_result.index[test_index], f"{label_name}_true"] = y_test
                df_result.loc[df_result.index[test_index], f"{label_name}_pred"] = y_pred
                df_result.loc[df_result.index[test_index], f"{label_name}_prob"] = y_prob[:, 1]

                # Plot results
                # train
                plot_confusion_matrix(y_train, y_pred_train, target_names, fold_path,
                                      n_split=f"{name}_{label_name}_train")
                plot_confusion_signals(X_train, y_train, y_pred_train, target_names, fold_path,
                                       n_split=f"{name}_{label_name}_train")

                # test
                plot_confusion_matrix(y_test, y_pred, target_names, fold_path, n_split=f"{name}_{label_name}_test")
                plot_confusion_signals(X_test, y_test, y_pred, target_names, fold_path,
                                       n_split=f"{name}_{label_name}_test")

                distance, k_index_augm = clf.kneighbors(X_test)
                k_ind = source_index[k_index_augm]
                for k in range(k_ind.shape[-1]):
                    df_kneighbors.loc[df_result.index[test_index], f"{k}_kneighbors"] = k_ind[:, k]
                    df_kneighbors.loc[df_result.index[test_index], f"{k}_distance"] = distance[:, k]



            balanced_accuracy = balanced_accuracy_score(df_result[f"{label_name}_true"].values,
                                                        df_result[f"{label_name}_pred"].values)
            df_accuracy.loc[name, label_name] = balanced_accuracy
            print(f"balanced_accuracy {name}_{label_name}: {balanced_accuracy}")

            plot_confusion_matrix(df_result[f"{label_name}_true"].values,
                                  df_result[f"{label_name}_pred"].values, target_names, clf_path,
                                  n_split=f"{label_name}_test")
            plot_confusion_signals(X, df_result[f"{label_name}_true"].values,
                                   df_result[f"{label_name}_pred"].values, target_names, clf_path,
                                   n_split=f"{label_name}_test")

            plt.close('all')
            gc.collect()

        df_true = df_result.filter(regex="true")
        df_pred = df_result.filter(regex="pred")
        df_prob = df_result.filter(regex="prob")
        df_results = pd.concat([df_true, df_prob], axis=1)
        for la in label_names:
            df_results[f"{la}_correct_classification"] = (df_result[f"{la}_true"] == df_result[f"{la}_pred"])+0
        write_excel(df_results, labels_path, df_kneighbors, output_path=output_path / f'{name}_results.xlsx')

    df_accuracy["mean"] = df_accuracy.mean(axis=1)
    overall_mean = df_accuracy["mean"].mean()
    df_accuracy.to_csv(output_path / f"balanced_accuracy_{overall_mean:.2f}.csv")
