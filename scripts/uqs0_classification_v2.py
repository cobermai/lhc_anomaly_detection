import os
import typing
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
import xlsxwriter
import xlsxwriter.utility as xl_utility
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC

from src.utils.dataset_utils import u_diode_data_to_df
from src.utils.hdf_tools import load_from_hdf_with_regex
from src.visualisation.uqs0_classification_visualisation import plot_confusion_signals, plot_confusion_matrix, \
    write_excel


def filter_df(df_filt, window_size, step):

    # create a series of group labels based on the step
    offset = abs(df_filt.index.min())  # rolling cannot handle zero passing
    df_filt.index = df_filt.index + offset
    group_labels = np.arange(len(df_filt)) // step

    # apply the rolling method to each group
    med_df = df_filt.groupby(group_labels)\
        .rolling(window_size, center=True).mean().reset_index(level=0, drop=True).dropna(how="all")

    return med_df


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

        window_size = 100
        step = 100
        df = df.iloc[500:]
        df = filter_df(df, window_size, step)
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
    label_names = ["true_2_normal",
                   "true_2_long_unbalanced",
                   "true_2_very_long_unbalanced",
                   "true_2_boardA_boardB"]

    labels_path = Path("../data/UQS0_labels/dest_file_v02.xlsx")
    df_event_labels = pd.read_excel(labels_path)
    df_event_labels["magnet"] = df_event_labels["event"].apply(lambda x: x.split("_")[0])
    df_magnet_labels = df_event_labels.set_index('magnet')[label_names] #generate_magnet_labels_from_event(df_event_labels, label_names)

    experiment_path = Path("../data/UQS0_labels/Outliers_summary_Meas_2021_2022_NoZero.xlsx")
    df_all_experiments = pd.read_excel(experiment_path)
    df_experiment = df_all_experiments[(df_all_experiments["Current_level"] == current_level) &
                                        (df_all_experiments["Expected_dI_dt"] == dI_dt)]

    # Load data
    data_path = Path("D:/datasets/20230220_snapshot_data")  # Path('/eos/project/s/steam/measurement_database/RB/data')
    df_data = load_uqs0_data_from_hdf5(df_context_subset, data_path)[df_magnet_labels.index.values]

    # Process data
    clip = 0.15
    n_signals = 2  # Board A & B
    df_data[(df_data > clip)] = clip
    df_data[df_data < -clip] = -clip
    data = df_data.T.values
    X = np.nan_to_num(data)
    target_names = label_names
    y = df_magnet_labels.loc[df_magnet_labels.index.values, target_names].values
    y_argmax = np.argmax(y, axis=1)

    # define classifiers
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    # Leave one out cross validation
    skf = StratifiedKFold(n_splits=8, random_state=0, shuffle=True)
    df_result = pd.DataFrame([target_names[i] for i in y_argmax],
                             index=df_magnet_labels.index,
                             columns=[true_label_name])

    # Iterate over classifiers
    for name, clf in zip(names, classifiers):
        print(f"{name}:")
        clf_path = output_path / name

        # Iterate over labels - one vs. rest classification
        for l, label_name in enumerate(label_names):
            target_names = [f"not_{label_name}" ,label_name]

            # Iterate over data splits
            for i, (train_index, test_index) in enumerate(skf.split(X, y_argmax)):
                print(f"Fold {i}:")
                fold_path = clf_path / Path(f"fold_{i}")
                fold_path.mkdir(parents=True, exist_ok=True)


                # Define data in this iteration
                X_train = X[train_index]
                y_train = y[train_index, l]
                y_test = y[test_index, l]

                # Train classifier
                clf.fit(X_train, y_train)
                y_pred_train = clf.predict(X_train)

                # Evaluate classifier
                y_pred = clf.predict(X[test_index])
                result = classification_report(y_test, y_pred, target_names=target_names)
                print(result)

                # Add result to table
                df_result.loc[df_result.index[test_index], f"{label_name}_true"] = y_test
                df_result.loc[df_result.index[test_index], f"{label_name}_pred"] = y_pred

                # Plot results
                # train
                plot_confusion_matrix(y_train, y_pred_train, target_names, fold_path, n_split=f"{label_name}_train")
                plot_confusion_signals(X_train, y_train, y_pred_train, target_names, fold_path, n_split=f"{label_name}_train")

                # test
                plot_confusion_matrix(y_test, y_pred, target_names, fold_path, n_split=f"{label_name}_test")
                plot_confusion_signals(X[test_index], y_test, y_pred, target_names, fold_path, n_split=f"{label_name}_test")

        
            plot_confusion_matrix(df_result[f"{label_name}_true"].values, df_result[f"{label_name}_pred"].values, target_names, clf_path, n_split="test_all")
            plot_confusion_signals(X, df_result[f"{label_name}_true"].values, df_result[f"{label_name}_pred"].values, target_names, clf_path, n_split="test_all")


