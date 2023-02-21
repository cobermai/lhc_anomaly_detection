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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC

from src.utils.dataset_utils import u_diode_data_to_df
from src.utils.hdf_tools import load_from_hdf_with_regex
from src.visualisation.uqs0_classification_visualisation import plot_confusion_signals, plot_confusion_matrix


def generate_magnet_labels_from_event(df_event_labels: pd.DataFrame, label_names: list) -> pd.DataFrame:
    """
    generates labels for each magnet from event data
    :param df_event_labels: dataframe with labels for each event
    :param label_names: list of labels to generate
    :return: df_magnet_labels: dataframe with labels for each magent
    """
    df_event_labels["magnet"] = df_event_labels["event"].apply(lambda x: x.split("_")[0])
    magnet_list = df_event_labels["magnet"].unique()

    df_magnet_labels = pd.DataFrame(np.zeros((len(magnet_list), len(label_names))),
                                   columns=label_names,
                                   index=magnet_list)

    for m in magnet_list:
        df_subset = df_event_labels[df_event_labels["magnet"] == m]
        j = 0
        for l in label_names:
            if ((df_subset[l] == "x") | (df_subset[l] == "xx")).any():
                df_magnet_labels.loc[m, l] = 1
                j += 1
        if j == 0:
            df_magnet_labels.loc[m, "normal"] = 1
        else:
            df_magnet_labels.loc[m, "normal"] = 0

    return df_magnet_labels


def filter_df(df_filt, window_size, step):

    # create a series of group labels based on the step
    offset = abs(df_filt.index.min())  # rolling cannot handle zero passing
    df_filt.index = df_filt.index + offset
    group_labels = np.arange(len(df_filt)) // step
    # apply the rolling method to each group
    max_df = df_filt.groupby(group_labels)\
        .rolling(window_size, center=True).median().reset_index(level=0, drop=True).dropna(how="all")
    max_df.index = max_df.index - offset

    return max_df


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

        window_size = 50
        step = 50
        df = df.iloc[60:]
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


class WindowSlice():
    def __init__(self, labels=None, reduce_ratio=0.9):
        self.labels = labels
        self.reduce_ratio = reduce_ratio
        self.name = "window_slice"

    def augment(self, X: np.array, y: np.array, oversampling_rate=None) -> tuple:
        if oversampling_rate is None:
            oversampling_rate = np.sum(y, axis=0).max() / np.sum(y, axis=0)

        X_augmented_list = []
        y_augmented_list = []
        for i, label in enumerate(self.labels):
            label_bool = (y == label).all(axis=1)
            X_label = X[label_bool]
            y_label = y[label_bool]

            random_choices = np.random.choice(len(X_label), size=int(np.floor(oversampling_rate[i] * len(X_label))))
            X_oversampled = X_label[random_choices]
            y_oversampled = y_label[random_choices]

            # https://halshs.archives-ouvertes.fr/halshs-01357973/document
            X_window_sliced = np.zeros_like(X_oversampled)
            target_len = np.ceil(self.reduce_ratio * X_oversampled.shape[1]).astype(int)

            starts = np.random.randint(low=0, high=X.shape[1] - target_len, size=(X_oversampled.shape[0])).astype(int)
            ends = (target_len + starts).astype(int)

            for i, pattern in enumerate(X_oversampled):
                X_window_sliced[i] = np.interp(np.linspace(0, target_len, num=X_oversampled.shape[1]),
                                               np.arange(target_len),
                                               pattern[starts[i]:ends[i]]).T

            X_augmented_list.append(X_window_sliced)
            y_augmented_list.append(y_oversampled)

        X_augmented = np.vstack(X_augmented_list)
        y_augmented = np.vstack(y_augmented_list)

        idx_shuffled = np.random.permutation(np.arange(len(X_augmented)))

        return X_augmented[idx_shuffled], y_augmented[idx_shuffled]

def write_excel():
    # Open the source Excel file
    source_workbook = openpyxl.load_workbook(labels_path)
    source_sheet = source_workbook.active

    column_names = ["A", "B", "C", "D"]
    with open(output_path / 'dest_file.xlsx', 'wb') as dest_file:
        dest_workbook = xlsxwriter.Workbook(dest_file)
        dest_sheet = dest_workbook.add_worksheet()

        columns = ['event', 'Electrical order', true_label_name, pred_label_name, "pred probability"]
        dest_sheet.write_row(0, 0, columns)

        red_format = dest_workbook.add_format({'bg_color': '#FFC7CE'})
        j = 1
        for i, row in df_experiment.iterrows():
            # Read the URL from the source file
            url = source_sheet.cell(row=1 + i, column=2).hyperlink.target

            # Create a new Excel file and copy the URL to it
            dest_sheet.write(j, 0, row["event"])
            dest_sheet.write_url(j, 1, url, string=str(row['Electrical order']))

            df_result_magnet = df_result[df_result.index == row["magnet"]]
            man_label = df_result_magnet[true_label_name].values[0]
            pred_label = df_result_magnet[pred_label_name].values[0]
            if not man_label == pred_label:
                dest_sheet.write(j, 2, man_label, red_format)
                dest_sheet.write(j, 3, pred_label, red_format)
            else:
                dest_sheet.write(j, 2, man_label)
                dest_sheet.write(j, 3, pred_label)

            dest_sheet.write(j, 4, df_result_magnet["ML probability"].values[0])
            j += 1

        for column_index in range(len(columns)):
            column_name = xl_utility.xl_col_to_name(column_index)  # Convert index to column name
            dest_sheet.set_column(f'{column_name}:{column_name}', 20)

        dest_workbook.close()


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
    label_names = [#"AUTO_unbalanced",
                   "MANUAL_long_unbalanced",
                   "MANUAL_very_long_unbalanced",
                   "MANUAL_boardA_boardB"]
    labels_path = Path("../data/UQS0_labels/Outliers_summary_Meas_2021_2022_NoZero_backup.xlsx")
    df_event_labels = pd.read_excel(labels_path)
    df_magnet_labels = generate_magnet_labels_from_event(df_event_labels, label_names)
    df_experiment = df_event_labels[(df_event_labels["Current_level"] == current_level) &
                                    (df_event_labels["Expected_dI_dt"] == dI_dt)]

    # Load data
    data_path = Path("D:/datasets/20230220_snapshot_data")  # Path('/eos/project/s/steam/measurement_database/RB/data')
    df_data = load_uqs0_data_from_hdf5(df_context_subset, data_path)

    # Process data
    clip = 0.15
    df_data[(df_data > clip)] = clip
    df_data[df_data < -clip] = -clip
    X = np.nan_to_num((df_data.T.values - np.nanmean(df_data.T.values)) / np.nanstd(df_data.T.values))
    all_labels = ["normal"] + label_names
    y = df_magnet_labels.loc[df_data.columns, all_labels].values
    y_argmax = np.argmax(y, axis=1)

    # Select classification parameter
    augment_data = True
    train_test_split = False
    classifier = KNeighborsClassifier

    # Leave one out cross validation
    if train_test_split:
        fold_path = output_path / "folds"
        fold_path.mkdir(parents=True, exist_ok=True)

        skf = StratifiedKFold(n_splits=8, random_state=0, shuffle=True)
        df_result = pd.DataFrame([all_labels[i] for i in y_argmax],
                                 index=df_magnet_labels.index,
                                 columns=[true_label_name])
        for i, (train_index, test_index) in enumerate(skf.split(X, y_argmax)):
            print(f"Fold {i}:")

            # data augmentation
            if augment_data:
                ws = WindowSlice(labels=np.eye(len(all_labels)))
                X_train, y_train = ws.augment(X[train_index], y[train_index])
            else:
                X_train = X[train_index]
                y_train = y[train_index]

            # Train classifier
            dt_clf = classifier() # , class_weight="balanced", max_depth=1
            dt_clf.fit(X_train, y_train)

            # Evaluate classifier
            # test
            y_pred = dt_clf.predict(X[test_index])
            y_test_argmax = np.argmax(y[test_index], axis=1)
            y_pred_argmax = np.argmax(y_pred, axis=1)
            result = classification_report(y_test_argmax, y_pred_argmax, target_names=all_labels)
            print(result)
            # augmented
            y_pred_train = dt_clf.predict(X_train)
            y_train_argmax = np.argmax(y_train, axis=1)
            y_pred_train_argmax = np.argmax(y_pred_train, axis=1)


            # Add result to table
            # probabilities
            y_prob = np.array(dt_clf.predict_proba(X[test_index]))
            y_prob_flat = np.array([y_prob[v, i, 1] for i, v in enumerate(y_pred_argmax)])
            df_result.loc[df_result.index[test_index], pred_label_name] = [all_labels[i] for i in y_pred_argmax]
            df_result.loc[df_result.index[test_index], "ML probability"] = y_prob_flat
            df_result.loc[df_result.index[test_index], "pred_int"] = y_pred_argmax
            df_result.loc[df_result.index[test_index], "true_int"] = y_test_argmax

            # Plot results
            # train
            plot_confusion_matrix(y_train_argmax, y_pred_train_argmax, all_labels, fold_path, n_split=f"{i}_train")
            plot_confusion_signals(X_train, y_train_argmax, y_pred_train_argmax, all_labels, fold_path, n_split=f"{i}_train")

            # test
            plot_confusion_matrix(y_test_argmax, y_pred_argmax, all_labels, fold_path, n_split=f"{i}_test")
            plot_confusion_signals(X[test_index], y_test_argmax, y_pred_argmax, all_labels, fold_path, n_split=f"{i}_test")

        plot_confusion_matrix(df_result["true_int"].values, df_result["pred_int"].values, all_labels, output_path, n_split="test_all")
        plot_confusion_signals(X, df_result["true_int"].values, df_result["pred_int"].values, all_labels, output_path, n_split="test_all")

    #write_excel()

    # train on whole dataset
    ws = WindowSlice(labels=np.eye(len(all_labels)))
    X_train, y_train = ws.augment(X, y)

    # Train classifier
    dt_clf = classifier(n_neighbors=10)
    dt_clf.fit(X_train, y_train)

    # Evaluate classifier
    y_pred = dt_clf.predict(X)
    y_test_argmax = np.argmax(y, axis=1)
    y_pred_argmax = np.argmax(y_pred, axis=1)

    # test
    plot_confusion_matrix(y_test_argmax, y_pred_argmax, all_labels, output_path, n_split="train_all")
    plot_confusion_signals(X, y_test_argmax, y_pred_argmax, all_labels, output_path, n_split="train_all")
