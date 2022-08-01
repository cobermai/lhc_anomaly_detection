import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from src.modeling.sec_quench import get_std_of_diff, get_sec_quench_frame_exclude_quench, sort_by_metadata
from src.utils.hdf_tools import load_from_hdf_with_regex, u_diode_data_to_df


def plot_wiggle_analysis(
    mp3_fpa_df: pd.DataFrame,
    df_results: pd.DataFrame,
    data_path: Path,
    show_n_quenches: int = 5,
    features: list = [""],
    sort_feature: str = "",
    time_frame_after_quench: list = [0,2]):
    """
    plots data signals given wiggle analysis results
    :param mp3_fpa_df: dataframe with mp3 excel data
    :param df_results: dataframe with wiggle analysis results
    :param data_path: path where u diode hdf5 data is stored
    :param show_n_quenches: amount of events to plot
    :param features: features to show
    :param sort_feature: features to sort, show_n_quenches of the highest feature are plotted
    :param time_frame_after_quench: time period to plot
    :return:
    """
    for i, row in df_results[:show_n_quenches].iterrows():
        circuit_name = row["Circuit Name"]
        timestamp_fgc = row["timestamp_fgc"]
        fpa_identifier = f"{row['Circuit Family']}_{row['Circuit Name']}_{int(row['timestamp_fgc'])}"

        data_dir = data_path / (fpa_identifier + ".hdf5")
        data = load_from_hdf_with_regex(
            file_path=data_dir,
            regex_list=["VoltageNXCALS.*U_DIODE"])
        df_data_nxcals = u_diode_data_to_df(data)

        df_subset = mp3_fpa_df[(mp3_fpa_df.timestamp_fgc == timestamp_fgc) & (
            mp3_fpa_df["Circuit Name"] == circuit_name)]

        quench_times = df_subset["Delta_t(iQPS-PIC)"].values / 1e3

        sec_quenches = get_sec_quench_frame_exclude_quench(
            df_data=df_data_nxcals,
            all_quenched_magnets=df_subset.Position.values,
            quench_times=quench_times,
            time_frame=time_frame_after_quench)

        sec_quench_number = int(row["sec_quench_number"])

        df_std = get_std_of_diff(df=sec_quenches[sec_quench_number])

        df_std_meta = sort_by_metadata(
            df=df_std,
            quenched_magnet=df_subset.Position.values[sec_quench_number + 1],
            circuit=row["Circuit Name"],
            by="Position")

        df_std_meta_elpos = sort_by_metadata(
            df=df_std,
            quenched_magnet=df_subset.Position.values[sec_quench_number + 1],
            circuit=circuit_name,
            by="#Electric_EE")

        # look at secondary quenches within currently analyses secondary quench
        quenches_within_frame = [q for q in quench_times if
                                 (q > quench_times[sec_quench_number + 1] - time_frame_after_quench[0]) &
                                 (q < quench_times[sec_quench_number + 1] + time_frame_after_quench[1])]

        fig, ax = plt.subplots(1, 5, figsize=(25, 4))
        df_data_nxcals.plot(legend=False, ax=ax[0])
        ax[0].set_title(
            f"{fpa_identifier} \n {df_subset['Date (FGC)'].values[0]} {df_subset['Time (FGC)'].values[0]}")
        ax[0].set_xlabel("Time \\ s")
        ax[0].set_ylabel("Voltage \\ V")
        ax[0].axvline(quench_times[sec_quench_number + 1],
                      color='b', linestyle='--')

        sec_quenches[sec_quench_number].plot(legend=False, ax=ax[1])
        # no primary, start from 0
        ax[1].set_title(f"U_diode, zoom in to quench nr.: {sec_quench_number + 2} ({row['Position']})")
        ax[1].set_xlabel("Time \\ s")
        ax[1].set_ylabel("Voltage \\ V")
        for quench_within_frame in quenches_within_frame:
            ax[1].axvline(
                quench_within_frame,
                color='b',
                linestyle='--',
                label="time of next quench")

        sec_quenches[sec_quench_number].diff().plot(legend=False, ax=ax[2])
        ax[2].set_title("dU_diode")
        ax[2].set_xlabel("Time \\ s")
        ax[2].set_ylabel("Voltage \\ s")
        ax[2].set_ylim((df_results[:show_n_quenches].dU_min.min(
        ) * 1.1, df_results[:show_n_quenches].dU_max.max() * 1.1))
        for quench_within_frame in quenches_within_frame:
            ax[2].axvline(
                quench_within_frame,
                color='b',
                linestyle='--',
                label="time of next quench")

        df_std_meta.fillna(0).plot(
            x="distance_to_quench",
            y="dstd",
            ax=ax[3],
            label="dstd phys. position")
        df_std_meta_elpos.fillna(0).plot(
            x="distance_to_quench",
            y="dstd",
            ax=ax[3],
            label="dstd el. position",
            zorder=-1)
        ax[3].set_title(f"std of derivative=f(distance to quench)")
        ax[3].set_ylim((0, df_results[:show_n_quenches].dstd_max.max() * 1.1))

        df_interpol = df_std_meta_elpos.interpolate(method='linear')
        df_median = df_interpol.rolling(15).mean()
        df_median.plot(x="distance_to_quench", y="dstd", ax=ax[3])

        ax[4].axis('off')
        for j, feature in enumerate(features):
            rank = int(df_results[:show_n_quenches]
                       [feature].rank(ascending=False)[i])
            if feature == sort_feature:
                ax[4].text(0, 1 - j / len(features),
                    f"{feature}: {row[feature]:.2f} (#{rank})",
                    weight="bold")
            else:
                ax[4].text(0, 1 - j / len(features),
                    f"{feature}: {row[feature]:.2f} (#{rank})")

        plt.tight_layout()
        fig.show()


if __name__ == "__main__":
    # define paths
    data_path = Path(
        '/eos/project/m/ml-for-alarm-system/private/RB_signals/20220707_data')
    if not os.path.isdir(data_path):
        data_path = Path('/mnt/d/datasets/20220707_data')
    results_path = Path("../../data/sec_quench_feature.csv")
    mp3_excel_path = Path(
        "../../data/RB_TC_extract_2022_07_07_processed_filled.csv")

    # load mp3 file
    mp3_fpa_df = pd.read_csv(mp3_excel_path)

    # load analysis results
    df_results = pd.read_csv(results_path)

    # define features to show
    features = [
        'min_time',
        'max_time',
        'min_amplitude',
        'max_amplitude',
        "dU_min",
        "dU_max",
        "dstd_max",
        'dstd_score_pos_15',
        'dstd_score_elpos_15',
        'dstd_score_pos_15_exp',
        'dstd_score_elpos_15_exp',
        'dstd_score_pos_15_tuk',
        'dstd_score_elpos_15_tuk',
        'wiggle_area_pos',
        'wiggle_area_elpos',
        "el_peak12_ratio",
        "n_other_quenches",
        "wiggle_magnets"]

    # plot wiggle analysis
    plot_wiggle_analysis(
        mp3_fpa_df=mp3_fpa_df,
        df_results=df_results,
        data_path=data_path,
        show_n_quenches=5,
        features=features,
        sort_feature="dstd_score_pos_15")
