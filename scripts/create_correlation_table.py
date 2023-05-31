import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # Define Paths
    file_path = Path('/eos/project/m/ml-for-alarm-system/private/RB_signals')
    data_path = file_path / 'backup/20220707_data'
    simulation_path = file_path / 'backup/20220707_simulation'

    # Read Snapshot metadata
    snapshot_context_path = Path("../data/RB_snapshot_context.csv")
    snapshot_context_df = pd.read_csv(snapshot_context_path)
    #snapshot_context_df[['El. Position', 'Phys. Position']] = 0

    # Read the (clean) MP3 file
    mp3_fpa_df_path = Path("../data/RB_TC_extract_2022_07_07_processed_filled.csv")
    mp3_fpa_df = pd.read_csv(mp3_fpa_df_path)
    mp3_fpa_df = pd.concat([snapshot_context_df, mp3_fpa_df[mp3_fpa_df['timestamp_fgc'] >= 1526582397220000000]])
    mp3_fpa_df_unique = mp3_fpa_df.drop_duplicates(subset=['fpa_identifier'])

    # Read metadata
    metadata_path_path = "../data/RB_metadata.csv"
    metadata_path = Path(metadata_path_path)
    rb_magnet_metadata = pd.read_csv(metadata_path, index_col=False)
    # python index starts at 0:
    rb_magnet_metadata[['El. Position', 'Phys. Position']] = rb_magnet_metadata[['El. Position', 'Phys. Position']] - 1

    # Read masterfile MB_feature_analysis
    master_file_path = Path("../data/analysis_conductor_fieldQuality_FPA2021_FTM2023_year2021_FPA2021_TFM2023.csv")
    df_master_file = pd.read_csv(master_file_path, index_col=False)
    empty_columns = df_master_file.columns[df_master_file.isnull().all()]
    df_master_file = df_master_file.drop(columns=empty_columns)
    df_master_file['Magnet'] = df_master_file.positions.apply(lambda x: 'MB.' + x)

    # Merge files
    df_position_context = rb_magnet_metadata.merge(df_master_file,
                                               left_on=["Magnet"],
                                               right_on=["Magnet"],
                                               how="left", suffixes=('', '_y'))
    mp3_fpa_df_unique = mp3_fpa_df_unique.merge(df_position_context,
                                                left_on=["Magnet"],
                                                right_on=["Magnet"],
                                                how="left", suffixes=('', '_y'))
    drop_columns = mp3_fpa_df_unique.filter(regex='Unnamed').columns.to_list()
    drop_columns += mp3_fpa_df_unique.filter(regex='_y').columns.to_list()
    mp3_fpa_df_unique = mp3_fpa_df_unique.drop(columns=drop_columns)

    # Load description of context variables
    feature_context_path = Path("../data/metadata_context_variables.xlsx")
    df_feature_context = pd.read_excel(feature_context_path, engine='openpyxl', index_col="Unnamed: 0")
    df_feature_context['nunique'] = mp3_fpa_df_unique[df_feature_context.index.values].nunique()
    df_feature_context['nmissing'] = mp3_fpa_df_unique[df_feature_context.index.values].isna().sum()
    # df_feature_context.to_excel('../data/metadata_context_variables.xlsx')

    # Load frequency data
    df_comp = pd.read_csv("../data/final_components/cweights_trend.csv")
    fpa_identifiers_fitted = df_comp.fpa_identifier
    fpa_identifiers = mp3_fpa_df_unique.loc[mp3_fpa_df_unique.fpa_identifier.isin(fpa_identifiers_fitted),
                                            "fpa_identifier"].values

    # Load sorting data
    sort_map_path = Path("../data/circuit_sort.csv")
    df_sort_map = pd.read_csv(sort_map_path)
    df_sort_map = df_sort_map.set_index(['Circuit', 'El. Quench Position', 'Phys. Quench Position'])
    sort_columns = ['Phys. Dist. to PC', 'Phys. Position ODD', 'Phys. Position EVEN', 'Phys. Dist. to Quench',
                    'El. Dist. to Quench Main', 'El. Dist. to Quench Mirror']

    ## Calculate correlation matix event context
    # define event and position data
    df_event = mp3_fpa_df_unique.loc[mp3_fpa_df_unique.fpa_identifier.isin(fpa_identifiers_fitted), df_feature_context[
        (df_feature_context.ftype == 'event')].index.values]
    df_position = df_position_context[df_feature_context[
        (df_feature_context.ftype == 'position') | (df_feature_context.ftype == 'conductor')].index.values]

    # process event data
    one_hot_columns = df_feature_context[
        (df_feature_context.ftype == 'event') & (df_feature_context.one_hot == True)].index.values
    df_event_processed = pd.get_dummies(df_event, columns=one_hot_columns)

    # process position data
    bool_pos_features = ((df_feature_context.ftype == 'position') | (df_feature_context.ftype == 'conductor')) & (
            df_feature_context.one_hot == True)
    one_hot_columns = df_feature_context[bool_pos_features].index.values
    df_position_processed = pd.get_dummies(df_position, columns=one_hot_columns)

    # Scale data
    scaling = False
    if scaling == True:
        scale_columns = df_feature_context[
            (df_feature_context.ftype == 'event') & (df_feature_context.one_hot == False)].index.values
        df_event_processed[scale_columns] = \
            ((df_event[scale_columns] - df_event[scale_columns].mean()) / df_event[scale_columns].std())[scale_columns]
        scale_columns = df_feature_context[
            (df_feature_context.ftype == 'position') & (df_feature_context.one_hot == False)].index.values
        df_position_processed[scale_columns] = \
            ((df_position[scale_columns] - df_position[scale_columns].mean()) / df_position[scale_columns].std())[
                scale_columns]

    # create correlation table, size is (n_events, n_magnets, n_features) -> (560, 154, 121)
    n_events = len(fpa_identifiers)
    n_magnets = 154
    data_list = []
    for i, f in enumerate(fpa_identifiers):
        circuit = f.split('_')[1]
        timestamp_fgc = int(f.split('_')[2])
        quenched_magnet = mp3_fpa_df_unique[mp3_fpa_df_unique.fpa_identifier == f].Magnet.values[0]

        rb_metadata_subset = rb_magnet_metadata[(rb_magnet_metadata.Circuit == circuit)].sort_values(by="El. Position")
        df_pos_data = df_position_processed[(rb_magnet_metadata.Circuit == circuit)].sort_values(by="El. Position")
        df_event_data = df_event_processed[df_event.timestamp_fgc == timestamp_fgc]
        df_pos_data["fpa_identifier"] = f

        # do not add for snapshots
        if isinstance(quenched_magnet, str):
            q_pos = rb_magnet_metadata[rb_magnet_metadata.Magnet == quenched_magnet]["El. Position"].values[0]


            df_q_pos_data = df_position_processed[(rb_magnet_metadata.Circuit == circuit) &
                                                  (rb_magnet_metadata.Magnet == quenched_magnet)].add_suffix('_q')

            df_pos_data[sort_columns] = df_sort_map.loc[circuit, q_pos].sort_values(by="El. Position")[sort_columns].values

            df_pos_data['El. Dist. to Quench'] = np.abs(df_pos_data['El. Position'].values
                                                        - df_q_pos_data['El. Position_q'].values)

            q_cryo = rb_magnet_metadata[rb_magnet_metadata.Magnet == quenched_magnet]["cryostat_group"].values[0]
            q_qps = rb_magnet_metadata[rb_magnet_metadata.Magnet == quenched_magnet]['QPS Crate'].values[0]
            df_pos_data[["is_quench_cryostat", "is_quench_qps"]] = 0
            df_pos_data.loc[rb_metadata_subset["cryostat_group"] == q_cryo, "is_quench_cryostat"] = 1
            df_pos_data.loc[rb_metadata_subset['QPS Crate'] == q_qps, "is_quench_qps"] = 1

            df_pos_data[df_q_pos_data.columns] = np.repeat(df_q_pos_data.values, repeats=n_magnets, axis=0)
        else:
            df_pos_data[sort_columns + ['El. Dist. to Quench', "is_quench_cryostat","is_quench_qps"]] = None
            df_pos_data[df_position_processed.columns+'_q'] = None

        df_pos_data[df_event_data.columns] = np.repeat(df_event_data.values, repeats=n_magnets, axis=0)

        all_columns = df_pos_data.columns.to_list() #df_event_data.columns.to_list() +
        data_list.append(df_pos_data[all_columns])
    df_data = pd.concat(data_list, axis=0, ignore_index=True).reset_index(drop=True)

    # make bools of different plateaus and experiment types
    test_conditions = ((mp3_fpa_df['Delta_t(iQPS-PIC)'] / 1000 < 5) &
                       (mp3_fpa_df['Nr in Q event'].astype(str) != '1'))
    fpa_identifiers_double = np.hstack((fpa_identifiers, fpa_identifiers))
    bool_test = np.isin(fpa_identifiers_double, mp3_fpa_df[test_conditions].fpa_identifier.unique())
    bool_1EE = np.hstack([np.ones(len(fpa_identifiers)), np.zeros(len(fpa_identifiers))])
    bool_test_flat = np.stack([bool_test for l in range(n_magnets)]).T.reshape(-1)
    bool_1EE_flat = np.stack([bool_1EE for l in range(n_magnets)]).T.reshape(-1)
    df_data_all = pd.concat([df_data, df_data], axis=0).reset_index(drop=True)
    df_data_all["no_sec_q"] = ~bool_test_flat + 0
    df_data_all["sec_q"] = bool_test_flat + 0
    df_data_all["1EE"] = bool_1EE_flat + 0
    df_data_all["2EE"] = bool_1EE_flat[::-1] + 0

    # add components to table
    #df_comp_cut = df_comp.loc[fpa_identifiers]
    #df_data_all = df_data_all.set_index(df_comp_cut.index)
    #df = pd.concat([df_data_all, df_comp_cut], axis=1)

    df = df_data_all.merge(df_comp,
                           left_on=["fpa_identifier", 'El. Position', "1EE"],
                           right_on=["fpa_identifier", 'El. Position', "1EE"])

    df.to_csv('../data/correlation_tables/correlation_table.csv')
    print(f"Correlation Table Shape: {df.shape}")

    # calculate and save correlation
    #method = "pearson"
    #df_corr = df.corr(method=method)
    #df_corr.to_excel(f'../data/correlation_tables/correlation_{method}.xlsx')
