from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == "__main__":
    year = 2021

    data_dir = Path("../../raw")
    RB_LayoutDetails_path = data_dir / "SIGMON_context/RB_LayoutDetails.csv"
    df_RB_LayoutDetails = pd.read_csv(RB_LayoutDetails_path)

    # add busbar protection details
    # https://gitlab.cern.ch/LHCData/lhc-sm-api/-/blob/dev/info/metadata_and_references.md
    nQPS_RB_busBarProtectionDetails_path = data_dir / "SIGMON_context/nQPS_RB_busBarProtectionDetails.csv"
    df_nQPS_RB_busBarProtectionDetails = pd.read_csv(nQPS_RB_busBarProtectionDetails_path)
    df_metadata = df_RB_LayoutDetails.merge(df_nQPS_RB_busBarProtectionDetails,
                                            left_on=["Magnet", "Circuit"],
                                            right_on=["2nd Magnet", "Circuit"],
                                            how="left", suffixes=('', '_y'))
    # add cryostat group
    df_metadata['cryostat_group'] = df_metadata['Cryostat2'].apply(lambda x: x.split('_')[1])
    # add physical position
    for circuit in df_metadata['Circuit'].unique():
        df_metadata.loc[df_metadata['Circuit'] == circuit, "phys_pos"] = np.arange(1, 155, dtype=int)
    df_metadata["Phys. Position"] = df_metadata["phys_pos"].astype(int)
    df_metadata["El. Position"] = df_metadata["#Electric_circuit"].astype(int)

    # add beamscreen resistance
    RB_Beamscreen_Resistances_path = data_dir / "MB_feature_analysis/RB_Beamscreen_Resistances.csv"
    df_RB_Beamscreen_Resistances = pd.read_csv(RB_Beamscreen_Resistances_path)
    df_RB_Beamscreen_Resistances["Magnet"] = df_RB_Beamscreen_Resistances.Name.apply(lambda x: "MB." + x)
    df_metadata = df_metadata.merge(df_RB_Beamscreen_Resistances,
                                    left_on=["Magnet"],
                                    right_on=["Magnet"],
                                    how="left", suffixes=('', '_y'))

    # add magnet metadata
    RB_Magnet_metadata_path = data_dir / "MP3_context/RB_TC_extract_2023_03_13.xlsx"
    df_RB_metadata_layout = pd.read_excel(RB_Magnet_metadata_path, sheet_name="layout")
    df_RB_metadata_magnet = pd.read_excel(RB_Magnet_metadata_path, sheet_name="magnet data")
    int_columns = [c for c in df_RB_metadata_layout.columns.values if type(c) is int]

    # calculate magnet age
    df_layout_diff = df_RB_metadata_layout.iloc[1:, :][int_columns].astype(int).diff(axis=1)
    df_layout_diff = df_layout_diff.apply(lambda row: np.max(row.index.values[abs(row) > 0], initial=2007), axis=1)
    df_RB_metadata_layout["age"] = year - df_layout_diff

    # merge
    df_metadata = df_metadata.merge(df_RB_metadata_layout[["Position", year, "age"]],
                                    left_on=["Name"],
                                    right_on=["Position"],
                                    how="left", suffixes=('', '_y'))
    df_metadata.rename(columns={year: "Short magnet ID"}, inplace=True)
    df_metadata = df_metadata.merge(df_RB_metadata_magnet,
                                    left_on=["Short magnet ID"],
                                    right_on=["Short magnet ID"],
                                    how="left", suffixes=('', '_y'))

    df_metadata["Magnet_construction_order"] = df_metadata['Short magnet ID']\
        .apply(lambda x: int(x-np.round(x/1000)*1000))

    # add features
    def enumerate_groups(x):
        x['num_group'] = np.arange(len(x))
        return x
    column = "QPS Crate"
    g = df_metadata.sort_values(by="#Electric_circuit").groupby(column, as_index=False)
    df_metadata["QPS Crate Number"] = g.apply(enumerate_groups).sort_values(by=["Circuit", "phys_pos"])['num_group']

    column = "cryostat_group"
    df_metadata[column] = df_metadata['Cryostat2'].apply(lambda x: x.split('_')[1])
    g = df_metadata.sort_values(by="phys_pos").groupby(column, as_index=False)
    df_metadata["Cryostat Number"] = g.apply(enumerate_groups).sort_values(by=["Circuit", "phys_pos"])['num_group']

    drop_columns = df_metadata.filter(regex='Unnamed').columns.to_list()
    drop_columns += df_metadata.filter(regex='_y').columns.to_list()
    df_metadata.drop(columns=drop_columns).to_csv("RB_position_context.csv", index=False)
