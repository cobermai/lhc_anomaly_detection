from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == "__main__":
    data_dir = Path("../../data")

    RB_LayoutDetails_path = data_dir / "SIGMON_context_data/RB_LayoutDetails.csv"
    df_RB_LayoutDetails = pd.read_csv(RB_LayoutDetails_path)

    # add busbar protection details
    # https://gitlab.cern.ch/LHCData/lhc-sm-api/-/blob/dev/info/metadata_and_references.md
    nQPS_RB_busBarProtectionDetails_path = data_dir / "SIGMON_context_data/nQPS_RB_busBarProtectionDetails.csv"
    df_nQPS_RB_busBarProtectionDetails = pd.read_csv(nQPS_RB_busBarProtectionDetails_path)
    df_metadata = df_RB_LayoutDetails.merge(df_nQPS_RB_busBarProtectionDetails,
                                            left_on=["Magnet", "Circuit"],
                                            right_on=["2nd Magnet", "Circuit"],
                                            how="left")
    # add physical position
    for circuit in df_metadata['Circuit'].unique():
        df_metadata.loc[df_metadata['Circuit'] == circuit, 'phys_pos'] = np.arange(1, 155, dtype=int)
    df_metadata['phys_pos'] = df_metadata['phys_pos'].astype(int)

    # add beamscreen resistance
    RB_Beamscreen_Resistances_path = data_dir / "STEAM_context_data/RB_Beamscreen_Resistances.csv"
    df_RB_Beamscreen_Resistances = pd.read_csv(RB_Beamscreen_Resistances_path)
    df_RB_Beamscreen_Resistances["Magnet"] = df_RB_Beamscreen_Resistances.Name.apply(lambda x: "MB." + x)
    df_metadata = df_metadata.merge(df_RB_Beamscreen_Resistances,
                                    left_on=["Magnet"],
                                    right_on=["Magnet"],
                                    how="left")

    # NOT MERGED YET:
    # magnet added to BeamScreen_EAMdata.xlsx by marvin, one entry per aperture, sometimes 3 entries/magnet?
    BeamScreen_EAMdata_path = data_dir / "STEAM_context_data/BeamScreen_EAMdata_Magnets.csv"
    df_BeamScreen_EAMdata = pd.read_csv(BeamScreen_EAMdata_path)
    df_BeamScreen_EAMdata["Magnet"] = df_BeamScreen_EAMdata.Name.apply(lambda x: "MB." + x)
    # drop beamscreens not in use, not all magnets can be mapped
    df_BeamScreen_EAMdata = df_BeamScreen_EAMdata[df_BeamScreen_EAMdata.Magnet.isin(df_RB_LayoutDetails.Magnet)]

    drop_columns = df_metadata.filter(regex='Unnamed').columns
    df_metadata.drop(columns=drop_columns).to_csv(data_dir / "RB_metadata.csv", index=False)
