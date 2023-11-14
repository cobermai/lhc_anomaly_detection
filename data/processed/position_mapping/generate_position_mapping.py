import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.sort_utils import map_position_index, center_array, split_main_mirror
from src.utils.utils import pd_dict_filt, nanargsort

warnings.filterwarnings('ignore')


def generate_position_mapping(input_dir: Path, output_dir: Path, file_name: str, print_output: bool=False):

    rb_magnet_metadata = pd.read_csv(input_dir / (file_name + ".csv"), index_col=False)

    # Define position map
    pos_map = rb_magnet_metadata[["Circuit", "Magnet", "El. Position", "Phys. Position"]]

    # python index starts with 0
    pos_map[["El. Position", "Phys. Position"]] = pos_map[["El. Position", "Phys. Position"]] - 1

    # for each circuit get index to sort by
    phys_pos_index_odd = pos_map[pos_map.Circuit == 'RB.A12']["Phys. Position"].values
    phys_pos_index_even_in_odd = pos_map[pos_map.Circuit == 'RB.A12']["Phys. Position"].values[::-1]
    phys_pos_index_even = np.array(
        [[d, c] for c, d in zip(phys_pos_index_even_in_odd[::2], phys_pos_index_even_in_odd[1::2])]).flatten()
    for circuit in pos_map["Circuit"].unique():
        if int(circuit[4]) % 2 == 0:
            pc_dist_index = phys_pos_index_even
        else:
            pc_dist_index = phys_pos_index_even_in_odd
        pos_map.loc[pos_map["Circuit"] == circuit, "Phys. Dist. to PC"] = pc_dist_index
        pos_map.loc[pos_map["Circuit"] == circuit, "Phys. Position ODD"] = phys_pos_index_odd
        pos_map.loc[pos_map["Circuit"] == circuit, "Phys. Position EVEN"] = phys_pos_index_even

    # for each circuit get index to sort by
    # current sort is by phys position, so we need to transform the range 0-154 to the oposite sites
    phys_pos_index = pos_map[pos_map.Circuit == 'RB.A12']["Phys. Position"].values
    phys_pos_index_even_in_odd = pos_map[pos_map.Circuit == 'RB.A12']["Phys. Position"].values[::-1]
    phys_pos_index_odd_in_even = pos_map[pos_map.Circuit == 'RB.A23']["Phys. Position"].values[::-1]
    phys_pos_index_even = np.array(
        [[d, c] for c, d in zip(phys_pos_index_even_in_odd[::2], phys_pos_index_even_in_odd[1::2])]).flatten()
    phys_pos_index_odd = np.array(
        [[d, c] for c, d in zip(phys_pos_index_odd_in_even[::2], phys_pos_index_odd_in_even[1::2])]).flatten()

    for circuit in pos_map["Circuit"].unique():
        if int(circuit[4]) % 2 == 0:
            pos_map.loc[pos_map["Circuit"] == circuit, "Phys. Dist. to PC"] = phys_pos_index
            pos_map.loc[pos_map["Circuit"] == circuit, "Phys. Position ODD"] = phys_pos_index_odd
            pos_map.loc[pos_map["Circuit"] == circuit, "Phys. Position EVEN"] = phys_pos_index
        else:
            # it already works for odd circuits
            pos_map.loc[pos_map["Circuit"] == circuit, "Phys. Dist. to PC"] = phys_pos_index_even_in_odd
            pos_map.loc[pos_map["Circuit"] == circuit, "Phys. Position ODD"] = phys_pos_index
            pos_map.loc[pos_map["Circuit"] == circuit, "Phys. Position EVEN"] = phys_pos_index_even

    # inflate df, add quench position. Size of df gets (8*154)*154
    df_list = []
    for q in pos_map[pos_map.Circuit == 'RB.A12']["Phys. Position"].values:
        for circuit in pos_map["Circuit"].unique():
            el_quench_pos = \
                map_position_index(pos_map, old_index=np.array(q, dtype=int), origin="El. Position",
                                   to="Phys. Position",
                                   filt={"Circuit": circuit})[0]
            pos_map.loc[pos_map["Circuit"] == circuit, "El. Quench Position"] = el_quench_pos
        pos_map["Phys. Quench Position"] = q
        df_list.append(pos_map.copy())

    pos_map_q = pd.concat(df_list)
    # add Distances to Quench
    for circuit in pos_map["Circuit"].unique():
        for q in np.arange(154):
            el_quench_pos = map_position_index(pos_map, old_index=[q], origin="El. Position", to="Phys. Position",
                                               filt={"Circuit": circuit})[0]

            phys_bool = (pos_map_q["Circuit"] == circuit) & (pos_map_q["Phys. Quench Position"] == q)
            df = pos_map_q[phys_bool]
            pos_map_q.loc[phys_bool, "Phys. Dist. to Quench"] = nanargsort(center_array(df["Phys. Position"].values, q))

    # add Distances to Quench el. Position
    # switch to el. Position sort
    pos_map_q = pos_map_q.sort_values(by="El. Position")
    for circuit in pos_map["Circuit"].unique():
        for q in np.arange(154):
            el_bool = ((pos_map_q["Circuit"] == circuit) & (pos_map_q["El. Quench Position"] == q)).values
            df = pos_map_q[el_bool]

            main_index, mirror_index = split_main_mirror(df["El. Position"].values, q)
            el_bool_full_main = el_bool.copy()
            el_bool_main = el_bool_full_main[el_bool]
            el_bool_main[mirror_index[mirror_index >= 0].astype(int)] = False
            el_bool_full_main[el_bool] = el_bool_main

            el_bool_full_mirror = el_bool.copy()
            el_bool_mirror = el_bool_full_mirror[el_bool]
            el_bool_mirror[main_index[main_index >= 0].astype(int)] = False
            el_bool_full_mirror[el_bool] = el_bool_mirror

            pos_map_q.loc[el_bool_full_main, "El. Dist. to Quench Main"] = nanargsort(main_index)
            pos_map_q.loc[el_bool_full_mirror, "El. Dist. to Quench Mirror"] = nanargsort(mirror_index)

    # convert float to int
    nan_columns = pos_map_q.isna().any(axis=0)
    int_columns = pos_map_q.loc[:, ~nan_columns].drop(columns=['Circuit', 'Magnet']).columns
    pos_map_q[int_columns] = pos_map_q[int_columns].astype(int)

    if print_output:
        # print example for odd and even circuit
        filt = {"Circuit": "RB.A12", "El. Quench Position": 0}
        print(pd_dict_filt(pos_map_q, filt).sort_values(by="El. Position").reset_index(drop=True).T.to_string())
        filt = {"Circuit": "RB.A23", "El. Quench Position": 0}
        print(pd_dict_filt(pos_map_q, filt).sort_values(by="El. Position").reset_index(drop=True).T.to_string())

        pos_map_q.reset_index(drop=True).to_csv(output_dir / "position_mapping.csv", index=False)


if __name__ == "__main__":
    data_dir = Path("../RB_position_context/")
    output_dir = Path("")
    file_name = "RB_position_context"

    generate_position_mapping(input_dir=data_dir, output_dir=output_dir, file_name=file_name, print_output=True)
