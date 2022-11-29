from typing import Optional

import numpy as np
import pandas as pd

from src.utils.utils import pd_dict_filt


def calc_snr(mu: np.array, sigma: np.array) -> float:
    """
    calculates signal to noise ratio
    :param mu: mean of data
    :param sigma: standard deviation of data
    :return: signal to noise ratio
    """
    snr = abs(np.where(np.nan_to_num(sigma) == 0, 0, mu / sigma))
    return snr


def map_position_index(pos_map: pd.DataFrame,
                       origin: str = "El. Position",
                       to: str = "Phys. Position",
                       filt: Optional[dict] = None,
                       old_index: np.array = np.arange(0, 154)):
    """
    function maps origin index to new index, nan values are not taken
    e.g. RB.A12 origin="El. Position"=[0, ..., 154] -> to="Phys. Position"=[78, ..., 1]
    :param pos_map: lookup table to map positions
    :param origin: origin index name with which the data is sorted, E.g. "El. Position"
    :param to: origin index name with which the data is sorted, E.g. "Phys. Position"
    :param old_index: old index to sort, default is [0, ..., 153]
    :param filt: optional filter to apply on pos_map, filt keys must be in pos_map (multi)index
    :return: electrical position array index
    """
    pos_map_subset = pd_dict_filt(pos_map, filt)
    new_index = pos_map_subset.sort_values(by=to)[origin].values
    not_nan_values = pos_map_subset.sort_values(by=to)[to].values >= 0
    return new_index[old_index][not_nan_values[old_index]].astype(int)


def center_array(array, center_index):
    """
    shifts array to the right, such that center_index is in the middle. Padding with nan values
    :param array: data to center
    :param center_index: index indicating the middle of the array
    :return: centered array
    """
    mask = np.zeros(((array.shape[0] * 2,) + array.shape[1:])) * np.nan  # mask in both directions necessary
    mask[:len(array)] = array

    roll = len(array) - center_index
    centered_array = np.roll(mask, roll, axis=0)

    return centered_array


def split_main_mirror(matrix: np.array, center: np.array) -> tuple:
    """
    shifts split matrix into right and left part, both parts are centered
    :param matrix: data to split
    :param center: index indicating the middle of the array
    :return: main, mirror data, both centered
    """
    im_len = int(len(matrix) / 2)
    if center > len(matrix) / 2:
        # quench is in second half
        matrix_main = matrix[im_len:]
        matrix_mirror = matrix[:im_len]

        center_main = center - im_len
        center_mirror = len(matrix) - center

    else:
        # center is in first half
        matrix_main = matrix[:im_len]
        matrix_mirror = matrix[im_len:]

        center_main = center
        center_mirror = im_len - center

    main_centered = center_array(matrix_main, center_main)
    mirror_centered = center_array(matrix_mirror, center_mirror)
    return main_centered, mirror_centered


def main_mirror_to_el(main: np.array, mirror: np.array, quench_pos: int) -> np.array:
    """
    transforms main, mirror back to matrix
    :param main: half of matrix, where quench_pos
    :param mirror: other half of matrix
    :param quench_pos: position of quench
    :return: merged matrix
    """
    matrix = np.zeros_like(main)
    center = int(len(matrix) / 2)
    if quench_pos > center:  # quench in second half
        matrix[center:] = np.roll(main, -center + quench_pos, axis=0)[center:]
        matrix[:center] = np.roll(mirror, center - quench_pos, axis=0)[:center]
    else:
        matrix[:center] = np.roll(main, -center + quench_pos, axis=0)[:center]
        matrix[center:] = np.roll(mirror, center - quench_pos, axis=0)[center:]  # 153 - center - quench_pos
    return matrix


def generate_sorted_value_dict(values: np.array, df_pos_map: pd.DataFrame,
                               df_event_context: pd.DataFrame, sort_columns: Optional[list] = None,
                               current_sort: str = 'El. Position') -> dict:
    """
    sorts values by sort_columns and puts them in dict, calculates snr for each entry
    :param values: values to sort, dim: (events, n_magnets, n_components)
    :param df_pos_map: lookup table to map positions, must contain sort_columns
    :param df_event_context: Dataframe which contains 'Circuit', "El. Quench Position" and "Phys. Quench Position"
    for each event
    :param sort_columns: columns to sort by
    :param current_sort: current sort of df_pos_map
    :return: sorted dict with values, index and snr
    """
    el_filt_list = df_event_context[['Circuit', "El. Quench Position"]].to_dict(orient='records')
    phys_filt_list = df_event_context[['Circuit', "Phys. Quench Position"]].to_dict(orient='records')

    if sort_columns is None:
        sort_columns = ['El. Position', 'Phys. Position', 'Phys. Position ODD', 'Phys. Position EVEN',
                        'Phys. Dist. to PC', 'Phys. Dist. to Quench', 'El. Dist. to Quench Main',
                        'El. Dist. to Quench Mirror']

    c_weights_dict = {}
    for target in sort_columns:
        print(target)
        max_index = int(df_pos_map[target].max())
        if "El." in target:
            filt_list = el_filt_list
        elif "Phys." in target:
            filt_list = phys_filt_list

        mask = np.empty((values.shape[0], max_index + 1, values.shape[-1])) * np.nan
        for i, f in enumerate(filt_list):
            index = map_position_index(df_pos_map, origin=current_sort, to=target, filt=f)
            target_index = map_position_index(df_pos_map, origin=target, to=target, filt=f)
            mask[i][target_index] = values[i][index]

        if 'Quench' in target:
            x_label = np.arange(-int((max_index + 1) / 2), int((max_index + 2) / 2))
        else:
            x_label = np.arange(1, int(max_index + 2))

        c_weights_dict[target] = {"values": mask,
                                  "index": x_label}
    # add snr for each method
    for sort in c_weights_dict:
        y = np.nanmean(c_weights_dict[sort]["values"], axis=0)
        error = np.nanstd(c_weights_dict[sort]["values"], axis=0)
        c_weights_dict[sort]["snr"] = np.nanmean(calc_snr(y, error), axis=0)

    return c_weights_dict