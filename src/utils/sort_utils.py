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
