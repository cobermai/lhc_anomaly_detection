
def flatten_list(stacked_list) -> list:
    """
    method flattens list
    :param stacked_list: list with dim > 1
    :return: list with dim = 1
    """
    return [item for sublist in stacked_list for item in sublist]
