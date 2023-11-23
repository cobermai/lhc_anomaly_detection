import json
from abc import ABC
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from src.utils.utils import get_folders_with_suffix
from src.visualisation.NMF_visualization import box_plot


class Result(ABC):
    """
    Abstract class for handling results with default load and save methods.
    :param out_path: The base path where result data is stored.
    :param name: Name of the result, used for identifying and storing the result.
    """

    def __init__(self, out_path: Path, name: str, **kwargs):
        self.out_path = out_path
        self.name = name
        self.result_path = self.out_path / self.name
        self.result_path.mkdir(parents=True, exist_ok=True)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_result(self, **kwargs):
        """
        add individual results to attributes of Result class
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_results(self, **kwargs):
        """
        Update existing attributes by stacking the new values with the old ones.
        Only works for attributes that are numpy arrays or can be converted to numpy arrays.
        """
        for key, new_value in kwargs.items():
            if hasattr(self, key):
                existing_value = getattr(self, key)
                if existing_value is not None:
                    if isinstance(new_value, list):
                        new_value = np.array(new_value)
                    if isinstance(existing_value, list):
                        existing_value = np.array(existing_value)
                    updated_value = np.concatenate((existing_value, new_value))
                    setattr(self, key, updated_value)
            else:
                # If the attribute does not exist, simply set it
                setattr(self, key, new_value)

    def load(self, input_path: Optional[Path] = None):
        """
        Load the object from a JSON file, transform list values to np.arrays,
        convert keys ending with '_path' to Path objects, and only set attributes
        if they do not already exist.
        :param input_path: path where results are stored, assumes they are stored in self.result_path if None
        """
        if input_path is None:
            input_path = self.result_path
        try:
            with open(input_path / "results_db.json", 'r') as file:
                data = json.load(file)
                for key, value in data.items():
                    # Convert list values to numpy arrays
                    if isinstance(value, list):
                        value = np.array(value)
                    # Convert values of keys ending with '_path' to Path objects
                    elif key.endswith('_path'):
                        value = Path(value)

                    # Set attribute if it does not exist yet or is None
                    if not hasattr(self, key) or getattr(self, key) is None:
                        setattr(self, key, value)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {input_path / 'results_db.json'} was not found.")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"The file {input_path / 'results_db.json'} is not a valid JSON file.")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")

    def save(self):
        """
        Save the object to a JSON file.
        """
        # Create a serializable representation of the object
        serializable_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                # Convert Path objects to strings
                serializable_dict[key] = str(value)
            elif isinstance(value, np.ndarray):
                # Convert ndarray objects to lists
                serializable_dict[key] = value.tolist()
            else:
                # Add other values as they are
                serializable_dict[key] = value

        with open(self.result_path / "results_db.json", 'w') as file:
            json.dump(serializable_dict, file, indent=4)


class SensitivityAnalysis:
    """
    A class for performing sensitivity analysis on a set of results.

    :param result_path: The path where result data is stored.
    :param event_identifiers: Optional list of identifiers for events.
    """

    def __init__(self,
                 result_path: Path,
                 event_identifiers: Optional[List[str]] = None):
        self.result_path = result_path
        self.result_names = []
        self.event_identifiers = event_identifiers

    def add_class_result(self, result: Result):
        """
        add class Result to attributes of SensitivityAnalysis
        :param result: object to add
        """
        self.result_names.append(result.name)
        setattr(self, result.name, result)

    def load_results(self, result_class: Result):
        """
        Loads results from the specified path into the analysis.
        :param result_class: The class type of the results to be loaded.
        :type result_class: Result
        """
        if not self.result_names:
            self.resul_names = get_folders_with_suffix(result_path=self.result_path, suffix=".json")
        for result_name in self.resul_names:
            result_instance = result_class(out_path=self.result_path, name=result_name)
            result_instance.load()

            setattr(self, result_instance.name, result_instance)

    def get_attribute_from_results(self, attribute: np.array) -> np.array:
        """
        Retrieves a specific attribute from all loaded results.
        :return: An array of the specified attribute from all results.
        """
        return np.array([self.__getattribute__(r).__getattribute__(attribute) for r in self.result_names])

    def get_outlier_events(self,
                           n_outliers: int,
                           plot_outliers: bool = False,
                           save_outliers: bool = True) -> pd.DataFrame:
        """
        Identifies and saves outlier events based on p-values.
        :param n_outliers: Number of outlier events to identify.
        :param plot_outliers: Flag to indicate whether to plot outliers.
        :param save_outliers: Flag to indicate whether to save outliers as csv
        :return: Dataframe with outlier events.
        """
        p_values = self.get_attribute_from_results("p_values")
        p_median = np.nanmedian(p_values, axis=0)
        sorted_idx = np.argsort(p_median)

        if self.event_identifiers is None:
            outlier_events = sorted_idx
        else:
            outlier_events = self.event_identifiers[sorted_idx]

        if plot_outliers:
            box_plot(p_values[:, sorted_idx[:n_outliers]], outlier_events[:n_outliers], self.result_path)

        df = pd.DataFrame(p_values[:, sorted_idx],
                          index=self.result_names,
                          columns=outlier_events).T

        if save_outliers:
            df.to_csv(self.result_path / "p_values.csv")

        return df.head(n_outliers).T
