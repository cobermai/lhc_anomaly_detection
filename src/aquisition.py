from abc import ABC, abstractmethod
import numpy as np


class DataAcquisition(ABC):
    """
    abstract class which acts as a template to download LHC cicuit data
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int
                 ):
        """
        Specifies data to query from
        """
        self.circuit_type = circuit_type
        self.circuit_name = circuit_name
        self.timestamp_fgc = timestamp_fgc

    def get_signal_timestamp(self) -> int:
        """
        abstract method to find correct timestamp for selected signal, default is fgc timestamp
        """
        return self.timestamp_fgc

    @abstractmethod
    def get_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """

    @abstractmethod
    def get_reference_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """


def acquire_data(creator: DataAcquisition, circuit_type, circuit_name, timestamp_fgc) -> list:
    return creator(circuit_type, circuit_name, timestamp_fgc).get_signal_data()
