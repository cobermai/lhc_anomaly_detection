from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union
from pathlib import Path


class DataAcquisition(ABC):
    """
    abstract class which acts as a template to download LHC cicuit data
    Questions:
    * EE_U_DUMP_RES_PM: query both 'EE_ODD', 'EE_EVEN'?, take only first element of list?
    * EE_T_RES_PM: what is t_res_odd_1_df ? - not implemented yet
    * VOLTAGE_NXCALS: whats the best way to pass spark?
    * VOLTAGE_LOGIC_IQPS: do I need u_qds_dfs2 from second board (A/B)?
    * LEADS: can I pass system as list
    """

    def __init__(self,
                 circuit_type: str,
                 circuit_name: str,
                 timestamp_fgc: int,
                 ):
        """
        Specifies data to query from
        """
        self.circuit_type = circuit_type
        self.circuit_name = circuit_name
        self.timestamp_fgc = timestamp_fgc

    def get_signal_timestamp(self) -> Union[int, pd.DataFrame]:
        """
        method to find correct timestamp for selected signal, default is fgc timestamp
        """
        return self.timestamp_fgc

    @abstractmethod
    def get_signal_data(self) -> list:
        """
        abstract method to get selected signal
        """

    @staticmethod
    def flatten_list(stacked_list) -> list:
        """
        abstract method to flatten list of lists
        """
        return [item for sublist in stacked_list for item in sublist]

    def log_acquisition(self, context_data: dict, context_path: Path) -> None:
        """
        method to store meta data
        """
        identifier = {'circuit_type': self.circuit_type,
                      'circuit_name': self.circuit_name,
                      'timestamp_fgc': self.timestamp_fgc}
        if not context_path.is_file():
            context_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(identifier, index=[0])
        else:
            df = pd.read_csv(context_path)
            # add identifier if not existing
            if not df[identifier.keys()].isin(identifier.values()).all(axis=1).values[-1]:
                print(f"ADDED IDENTIFIER")
                df_new = pd.DataFrame(identifier, index=[0])
                df = pd.concat([df, df_new], axis=0)

        # add context data
        for key, value in context_data.items():
            df.loc[df[identifier.keys()].isin(identifier.values()).all(axis=1), key] = value
        df.to_csv(context_path, index=False)
        return df

    def to_hdf5(self, data_dir: Path) -> None:
        """
        method to store data
        """
        context_path = data_dir / "context_data.csv"
        failed_queries_path = data_dir / "failed_queries.csv"
        hdf_dir = data_dir / "data"
        hdf_dir.mkdir(parents=True, exist_ok=True)

        data = self.get_signal_data()
        try:
            for df in data:
                if isinstance(df, pd.DataFrame):
                    if not df.empty:
                        file_name = f"{self.circuit_type}_{self.circuit_name}_{self.timestamp_fgc}_{df.columns.values[0]}.pkl"
                        df.to_pickle(hdf_dir / file_name)

                        context_data = {f"{df.columns.values[0]}": len(df)}
                        logging_df = self.log_acquisition(context_data=context_data,
                                                          context_path=context_path)
            print(f"finished to download: {str(self.__class__.__name__)}")
            return logging_df

        except Exception as e:
            print(e)
            failed_df = self.log_acquisition(context_data={"error": e},
                                             context_path=failed_queries_path)
            print(f"failed to download: {str(self.__class__.__name__)}")
            return failed_df