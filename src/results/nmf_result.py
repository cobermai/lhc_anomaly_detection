from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.models.nmf import _beta_divergence, _beta_loss_to_float
from src.result import Result
from src.visualisation.NMF_visualization import plot_loss_hist


class NMFResult(Result):
    """
    Store results of NMF calculations.

    :param out_path: The path where the NMF results are stored.
    :param name: Name of the NMF result.
    :param components: Optional array of NMF components.
    :param component_weights: Optional array of component weights.
    :param loss: Optional array representing loss values.
    :param p_values: Optional array of p-values.
    :param kwargs: Additional keyword arguments.
    """

    def __init__(self,
                 out_path: Path,
                 name: str,
                 components: Optional[np.array] = None,
                 component_weights: Optional[np.array] = None,
                 loss: Optional[np.array] = None,
                 p_values: Optional[np.array] = None, **kwargs):
        super().__init__(out_path, name, **kwargs)
        self.nmf_loss_per_signal = None
        self.nmf_loss_per_event = None
        self.components = components
        self.component_weights = component_weights
        self.loss = loss
        self.p_values = p_values

    def calculate_nmf_loss_per_signal(self, X_flat: np.array, beta: Optional[str] = 'frobenius'):
        """
        Calculate NMF loss per signal.
        :param X_flat: Input data for NMF.
        :param beta: Type of beta-divergence (default is 'frobenius').
        :raises ValueError: If components or component_weights are not set.
        """
        # Check if components and component_weights are not None
        if self.components is None or self.component_weights is None:
            raise ValueError("components and component_weights not set yet")

        self.nmf_loss_per_signal = np.array([
            _beta_divergence(x_row, w_row, self.components, beta)
            for x_row, w_row in zip(X_flat, self.component_weights)
        ])

    def calculate_nmf_loss_per_event(self, X: np.array, beta: Optional[str] = None):
        """
        Calculate NMF loss per FPA event.
        :param X: Input data for NMF.
        :param beta: Type of beta-divergence (optional).
        :raises ValueError: If components or component_weights are not set.
        """
        # Check if components and component_weights are not None
        if self.components is None or self.component_weights is None:
            raise ValueError("components and component_weights not set yet")

        reshaped_c_weights = np.reshape(self.component_weights, X.shape[:-1] + (self.component_weights.shape[-1],))

        if beta is None:
            na_fft_nmf_amp = (self.component_weights @ self.components).reshape(X.shape)
            loss_sample = X - na_fft_nmf_amp
            masked_arr = np.ma.masked_invalid(loss_sample)
            loss_magnet = np.linalg.norm(masked_arr, axis=2)
            outlier_radius = 3
            self.nmf_loss_per_event = pd.DataFrame(loss_magnet.T).rolling(outlier_radius).mean().max().values
        else:
            self.nmf_loss_per_event = np.array([
                _beta_divergence(x_row, w_row, self.components, beta)
                for x_row, w_row in zip(X, reshaped_c_weights)
            ])

    def calculate_p_values(self, X: np.array, beta: Optional[str] = None, plot_fit=False):
        """
        Calculate p-values for NMF loss.
        :param X: Input data for NMF.
        :param beta: Type of beta-divergence (optional).
        :param plot_fit: Whether to plot the loss histogram of the fit (default is False).
        """
        self.calculate_nmf_loss_per_event(X=X, beta=beta)

        # calculate p values
        params_fit = chi2.fit(self.nmf_loss_per_event, floc=0)  # fit df, and fshape
        self.p_values = 1 - chi2.cdf(self.nmf_loss_per_event, *params_fit)

        if plot_fit:
            plot_loss_hist(self.nmf_loss_per_event, self.result_path, params_fit)