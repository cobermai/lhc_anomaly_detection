from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors


def plot_position_frequency_map(ax, x_fft, frequency,
                                norm: Optional[colors.LogNorm] = colors.LogNorm(vmin=1e-5, vmax=1e-2)):
    im = ax.imshow(x_fft.T,
                   extent=[1, 154, frequency.min(), frequency.max()],
                   aspect='auto',
                   norm=norm,
                   origin='lower')
    ax.set_ylabel('Frequency / Hz')
    ax.set_xlabel('El. Distance to Quench')
    plt.tight_layout()
    return im

def plot_NMF_components(X, W, H, frequency_cut: Optional[pd.DataFrame]=None, event_idex=1, mp3_fpa_df_subset: Optional[pd.DataFrame]=None):
    image_len = 154
    x_fft_cut = X[event_idex * image_len: event_idex * image_len + image_len]
    if frequency_cut is None:
        frequency_cut = np.arange(len(x_fft_cut[0]))
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plot_position_frequency_map(ax[0, 0], x_fft_cut, frequency_cut, norm=None)
    ax[0, 0].set_ylabel('frequency')
    ax[0, 0].set_xlabel('position')
    if mp3_fpa_df_subset is not None:
        ax[0, 0].set_title(f'original image {mp3_fpa_df_subset.fpa_identifier.values[0]}\n'
                           f'{mp3_fpa_df_subset.Timestamp_PIC.values[0]}')

    ax[1, 0].plot(W[event_idex * image_len:event_idex * image_len + image_len])
    ax[1, 0].set_ylabel('value')
    ax[1, 0].set_xlabel('position')
    ax[1, 0].set_title(f'component weight {event_idex}')
    ax[1, 0].legend([f"component {i}" for i in range(len(H))])
    ax[1, 0].set_xlim([0, image_len])

    ax[0, 1].plot(frequency_cut, H.T)
    ax[0, 1].set_xlabel('frequency')
    ax[0, 1].set_ylabel('value')
    ax[0, 1].set_title('reconstructed frequency components')
    ax[0, 1].legend([f"component {i}" for i in range(len(H))])
    ax[0, 1].set_xlim([frequency_cut.min(), frequency_cut.max()])

    x_fft_reconstructed = W[event_idex * image_len:event_idex * image_len + image_len] @ H
    plot_position_frequency_map(ax[1, 1], x_fft_reconstructed, frequency_cut, norm=None)#, vmin=None, vmax=None)
    ax[1, 1].set_ylabel('frequency')
    ax[1, 1].set_xlabel('position')


    plt.tight_layout()
    return ax