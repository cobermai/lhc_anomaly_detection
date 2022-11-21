import warnings
from typing import Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors

warnings.filterwarnings('ignore')

def plot_position_frequency_map(ax, x_fft, frequency,
                                norm: Optional[colors.LogNorm] = colors.LogNorm, vmin=1e-5, vmax=1e-2):
    if norm is None: 
        im = ax.imshow(x_fft.T,
                       extent=[1, 154, frequency.min(), frequency.max()],
                       aspect='auto',
                       vmin=vmin,
                       vmax=vmax,
                       origin='lower')
    else:
        im = ax.imshow(x_fft.T,
                       extent=[1, 154, frequency.min(), frequency.max()],
                       aspect='auto',
                       norm=norm(vmin=vmin, vmax=vmax),
                       origin='lower')
    plt.tight_layout()
    return im


def plot_circuit_frequencies_phys_pos(ax, x_fft, frequency, rb_magnet_metadata_subset):
    phys_pos_index = rb_magnet_metadata_subset['#Electric_circuit'].values

    im = plot_position_frequency_map(ax, x_fft[phys_pos_index - 1], frequency)
    ax.set_ylabel('Frequency / Hz')
    ax.set_xlabel('Phys. Position')
    ax.set_xticks(np.arange(1, 155)[::9])

    rb_magnet_metadata_subset['cryostat_group'] = \
        rb_magnet_metadata_subset['Cryostat2'].apply(lambda x: x.split('_')[1])
    last_mpos_in_cryostat = rb_magnet_metadata_subset.groupby("cryostat_group").min().sort_values(
        by='phys_pos').phys_pos.values
    ax.set_xticks(last_mpos_in_cryostat)
    tick_labels = [str(l) if i % 3 == 0 else '' for i, l in enumerate(last_mpos_in_cryostat)]
    ax.set_xticklabels(tick_labels)

    plt.tight_layout()
    return im

def plot_circuit_frequencies(ax, x_fft, frequency):
    im = plot_position_frequency_map(ax, x_fft, frequency)

    ax.set_ylabel('Frequency / Hz')
    ax.set_xlabel('El. Position')
    ax.set_xticks(np.arange(1, 155)[::9])

    plt.tight_layout()
    return im

def plot_position_frequency_map_ee_plateau(fpa_identifier,
                                           dataset_1EE,
                                           dataset_1EE_fft,
                                           dataset_2EE_fft,
                                           dataset_2EE,
                                           mp3_fpa_df,
                                           rb_magnet_metadata,
                                           circuit_imgs: dict,
                                           filename):
    n_magnets = len(dataset_1EE.loc[{'event': fpa_identifier}].data)
    circuit = fpa_identifier.split('_')[1]
    fig, ax = plt.subplots(4,4, figsize=(25,12), gridspec_kw={'height_ratios': [0.2, 1.2, 5,5], 'width_ratios': [4, 4, 4, 1]})
    date = mp3_fpa_df[mp3_fpa_df['fpa_identifier'] == fpa_identifier]['Timestamp_PIC'].values[0]

    mp3_fpa_df_subset = mp3_fpa_df[(mp3_fpa_df['fpa_identifier'] == fpa_identifier)]
    current = mp3_fpa_df_subset['I_end_2_from_data'].values[0]
    mp3_fpa_df_subset_fast = mp3_fpa_df_subset[mp3_fpa_df_subset['Delta_t(iQPS-PIC)'] / 1000 < 1]

    rb_magnet_metadata_subset = rb_magnet_metadata[rb_magnet_metadata.Circuit==circuit]

    prim_quench_position = mp3_fpa_df_subset_fast['#Electric_circuit'].values[0]
    sec_quench_position = mp3_fpa_df_subset_fast['#Electric_circuit'].values[1:]

    prim_quench_position_phys = mp3_fpa_df_subset_fast['phys_position'].values[0]
    sec_quench_position_phys = mp3_fpa_df_subset_fast['phys_position'].values[1:]
    sec_quench_times = mp3_fpa_df_subset_fast['Delta_t(iQPS-PIC)'].values[1:]

    sec_quench_phys = [f"{int(pos)}@{int(time)}ms" for pos, time in zip(sec_quench_position_phys, sec_quench_times)]
    sec_quench_el = [f"{int(pos)}@{int(time)}ms" for pos, time in zip(sec_quench_position, sec_quench_times)]

    ax[0,0].text(0,1, f"FPA identifier: {fpa_identifier} \nDate: {date} \nMax. Current: {current} A")

    ax[0,1].text(0,3,   f"El. Position Primary")
    ax[0,1].text(0,1.5, f"Primary quench position: {int(prim_quench_position)}", c="r")
    ax[0,1].text(0,0,   f"Fast secondary quench: {sec_quench_el}", c="orange")


    ax[0,2].text(0,3,   f"Phys. Position Primary")
    ax[0,2].text(0,1.5, f"Primary quench position: {int(prim_quench_position_phys)}", c="r")
    ax[0,2].text(0,0,   f"Fast secondary quench: {sec_quench_phys}", c="orange")

    if int(circuit[-2]) %2==0:
        ax[1,1].imshow(circuit_imgs["el_pos_even"])
        ax[1,2].imshow(circuit_imgs["phys_pos_even"])
    else:
        ax[1,1].imshow(circuit_imgs["el_pos_odd"])
        ax[1,2].imshow(circuit_imgs["phys_pos_odd"])

    ax[1,1].set_title(f'Sector: {circuit}')
    ax[1,2].set_title(f'Sector: {circuit}')

    ax[1,0].set_axis_off()
    ax[1,1].set_axis_off()
    ax[1,2].set_axis_off()

    ax[0,0].set_axis_off()
    ax[0,1].set_axis_off()
    ax[0,2].set_axis_off()

    ax[0,3].set_axis_off()
    ax[1,3].set_axis_off()
    ax[2,3].set_axis_off()
    ax[3,3].set_axis_off()

    ax[1, 0].set_title("U_Diode Signals")
    # colors_1 = list(zip(np.linspace(0.5, 1 ,128), mpl.cm.jet(np.linspace(0,1,128))))
    # colors_2 = list(zip(np.linspace(0,0.5,128), mpl.cm.jet(np.linspace(1,0,128))))
    cmap = mpl.cm.jet #mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors_2 + colors_1)
    norm = mpl.colors.Normalize(vmin=1, vmax=154)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[1, 0], fraction=1, orientation = 'horizontal')
    cbar.set_label('El. Position')


    for i, x in enumerate(dataset_1EE.loc[{'event': fpa_identifier}].data):
        ax[2, 0].plot(dataset_1EE.loc[{'event': fpa_identifier}].time, x, c=cmap(i/n_magnets))
    ax[2, 0].grid()
    ax[2, 0].set_title(f"U_Diode @ 1st EE plateau")
    ax[2, 0].set_xlabel('Time / s')
    ax[2, 0].set_ylabel('Voltage / V')
    for t in sec_quench_times:
        ax[2, 0].axvline(x = t / 1000, color = 'r')
    ax[2, 0].set_xlim(dataset_1EE.loc[{'event': fpa_identifier}].time.min(),dataset_1EE.loc[{'event': fpa_identifier}].time.max())

    x_fft = dataset_1EE_fft.loc[{'event': fpa_identifier}].data
    frequency = dataset_1EE_fft.loc[{'event': fpa_identifier}].frequency
    im = plot_circuit_frequencies(ax[2, 1], x_fft, frequency)
    ax[2, 1].set_title(f'El. Position')
    ax[2, 1].tick_params(axis='x', colors='red')

    im1 = plot_circuit_frequencies_phys_pos(ax[2, 2], x_fft, frequency, rb_magnet_metadata_subset)
    ax[2, 2].set_title(f'Phys. Position')
    ax[2, 2].grid(linewidth=0.2)
    ax[2, 1].axvline(x = prim_quench_position, color = 'red')
    ax[2, 2].axvline(x = prim_quench_position_phys, color = 'red')
    for e, p in zip(sec_quench_position, sec_quench_position_phys):
        ax[2, 1].axvline(x = e, color = 'orange')
        ax[2, 2].axvline(x = p, color = 'orange')
    cbar = fig.colorbar(im, ax=ax[2, 3],fraction=1)
    cbar.set_label('Voltage / V')


    for i, x in enumerate(dataset_2EE.loc[{'event': fpa_identifier}].data):
        ax[3, 0].plot(dataset_2EE.loc[{'event': fpa_identifier}].time, x, c=cmap(i/n_magnets))
    ax[3, 0].grid()
    ax[3, 0].set_title(f"U_Diode @ 2nd EE plateau")
    ax[3, 0].set_xlabel('Time / s')
    ax[3, 0].set_ylabel('Voltage / V')
    for t in sec_quench_times:
        ax[3, 0].axvline(x = t / 1000, color = 'r')
    ax[3, 0].set_xlim(dataset_2EE.loc[{'event': fpa_identifier}].time.min(), dataset_2EE.loc[{'event': fpa_identifier}].time.max())
    #ax[3, 0].text(0.45, -4.9, "12")

    x_fft = dataset_2EE_fft.loc[{'event': fpa_identifier}].data
    frequency = dataset_2EE_fft.loc[{'event': fpa_identifier}].frequency
    im = plot_circuit_frequencies(ax[3, 1], x_fft, frequency)
    ax[3, 1].set_title(f'El. Position')
    ax[3, 1].tick_params(axis='x', colors='red')


    im1 = plot_circuit_frequencies_phys_pos(ax[3, 2], x_fft, frequency, rb_magnet_metadata_subset)
    ax[3, 2].set_title(f'Phys. Position')
    ax[3, 2].grid(linewidth=0.2)
    ax[3, 1].axvline(x = prim_quench_position, color = 'red')
    ax[3, 2].axvline(x = prim_quench_position_phys, color = 'red')
    for e, p in zip(sec_quench_position, sec_quench_position_phys):
        ax[3, 1].axvline(x = e, color = 'orange')
        ax[3, 2].axvline(x = p, color = 'orange')
    cbar = fig.colorbar(im, ax=ax[3, 3], fraction=1)
    cbar.set_label('Voltage / V')

    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close(fig)

def plot_NMF_components(X, W, H, frequency_cut: Optional[pd.DataFrame]=None, event_idex=1, mp3_fpa_df_subset: Optional[pd.DataFrame]=None, hyperparameters: Optional[dict]=None):
    image_len = 154
    x_fft_cut = X[event_idex * image_len: event_idex * image_len + image_len]
    if frequency_cut is None:
        frequency_cut = np.arange(len(x_fft_cut[0]))
    if hyperparameters:
        fig, ax = plt.subplots(2,3, figsize=(20,12))
    else:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    plot_position_frequency_map(ax[0, 0], x_fft_cut, frequency_cut, norm=None)
    ax[0, 0].set_ylabel('Frequency / Hz')
    ax[0, 0].set_xlabel('Position')
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
    ax[1, 1].set_ylabel('Frequency / Hz')
    ax[1, 1].set_xlabel('Position')

    if hyperparameters:
        ax[0, 2].set_axis_off()
        ax[1, 2].set_axis_off()
        ax[0, 2].text(0, 1, "Hyperparameters", fontsize="x-large")
        i = 1
        for key, value in hyperparameters.items():
            ax[0, 2].text(0, 1 - i * 0.05, f"{key} = {value}", fontsize="large", va="top")
            i += 1

    plt.tight_layout()
    return ax