import warnings
from typing import Optional
import datetime

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors

from src.utils.sort_utils import calc_snr

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


def plot_circuit_frequencies_phys_pos(ax, x_fft, frequency, rb_magnet_metadata_subset, vmin=1e-5, vmax=1e-2):
    phys_pos_index = rb_magnet_metadata_subset['#Electric_circuit'].values

    im = plot_position_frequency_map(ax, x_fft[phys_pos_index - 1], frequency, vmin=vmin, vmax=vmax)
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


def plot_circuit_frequencies(ax, x_fft, frequency, vmin=1e-5, vmax=1e-2):
    im = plot_position_frequency_map(ax, x_fft, frequency, vmin=vmin, vmax=vmax)

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
                                           filename,
                                           vmin=1e-5,
                                           vmax=1e-2):
    n_magnets = len(dataset_1EE.loc[{'event': fpa_identifier}].data)
    circuit = fpa_identifier.split('_')[1]
    fig, ax = plt.subplots(4, 4, figsize=(25, 12),
                           gridspec_kw={'height_ratios': [0.2, 1.2, 5, 5], 'width_ratios': [4, 4, 4, 1]})
    date = mp3_fpa_df[mp3_fpa_df['fpa_identifier'] == fpa_identifier]['Timestamp_PIC'].values[0]

    mp3_fpa_df_subset = mp3_fpa_df[(mp3_fpa_df['fpa_identifier'] == fpa_identifier)].fillna(0)
    current = mp3_fpa_df_subset['I_Q_circ'].values[0]
    mp3_fpa_df_subset_fast = mp3_fpa_df_subset[mp3_fpa_df_subset['Delta_t(iQPS-PIC)'] / 1000 < 1]

    rb_magnet_metadata_subset = rb_magnet_metadata[rb_magnet_metadata.Circuit == circuit]

    prim_quench_position = mp3_fpa_df_subset_fast['#Electric_circuit'].values[0]
    sec_quench_position = mp3_fpa_df_subset_fast['#Electric_circuit'].values[1:]

    prim_quench_position_phys = mp3_fpa_df_subset_fast['phys_position'].values[0]
    sec_quench_position_phys = mp3_fpa_df_subset_fast['phys_position'].values[1:]
    sec_quench_times = mp3_fpa_df_subset_fast['Delta_t(iQPS-PIC)'].values[1:]

    sec_quench_phys = [f"{int(pos)}@{int(time)}ms" for pos, time in zip(sec_quench_position_phys, sec_quench_times)]
    sec_quench_el = [f"{int(pos)}@{int(time)}ms" for pos, time in zip(sec_quench_position, sec_quench_times)]

    ax[0, 0].text(0, 1, f"FPA identifier: {fpa_identifier} \nDate: {date} \nMax. Current: {current} A")

    ax[0, 1].text(0, 3, f"El. Position Primary")
    ax[0, 1].text(0, 1.5, f"Primary quench position: {int(prim_quench_position)}", c="r")
    ax[0, 1].text(0, 0, f"Fast secondary quench: {sec_quench_el}", c="orange")

    ax[0, 2].text(0, 3, f"Phys. Position Primary")
    ax[0, 2].text(0, 1.5, f"Primary quench position: {int(prim_quench_position_phys)}", c="r")
    ax[0, 2].text(0, 0, f"Fast secondary quench: {sec_quench_phys}", c="orange")

    if int(circuit[-2]) % 2 == 0:
        ax[1, 1].imshow(circuit_imgs["el_pos_even"])
        ax[1, 2].imshow(circuit_imgs["phys_pos_even"])
    else:
        ax[1, 1].imshow(circuit_imgs["el_pos_odd"])
        ax[1, 2].imshow(circuit_imgs["phys_pos_odd"])

    ax[1, 1].set_title(f'Sector: {circuit}')
    ax[1, 2].set_title(f'Sector: {circuit}')

    ax[1, 0].set_axis_off()
    ax[1, 1].set_axis_off()
    ax[1, 2].set_axis_off()

    ax[0, 0].set_axis_off()
    ax[0, 1].set_axis_off()
    ax[0, 2].set_axis_off()

    ax[0, 3].set_axis_off()
    ax[1, 3].set_axis_off()
    ax[2, 3].set_axis_off()
    ax[3, 3].set_axis_off()

    ax[1, 0].set_title("U_Diode Signals")
    # colors_1 = list(zip(np.linspace(0.5, 1 ,128), mpl.cm.jet(np.linspace(0,1,128))))
    # colors_2 = list(zip(np.linspace(0,0.5,128), mpl.cm.jet(np.linspace(1,0,128))))
    cmap = mpl.cm.jet  # mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors_2 + colors_1)
    norm = mpl.colors.Normalize(vmin=1, vmax=154)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[1, 0], fraction=1, orientation='horizontal')
    cbar.set_label('El. Position')

    for i, x in enumerate(dataset_1EE.loc[{'event': fpa_identifier}].data):
        ax[2, 0].plot(dataset_1EE.loc[{'event': fpa_identifier}].time, x, c=cmap(i / n_magnets))
    ax[2, 0].grid()
    ax[2, 0].set_title(f"U_Diode @ 1st EE plateau")
    ax[2, 0].set_xlabel('Time / s')
    ax[2, 0].set_ylabel('Voltage / V')
    for t in sec_quench_times:
        ax[2, 0].axvline(x=t / 1000, color='r')
    ax[2, 0].set_xlim(dataset_1EE.loc[{'event': fpa_identifier}].time.min(),
                      dataset_1EE.loc[{'event': fpa_identifier}].time.max())

    x_fft = dataset_1EE_fft.loc[{'event': fpa_identifier}].data
    frequency = dataset_1EE_fft.loc[{'event': fpa_identifier}].frequency
    im = plot_circuit_frequencies(ax[2, 1], x_fft, frequency, vmin=vmin, vmax=vmax)
    ax[2, 1].set_title(f'El. Position')
    ax[2, 1].tick_params(axis='x', colors='red')

    im1 = plot_circuit_frequencies_phys_pos(ax[2, 2], x_fft, frequency, rb_magnet_metadata_subset, vmin=vmin, vmax=vmax)
    ax[2, 2].set_title(f'Phys. Position')
    ax[2, 2].grid(linewidth=0.2)
    ax[2, 1].axvline(x=prim_quench_position, color='red')
    ax[2, 2].axvline(x=prim_quench_position_phys, color='red')
    for e, p in zip(sec_quench_position, sec_quench_position_phys):
        ax[2, 1].axvline(x=e, color='orange')
        ax[2, 2].axvline(x=p, color='orange')
    cbar = fig.colorbar(im, ax=ax[2, 3], fraction=1)
    cbar.set_label('Voltage / V')

    for i, x in enumerate(dataset_2EE.loc[{'event': fpa_identifier}].data):
        ax[3, 0].plot(dataset_2EE.loc[{'event': fpa_identifier}].time, x, c=cmap(i / n_magnets))
    ax[3, 0].grid()
    ax[3, 0].set_title(f"U_Diode @ 2nd EE plateau")
    ax[3, 0].set_xlabel('Time / s')
    ax[3, 0].set_ylabel('Voltage / V')
    for t in sec_quench_times:
        ax[3, 0].axvline(x=t / 1000, color='r')
    ax[3, 0].set_xlim(dataset_2EE.loc[{'event': fpa_identifier}].time.min(),
                      dataset_2EE.loc[{'event': fpa_identifier}].time.max())
    # ax[3, 0].text(0.45, -4.9, "12")

    x_fft = dataset_2EE_fft.loc[{'event': fpa_identifier}].data
    frequency = dataset_2EE_fft.loc[{'event': fpa_identifier}].frequency
    im = plot_circuit_frequencies(ax[3, 1], x_fft, frequency, vmin=vmin, vmax=vmax)
    ax[3, 1].set_title(f'El. Position')
    ax[3, 1].tick_params(axis='x', colors='red')

    im1 = plot_circuit_frequencies_phys_pos(ax[3, 2], x_fft, frequency, rb_magnet_metadata_subset, vmin=vmin, vmax=vmax)
    ax[3, 2].set_title(f'Phys. Position')
    ax[3, 2].grid(linewidth=0.2)
    ax[3, 1].axvline(x=prim_quench_position, color='red')
    ax[3, 2].axvline(x=prim_quench_position_phys, color='red')
    for e, p in zip(sec_quench_position, sec_quench_position_phys):
        ax[3, 1].axvline(x=e, color='orange')
        ax[3, 2].axvline(x=p, color='orange')
    cbar = fig.colorbar(im, ax=ax[3, 3], fraction=1)
    cbar.set_label('Voltage / V')

    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close(fig)


def plot_NMF(X, W, H,
             frequency: Optional[pd.DataFrame] = None,
             event_idex=1,
             event_context: Optional[pd.DataFrame] = None,
             hyperparameters: Optional[dict] = None):
    image_len = 154
    x_fft_cut = X[event_idex * image_len: event_idex * image_len + image_len]
    if frequency is None:
        frequency = np.arange(len(x_fft_cut[0]))
    if hyperparameters:
        fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    else:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    plot_position_frequency_map(ax[0, 0], x_fft_cut, frequency, norm=None, vmin=0, vmax=1)
    ax[0, 0].set_ylabel('Frequency / Hz')
    ax[0, 0].set_xlabel('Position')
    if event_context is not None:
        ax[0, 0].set_title(f'original image {event_context.fpa_identifier.values[0]}\n'
                           f'{event_context.Timestamp_PIC.values[0]}')

    ax[1, 0].plot(W[event_idex * image_len:event_idex * image_len + image_len])
    ax[1, 0].set_ylabel('value')
    ax[1, 0].set_xlabel('position')
    ax[1, 0].set_title(f'component weight {event_idex}')
    ax[1, 0].legend([f"component {i}" for i in range(len(H))])
    ax[1, 0].set_xlim([0, image_len])

    ax[0, 1].plot(frequency, H.T)
    ax[0, 1].set_xlabel('frequency')
    ax[0, 1].set_ylabel('value')
    ax[0, 1].set_title('reconstructed frequency components')
    ax[0, 1].legend([f"component {i}" for i in range(len(H))])
    ax[0, 1].set_xlim([frequency.min(), frequency.max()])

    x_fft_reconstructed = W[event_idex * image_len:event_idex * image_len + image_len] @ H
    plot_position_frequency_map(ax[1, 1], x_fft_reconstructed, frequency, norm=None, vmin=0, vmax=1)
    ax[1, 1].set_ylabel('Frequency / Hz')
    ax[1, 1].set_xlabel('Position')
    ax[1, 1].set_title(f"reconstructed image \nloss: {np.linalg.norm(X - W @ H):.2f}")

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


def plot_nmf_event_composition(data_1EE, W, H, component_indexes, dataset_1EE_fft, fpa_identifier, mp3_fpa_df,
                               vmin=1e-5, vmax=1e-2):
    # get right event index
    all_fpa_identifiers = mp3_fpa_df.fpa_identifier.unique()
    fpa_identifiers = all_fpa_identifiers[np.isin(all_fpa_identifiers, dataset_1EE_fft.event.values)]
    event_idex = np.argmax(fpa_identifiers == fpa_identifier)

    # get context data of event
    date = mp3_fpa_df[mp3_fpa_df.fpa_identifier == fpa_identifier]['Timestamp_PIC'].values[0]
    mp3_fpa_df_subset = mp3_fpa_df[
        (mp3_fpa_df.fpa_identifier == fpa_identifier)]  # & (mp3_fpa_df['Delta_t(iQPS-PIC)'] / 1000 < 5)]
    current = mp3_fpa_df_subset['I_Q_M'].max()

    if not mp3_fpa_df_subset['Delta_t(iQPS-PIC)'].isnull().all():
        mp3_fpa_df_subset = mp3_fpa_df_subset[(mp3_fpa_df_subset['Delta_t(iQPS-PIC)'] / 1000 < 5)]
        prim_quench_position = mp3_fpa_df_subset['#Electric_circuit'].values[0]
        sec_quench_position = mp3_fpa_df_subset['#Electric_circuit'].values[1:]
        sec_quench_times = mp3_fpa_df_subset['Delta_t(iQPS-PIC)'].values[1:]
        sec_quench_el = [f"{int(pos)}@{int(time)}ms" for pos, time in zip(sec_quench_position, sec_quench_times)]
    else:
        prim_quench_position = ''
        sec_quench_el = ''

    # plot event
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(len(component_indexes), 8, figsize=(20, 2.5 * len(component_indexes)),
                             gridspec_kw={'width_ratios': [2, 1, 2, 2, 1.3, 2, 0.5, 2]})
    for i, ax in enumerate(axes[:, 0]):
        if i == 0:
            plot_position_frequency_map(ax, data_1EE[event_idex * 154:event_idex * 154 + 154],
                                        dataset_1EE_fft.frequency, norm=None, vmin=0, vmax=1)
            ax.set_ylabel('Frequency / Hz')
            ax.set_xlabel('El. Position')
        else:
            ax.axis('off')

    # plot reconstructed event
    for i, ax in enumerate(axes[:, -1]):
        if i == 0:
            plot_position_frequency_map(ax, W[event_idex * 154:event_idex * 154 + 154] @ H, dataset_1EE_fft.frequency,
                                        norm=None, vmin=0, vmax=1)
            ax.set_ylabel('Frequency / Hz')
            ax.set_xlabel('El. Position')
        else:
            ax.axis('off')

    # plot components
    for k, i in enumerate(component_indexes):
        if isinstance(i, list):
            W_temp = 0
            H_temp = 0
            for j in i:
                W_temp += W[event_idex * 154:event_idex * 154 + 154, j:j + 1]
                H_temp += H[j:j + 1]
            H_temp /= len(i)  # sum components, mean weight, or other way around
        else:
            W_temp = W[event_idex * 154:event_idex * 154 + 154, i:i + 1]
            H_temp = H[i:i + 1]
        data_reconstructed = W_temp @ H_temp

        axes[k, 3].plot(W_temp, c=default_colors[k])
        axes[k, 2].plot(dataset_1EE_fft.frequency, H_temp.T, c=default_colors[k])
        plot_position_frequency_map(axes[k, 5], data_reconstructed, dataset_1EE_fft.frequency, norm=None, vmin=0,
                                    vmax=1)

        axes[k, 2].set_ylabel('Component Values')
        axes[k, 3].set_ylabel('Voltage / V')
        axes[k, 3].yaxis.set_label_position("right")
        axes[k, 5].set_ylabel('Frequency / Hz')

        vdiff = np.log10(vmax) - np.log10(vmin)
        n_ticks = int(vdiff) + 1
        axes[k, 3].set_yticks(np.linspace(0, 1, n_ticks))
        axes[k, 3].set_yticklabels([f"$10^{{{int(np.log10(vmin)) + a}}}$" for a in range(n_ticks)], fontsize="large")

        axes[k, 3].yaxis.tick_right()

        if k < len(component_indexes) - 1:
            axes[k, 2].set_xticks([])
            axes[k, 3].set_xticks([])
            axes[k, 5].set_xticks([])
        else:
            axes[k, 2].set_xlabel('Frequency / Hz')
            axes[k, 3].set_xlabel('El. Position')
            axes[k, 5].set_xlabel('El. Position')

    axes[1, 0].text(0, 0.8,
                    f"Event: {fpa_identifier}\nDate: {date}\n$i$={event_idex * 154}\n"
                    f"\nMax Current: {current} A\nEl. Prim Quench Position: {prim_quench_position}"
                    f"\nEl. Sec. Quench Position@Time:\n{sec_quench_el}",
                    va="top")

    [ax.axis('off') for ax in axes[:, 1]]
    [ax.axis('off') for ax in axes[:, 4]]
    [ax.axis('off') for ax in axes[:, 6]]

    axes[0, 0].set_title("Input Event\n$V_{:,i:i+154}$", fontsize=12)
    axes[0, 1].set_title(f"$\sim\sum_{{k=1}}^{{r={len(component_indexes)}}}$", fontsize=15)
    axes[0, 2].set_title("Components $k$\n$W_k$", fontsize=12)
    axes[0, 3].set_title("Components Weight\n$H_{k,i:i+154}$", fontsize=12)
    axes[0, 4].set_title(f"$=\sum_{{k=1}}^{{r={len(component_indexes)}}}$", fontsize=15)
    axes[0, 5].set_title("Reconstructed Components\n$WH_{k,i:i+154}$", fontsize=12)
    axes[0, 6].set_title("=", fontsize=15)
    axes[0, 7].set_title("Reconstructed Event\n$WH_{:,i:i+154}$", fontsize=12)

    plt.tight_layout(h_pad=-2, w_pad=-8.5)


def plot_nmf_components(H, dataset_1EE_fft, W=None, loss=None, component_indexes=None, vmin=None, vmax=None,
                        hyperparameters=None, norm_component=True):
    if norm_component:
        max_W = W.max(axis=0, keepdims=True)
        H = (H * np.expand_dims(max_W.T, axis=0))[0]

    fig, ax = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [2, 1]})
    if component_indexes is None:
        component_indexes = np.arange(len(H))

    for k, i in enumerate(component_indexes):
        if isinstance(i, list):
            H_temp = 0
            for j in i:
                H_temp += H[j:j + 1]
        else:
            H_temp = H[i:i + 1]

        ax[0].plot(dataset_1EE_fft.frequency, H_temp.T, label=f"component {i}")

    if vmin and vmax is not None:
        vdiff = np.log10(vmax) - np.log10(vmin)
        n_ticks = int(vdiff) + 1
        ax[0].set_yticks(np.linspace(0, 1, n_ticks))
        ax[0].set_yticklabels([f"$10^{{{int(np.log10(vmin)) + a}}}$" for a in range(n_ticks)], fontsize="large")

    if hyperparameters:
        ax[1].set_axis_off()
        ax[1].set_axis_off()
        ax[1].text(0, 1, "Hyperparameters", fontsize="x-large")
        i = 1
        for key, value in hyperparameters.items():
            ax[1].text(0, 1 - i * 0.05, f"{key} = {value}", fontsize="large", va="top")
            i += 1

    if loss is not None:
        ax[0].set_title(f"mean loss: {np.mean(loss):.2f}", loc="right")

    ax[0].set_xlim([dataset_1EE_fft.frequency.values.min(), dataset_1EE_fft.frequency.values.max()])
    ax[0].grid()
    ax[0].set_xlabel('Frequency / Hz')
    ax[0].set_ylabel('Voltage / V')
    ax[0].legend()


def plot_avg_component_weight(ax, c_weights, component_number, xlabel):
    y = np.nanmean(c_weights["values"], axis=0)[:, component_number]
    # y_med = np.nanmedian(c_weights["values"], axis=0)[:, component_number]
    error = np.nanstd(c_weights["values"], axis=0)[:, component_number]
    upper_error = np.nanquantile(c_weights["values"], q=0.75, axis=0)[:, component_number]
    lower_error = np.nanquantile(c_weights["values"], q=0.25, axis=0)[:, component_number]

    n_components = c_weights["values"].shape[-1]
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ck = component_number % len(default_colors)

    ax.plot(c_weights["index"], y, c=default_colors[ck])
    # ax.plot(c_weights["index"], y_med, c=default_colors[ck], ls="--")
    # ax.plot(c_weights["index"], y, c=default_colors[ck], marker="x")
    ax.set_title(f"Average Component {component_number} Weight\nSNR: {np.mean(calc_snr(y, error)):.2f}")
    ax.set_xlabel(xlabel)
    ax.fill_between(c_weights["index"], lower_error, upper_error, alpha=0.1, edgecolor=default_colors[ck],
                    facecolor=default_colors[ck])


def plot_distribution_over_column(c_weights_dict, mp3_fpa_df_unique, fpa_identifiers, column, columns_values):
    n_components = c_weights_dict["El. Position"]["values"].shape[-1]
    for k in range(n_components):
        snr_sorted_index = np.argsort([-c_weights_dict[sort]["snr"][k] for sort in c_weights_dict])[0]

        fig, ax = plt.subplots(1, len(columns_values), figsize=(len(columns_values) * 4, 4))
        sort = np.array(list(c_weights_dict))[snr_sorted_index]

        data = c_weights_dict[sort]["values"][..., k]
        y_max = data[np.isfinite(data)].max()

        for i, circuit in enumerate(columns_values):
            circuit_fpa_identifiers = mp3_fpa_df_unique[mp3_fpa_df_unique[column] == circuit].fpa_identifier
            circuit_bool = np.isin(fpa_identifiers, circuit_fpa_identifiers)

            circuit_data_dict = {"values": c_weights_dict[sort]["values"][circuit_bool],
                                 "index": c_weights_dict[sort]["index"]}

            plot_avg_component_weight(ax[i], circuit_data_dict, component_number=k, xlabel=f"{circuit} {sort}")

            ax[i].set_ylim((0, y_max))
            if i == 0:
                yticks = ax[i].get_yticks().tolist()
                ax[i].set_yticklabels([f"$10^{{{(3 * a - 5):.2f}}}$" for a in yticks], fontsize="large")
                ax[i].set_ylabel("Voltage / V")
            else:
                ax[i].set_yticks([])

        plt.tight_layout()


def plot_component_distribution(c_weights_dict, mp3_fpa_df_subset, event_sort, event_sort_ticks, event_index=None,
                                is_date=False):
    if event_index is None:
        event_index = np.ones(len(mp3_fpa_df_subset), dtype=bool)
    n_components = c_weights_dict["El. Position"]["values"].shape[-1]

    mp3_fpa_df_sorted = mp3_fpa_df_subset.reset_index(drop=True)[event_index].sort_values(by=event_sort)
    event_sort_index = mp3_fpa_df_sorted.index.values
    mp3_fpa_df_sorted = mp3_fpa_df_sorted.reset_index()

    first_entry_df = mp3_fpa_df_sorted[event_sort_ticks].drop_duplicates()
    y_tick_index = first_entry_df.index.values
    yticklabels = mp3_fpa_df_sorted.iloc[y_tick_index][event_sort[0]]
    if is_date:
        yticklabels = yticklabels.apply(lambda x: x.strftime(format='%b %d'))

    fig, ax = plt.subplots(1, (n_components + 1), figsize=(4 * (n_components + 1), 10))
    for n in range(n_components):
        snr_sorted_index = np.argsort([-c_weights_dict[sort]["snr"][n] for sort in c_weights_dict])[0]
        position_sort = np.array(list(c_weights_dict))[snr_sorted_index]

        extent = [1, 154, len(mp3_fpa_df_sorted), 0]

        ax[n].set_title(f"component {n}")
        im_data = np.nan_to_num(c_weights_dict[position_sort]["values"][event_sort_index, :, n])
        im = ax[n].imshow(im_data, extent=extent, cmap="magma", origin="upper", aspect="auto", vmin=0, vmax=0.5)
        ax[n].set_xlabel(f"{position_sort}")
        ax[n].set_ylabel(f"{event_sort_ticks}")

        ax[n].set_yticks(y_tick_index[::-1])
        ax[n].set_yticklabels(yticklabels[::-1])

    cbar = fig.colorbar(im, ax=ax[-1], fraction=1)
    cbar.set_label('Voltage / V')
    ax[-1].set_axis_off()
    cticks = cbar.get_ticks().tolist()
    cbar.set_ticks(cticks)
    cbar.set_ticklabels([f"$10^{{{(3 * a - 5):.2f}}}$" for a in cticks])
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()


def plot_cweight_distribution_all_data(components, c_weights_dict, frequency, plot_n_highest_snr=2):
    n_components = c_weights_dict["El. Position"]["values"].shape[-1]
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 5

    best_sort_index = []
    for k in range(n_components):

        snr_sorted_index = np.argsort([-c_weights_dict[sort]["snr"][k] for sort in c_weights_dict])[:plot_n_highest_snr]
        best_sort_index.append(snr_sorted_index[0])

        fig, ax = plt.subplots(1, plot_n_highest_snr + 2, figsize=(5 * (plot_n_highest_snr + 2), 4))
        ax[0].plot(frequency, components[k], c=default_colors[k % n_components])
        ax[0].set_title(f"Component {k}")
        ax[0].set_xlabel("Frequency / Hz")
        ax[0].set_ylabel("Amplitude")
        ax[0].grid()

        if "Quench" in list(c_weights_dict)[snr_sorted_index[0]]:
            ax[0].set_title(f"Component {k}: DYNAMIC")
        else:
            ax[0].set_title(f"Component {k}: STATIC")

        for i, sort in enumerate(np.array(list(c_weights_dict))[snr_sorted_index]):
            plot_avg_component_weight(ax[i + 1], c_weights_dict[sort], component_number=k, xlabel=sort)

            yticks = ax[i + 1].get_yticks().tolist()

            ax[i + 1].set_yticklabels([f"$10^{{{(3 * a - 5):.2f}}}$" for a in yticks], fontsize="large")
            ax[i + 1].set_ylabel("Voltage / V")
            ax[i + 1].set_ylim(ax[1].get_ylim())

        best_sort = list(c_weights_dict)[snr_sorted_index[0]]
        V_mean = np.nanmean(c_weights_dict[best_sort]["values"], axis=0)[:, k:k + 1] @ components[k:k + 1]
        plot_position_frequency_map(ax[-1], V_mean, frequency, norm=None, vmin=0, vmax=1)
        ax[-1].set_xlabel(best_sort)
        ax[-1].set_ylabel("Frequency / Hz")

        plt.tight_layout()
    return best_sort_index


def plot_NMF_loss(loss, mp3_fpa_df_subset, outlier_events):
    def sort_data_by_event_column(data, event_sort, mp3_fpa_df_subset):
        mp3_fpa_df_sorted = mp3_fpa_df_subset.reset_index(drop=True).sort_values(by=event_sort)
        event_sort_index = mp3_fpa_df_sorted.index.values
        x_axis_data = mp3_fpa_df_sorted[event_sort].values
        return x_axis_data, data[event_sort_index]

    mp3_fpa_df_subset['datetime'] = pd.to_datetime(mp3_fpa_df_subset['Date (FGC)'])

    fpa_identifiers_train = mp3_fpa_df_subset.fpa_identifier.values
    outlier_index = np.isin(fpa_identifiers_train, outlier_events)

    sorted_fpa = fpa_identifiers_train[np.argsort(loss)[::-1]]
    n_outlier = np.arange(1, len(sorted_fpa) + 1)[np.isin(sorted_fpa, outlier_events)]

    x_train, y_train = sort_data_by_event_column(loss, ['datetime'], mp3_fpa_df_subset)
    x_outl, y_outl = sort_data_by_event_column(loss[outlier_index], ['datetime'],
                                               mp3_fpa_df_subset[outlier_index])
    plt.figure(figsize=(12, 5))
    plt.plot(x_train, y_train, ".")
    plt.plot(x_outl, y_outl, "o")
    plt.ylabel("NMF Loss")
    plt.xlabel("Date")
    plt.grid()
    plt.xlim([datetime.date(2021, 3, 1), datetime.date(2021, 12, 1)])
    plt.title(f"mean loss: {np.mean(loss):.2f}, outlier rank {n_outlier}")
    plt.tight_layout()
