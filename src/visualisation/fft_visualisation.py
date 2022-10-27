from matplotlib import pyplot as plt, colors


def plot_position_frequency_map(ax, x_fft, frequency, norm=colors.LogNorm(vmin=1e-5, vmax=1e-2)):
    im = ax.imshow(x_fft.T,
                   extent=[1, 154, frequency.min(), frequency.max()],
                   aspect='auto',
                   norm=norm,
                   origin='lower')
    ax.set_ylabel('Frequency / Hz')
    ax.set_xlabel('El. Distance to Quench')
    plt.tight_layout()
    return im

def plot_NMF_components(x_fft_cut, frequency_cut, W, H, event_idex, mp3_fpa_df_subset):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plot_position_frequency_map(ax[0, 0], x_fft_cut[event_idex], frequency_cut)
    ax[0, 0].set_ylabel('frequency')
    ax[0, 0].set_xlabel('position')
    ax[0, 0].set_title(f'original image {mp3_fpa_df_subset.fpa_identifier.values[0]}\n'
                       f'{mp3_fpa_df_subset.Timestamp_PIC.values[0]}')

    ax[1, 0].plot(W[event_idex * 154:event_idex * 154 + 154])
    ax[1, 0].set_ylabel('value')
    ax[1, 0].set_xlabel('position')
    ax[1, 0].set_title(f'component weight {event_idex}')
    ax[1, 0].legend([f"component {i}" for i in range(len(H))])
    ax[1, 0].set_xlim([0, 154])

    ax[0, 1].plot(H.T, frequency_cut)
    ax[0, 1].set_ylabel('frequency')
    ax[0, 1].set_xlabel('value')
    ax[0, 1].set_title('reconstructed frequency components')
    ax[0, 1].legend([f"component {i}" for i in range(len(H))])
    ax[0, 1].set_ylim([frequency_cut.min(), frequency_cut.max()])

    x_fft_reconstructed = W[event_idex * 154:event_idex * 154 + 154] @ H
    plot_position_frequency_map(ax[1, 1], x_fft_reconstructed, frequency_cut, norm=None)#, vmin=None, vmax=None)
    ax[1, 1].set_ylabel('frequency')
    ax[1, 1].set_xlabel('position')
    ax[1, 1].set_title(f'reconstructed image')

    plt.tight_layout()
    return ax