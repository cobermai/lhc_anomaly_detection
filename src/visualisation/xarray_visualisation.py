from matplotlib import pyplot as plt


def plot_xarray_event(dataset, data_var, idx=0):
    # TODO: descaling must be applied on scale_dims (for now its on even-> only works for ts)
    fig, ax = plt.subplots(2, 1)
    ax[1].set_title('scaled signal')
    ax[1].plot(dataset[{"event": idx}][data_var].values[0])

    ax[0].plot(dataset[{"event": idx}][data_var].values[0]
               * dataset[data_var].attrs["train_scale_coef"][1][idx]
               + dataset[data_var].attrs["train_scale_coef"][0][idx])
    ax[0].set_title('restored signal')
    plt.tight_layout()
    plt.show()