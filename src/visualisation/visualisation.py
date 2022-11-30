from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import re


def plot_hdf(data: list, column_regex: list = [""], fig_path: Path = None):
    """
    :param data: list of dataframes to plot
    :param column_regex: list of regex column names, to plot in one subplot
    :param fig_path: path to store fig
    """
    fig = plt.figure(figsize=(4 * len(column_regex), 3))
    for i, s in enumerate(column_regex):
        fig.add_subplot(1, len(column_regex), i + 1)
        n_signals = 0
        for df in data:
            if bool(re.search(s, df.columns.values[0])):
                # if s in  df.columns.values[0]:
                plt.plot(df.index.values, df.values)
                n_signals += 1
        plt.title(f"{s} ({n_signals})")
    if fig_path is None:
        plt.show()
    else:
        plt.savefig(str(fig_path))
    fig.clf()


def make_gif(im_paths, output_path):
    imgs = (Image.open(f) for f in im_paths)
    img = next(imgs)  # extract first image from iterator
    img.save(fp=output_path / "summary.gif", format='GIF', append_images=imgs, save_all=True, duration=400, loop=0)