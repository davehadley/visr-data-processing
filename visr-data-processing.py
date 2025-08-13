import h5py
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from numpy.typing import NDArray


def main():
    with h5py.File("spectroscopy_detector.h5") as file:
        raw_values = load_raw_photo_sensor_grayscale_values(file)
    grayscale_values = normalize_values(raw_values)
    figure = plot_grayscale(grayscale_values)
    figure.savefig("sample_image.png")


def normalize_values(raw_rgb_values: NDArray[np.int32]) -> NDArray[np.float64]:
    return raw_rgb_values / float(np.max(raw_rgb_values))


def load_raw_photo_sensor_grayscale_values(file) -> NDArray[np.int32]:
    # TODO: update when green and blue is available in the dataset
    red_name = green_name = blue_name = "entry/instrument/NDAttributes/RedTotal"
    dtype = np.int32
    red = np.array(file[red_name], dtype=dtype)
    green = np.array(file[green_name], dtype=dtype)
    blue = np.array(file[blue_name], dtype=dtype)
    if not red.shape == green.shape == blue.shape:
        raise ValueError("invalid input data RBG array shapes don't match")
    rgb = np.vstack((red, green, blue))
    assert rgb.shape == (3, red.shape[0])
    return rgb


def plot_grayscale(rgb_values: NDArray[np.float64]) -> plt.Figure:
    grayscale = np.mean(rgb_values, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(grayscale, cmap="gray")
    ax.set_title("Grayscale")
    ax.axis("off")
    return fig


if __name__ == "__main__":
    main()
