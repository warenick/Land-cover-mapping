""" A collection of functions to visualize patches of data. """

import numpy as np


def plot_s1_false_color_ratio(s1, axis, **kwargs):
    """Plot a Sentinel-1 image into an axis in false-color VV/VH/Ratio.

    Args:
       s1 (numpy.ndarray): Sentinel-1 image. 2 bands: VV, VH.
       axis: Matplotlib Axis to plot into.

    """

    # add a polarization ratio channel
    image = np.empty(shape=(3,) + s1.shape[1:])  # new shape is (3, H, W)
    image[:2] = s1  # the first two channels are the same
    image[2] = s1[0] - s1[1]  # the data is in dB, so ratio becomes difference

    # normalize channel-wise and swap axes from CHW to HWC
    shifted = image - image.min(axis=(1, 2), keepdims=True)
    spread = image.max(axis=(1, 2), keepdims=True) - image.min(
        axis=(1, 2), keepdims=True
    )

    s1 = np.transpose(shifted / spread, [1, 2, 0])

    image = axis.imshow(s1)

    return image


def plot_s2_rgb(s2, axis, **kwargs):
    """Plot a Sentinel-2 image into an axis in true colors (B4, B3, B2).

    Args:
        s2 (numpy.ndarray): Sentinel-2 image. 10 bands: 2, 3, 4, 5, 6, 7, 8, 8A, 11, 12.
        axis: Matplotlib Axis to plot into.

    """

    rgb = s2[2::-1]
    rgb -= rgb.min(axis=(1, 2), keepdims=True)
    rgb /= rgb.max(axis=(1, 2), keepdims=True) - rgb.min(axis=(1, 2), keepdims=True)
    rgb = np.transpose(rgb, [1, 2, 0])

    image = axis.imshow(rgb, **kwargs)

    return image


def plot_s2_fasle_color_infrared(s2, axis, **kwargs):
    """Plot a Sentinel-2 image into an axis in false-color infrared (B8, B4, B3).

    Args:
        s2 (numpy.ndarray): Sentinel-2 image. 10 bands: 2, 3, 4, 5, 6, 7, 8, 8A, 11, 12.
        axis: Matplotlib Axis to plot into.

    """

    rgb = s2[[6, 2, 1], ...]
    rgb -= rgb.min(axis=(1, 2), keepdims=True)
    rgb /= rgb.max(axis=(1, 2), keepdims=True) - rgb.min(axis=(1, 2), keepdims=True)
    rgb = np.transpose(rgb, [1, 2, 0])

    image = axis.imshow(rgb, **kwargs)

    return image


def plot_s2_fasle_color_urban(s2, axis, **kwargs):
    """Plot a Sentinel-2 image into an axis in false-color urban (B12, B11, B4).

    Args:
        s2 (numpy.ndarray): Sentinel-2 image. 10 bands: 2, 3, 4, 5, 6, 7, 8, 8A, 11, 12.
        axis: Matplotlib Axis to plot into.

    """

    rgb = s2[[9, 8, 2], ...]
    rgb -= rgb.min(axis=(1, 2), keepdims=True)
    rgb /= rgb.max(axis=(1, 2), keepdims=True) - rgb.min(axis=(1, 2), keepdims=True)
    rgb = np.transpose(rgb, [1, 2, 0])

    image = axis.imshow(rgb, **kwargs)

    return image


def plot_s2_fasle_color_landwater(s2, axis, **kwargs):
    """Plot a Sentinel-2 image into an axis in false-color land/water (B8, B11, B4).

    Args:
        s2 (numpy.ndarray): Sentinel-2 image. 10 bands: 2, 3, 4, 5, 6, 7, 8, 8A, 11, 12.
        axis: Matplotlib Axis to plot into.

    """

    rgb = s2[[6, 8, 2], ...]
    rgb -= rgb.min(axis=(1, 2), keepdims=True)
    rgb /= rgb.max(axis=(1, 2), keepdims=True) - rgb.min(axis=(1, 2), keepdims=True)
    rgb = np.transpose(rgb, [1, 2, 0])

    image = axis.imshow(rgb, **kwargs)

    return image


def plot_s2_fasle_color_healthy_vegetation(s2, axis, **kwargs):
    """Plot a Sentinel-2 image into an axis in false-color healty vegetation (B8, B11, B2).

    Args:
        s2 (numpy.ndarray): Sentinel-2 image. 10 bands: 2, 3, 4, 5, 6, 7, 8, 8A, 11, 12.
        axis: Matplotlib Axis to plot into.

    """

    rgb = s2[[6, 8, 0], ...]
    rgb -= rgb.min(axis=(1, 2), keepdims=True)
    rgb /= rgb.max(axis=(1, 2), keepdims=True) - rgb.min(axis=(1, 2), keepdims=True)
    rgb = np.transpose(rgb, [1, 2, 0])

    image = axis.imshow(rgb, **kwargs)

    return image
