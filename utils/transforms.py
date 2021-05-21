""" A collection of functions to transform data. """

import numpy as np


def unnatural_to_natural_s2_band_order(s2):
    """ Put the B08 after B07.  """

    return s2[[0, 1, 2, 4, 5, 6, 3, 7, 8, 9], ...]
