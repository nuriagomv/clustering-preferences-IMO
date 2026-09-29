"""Interpretable preference catalogues on the simplex.

These are the predefined weight vectors used when cluster preferences are
chosen from a catalogue instead of being optimized freely. Rows are clusters,
columns are objectives, and each row sums to one.
"""

import numpy as np


def weight_catalogue(n_objectives):
    """Return the standard catalogue for ``n_objectives`` in ``{2, 3}``.

    The catalogue mixes pure preferences (one objective only), pairwise
    compromises (50/50 and 75/25), and, for three objectives, the uniform
    weight vector. Returns ``None`` when no catalogue is defined.
    """
    if n_objectives == 2:
        return np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.5, 0.5],
        ])
    if n_objectives == 3:
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.75, 0.0],
            [0.75, 0.25, 0.0],
            [0.5, 0.0, 0.5],
            [0.25, 0.0, 0.75],
            [0.75, 0.0, 0.25],
            [0.0, 0.5, 0.5],
            [0.0, 0.25, 0.75],
            [0.0, 0.75, 0.25],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ])
    return None
