"""Loaders for the datasets that ship with the repository.

The ADHD-200 subject tables have their own loader in
:func:`ibb_adhd.data.load_groups`; this module covers the cortical-thickness
application, whose inputs are two small ``.mat`` files.
"""

import numpy as np
import scipy.io as sio

from .config import RAW_CT, RAW_CT_KEY


def load_cortical_thickness(paths=None, key=RAW_CT_KEY):
    """Load the gifted / control cortical-thickness matrices.

    Both are ``(n_subjects, 308)`` vertex-wise thickness measurements with far
    more vertices than subjects (15 and 14), which is the regime the estimator
    is built for.

    Parameters
    ----------
    paths : dict, optional
        ``{"control": path, "gifted": path}``; defaults to
        :data:`ibb.config.RAW_CT`.
    key : str
        Variable name inside the ``.mat`` files.

    Returns
    -------
    control : ndarray, shape (14, 308)
    gifted : ndarray, shape (15, 308)
    """
    paths = paths or RAW_CT

    out = {}
    for group in ("control", "gifted"):
        path = paths[group]
        mat = sio.loadmat(str(path))
        if key not in mat:
            available = [k for k in mat if not k.startswith("__")]
            raise KeyError(f"{path} has no variable {key!r}; found {available}")
        out[group] = np.asarray(mat[key], dtype=float)

    return out["control"], out["gifted"]
