"""
Tyson Reimer
University of Manitoba
May 22nd, 2023
"""

import numpy as np

###############################################################################


def apply_sys_cor(xs, ys, d_x=0.26, d_y=0.29, d_phi=-3.0):
    """Apply correction for systematic setup error in bed system

    Parameters
    ----------
    xs : array_like
        Observed x positions of target, in [cm]
    ys : array_like
        Observed y positions of target, in [cm]
    d_x : float
        Systematic x error in observed target position, in [cm]
    d_y : float
        Systematic y error in observed target position, in [cm]
    d_phi : float
        Systematic phi error in observed target position, in [deg]

    Returns
    -------
    cor_xs2 : array_like
        Corrected observed x positions of target, i.e., the position
        of the object in the antenna frame of reference, after
        applying correction for systematic errors, in [cm]
    cor_ys2 : array_like
        Corrected observed y positions of target, i.e., the position
        of the object in the antenna frame of reference, after
        applying correction for systematic errors, in [cm]
    """

    # Apply translation correction
    cor_xs = xs + d_x
    cor_ys = ys + d_y

    d_phi_rad = np.deg2rad(d_phi)  # Convert to radians, [rad]

    # Apply rotation correction
    cor_xs2 = (cor_xs * np.cos(d_phi_rad) - cor_ys * np.sin(d_phi_rad))
    cor_ys2 = (cor_xs * np.sin(d_phi_rad) + cor_ys * np.cos(d_phi_rad))

    return cor_xs2, cor_ys2
