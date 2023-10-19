"""
Tyson Reimer
University of Manitoba
July 07th, 2023
"""

import numpy as np

###############################################################################


def get_loc_err(img, roi_rho, tum_x, tum_y):
    """Return the localization error of the tumor response in the image

    Compute the localization error for the reconstructed image in meters

    Parameters
    ----------
    img : array_like
        The reconstructed image
    roi_rho : float
        The radius of the antenna trajectory during the scan, in meters
    tum_x : float
        The x-position of the tumor during the scan, in meters
    tum_y : float
        The y-position of the tumor during the scan, in meters

    Returns
    -------
    loc_err : float
        The localization error in meters
    """

    # Convert the complex-valued image to format used for display
    img_for_iqm = np.abs(img)**2

    # Find the conversion factor to convert pixel index to distance
    pix_to_dist = 2 * roi_rho / np.size(img, 0)

    # Set any NaN values to zero
    img_for_iqm[np.isnan(img_for_iqm)] = 0

    # Find the index of the maximum response in the reconstruction
    max_loc = np.argmax(img_for_iqm)

    # Find the x/y-indices of the max response in the reconstruction
    max_y_pix, max_x_pix = np.unravel_index(max_loc, np.shape(img))

    # Convert this to the x/y-positions
    max_x_pos = (max_x_pix - np.size(img, 0) // 2) * pix_to_dist
    max_y_pos = -1 * (max_y_pix - np.size(img, 0) // 2) * pix_to_dist

    # Compute the localization error
    loc_err = np.sqrt((max_x_pos - tum_x)**2 + (max_y_pos - tum_y)**2)

    return loc_err
