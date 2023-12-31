"""
Tyson Reimer
University of Manitoba
November 08, 2018
"""

import numpy as np

from umbms.recon.breastmodels import get_breast, get_roi


###############################################################################

__VAC_SPEED = 3e8  # Define propagation speed in vacuum

###############################################################################


def estimate_speed(adi_rad, ant_rho, m_size=500):
    """Estimates the propagation speed of the signal in the scan

    Estimates the propagation speed of the microwave signal for
    *all* antenna positions in the scan. Estimates using the average
    propagation speed.

    Parameters
    ----------
    adi_rad : float
        The approximate radius of the breast, in [cm]
    ant_rho : float
        The radius of the antenna trajectory in the scan, as measured
        from the black line on the antenna holder, in [cm]
    m_size : int
        The number of pixels along one dimension used to model the 2D
        imaging chamber

    Returns
    -------
    speed : float
        The estimated propagation speed of the signal at all antenna
        positions, in m/s
    """

    # Model the breast as a homogeneous adipose circle, in air
    breast_model = get_breast(m_size=m_size,
                              adi_rad=adi_rad,
                              roi_rho=ant_rho)

    # Get the region of the scan area within the antenna trajectory
    roi = get_roi(ant_rho, m_size, ant_rho)

    # Estimate the speed
    speed = np.mean(__VAC_SPEED / np.sqrt(breast_model[roi]))

    return speed
