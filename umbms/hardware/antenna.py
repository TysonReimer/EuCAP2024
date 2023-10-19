"""
Tyson Reimer
University of Manitoba
November 07, 2018
"""

###############################################################################


def apply_ant_t_delay(pix_ts):
    """Add the antenna time delay to the 1-way pixel time delays

    Parameters
    ----------
    pix_ts : array_like, NxMxL
        Array of the 1-way time-of-response for each pixel, N is
        number of antenna positions, MxL is size of 2D image, units
        of [s]

    Returns
    -------
    cor_ts : array_like, NxMxL
        The 1-way response times for each pixel after accounting
        for the 1-way antenna time delay, units of [s]
    """

    # Add the 1-way antenna time delay, in units of [s]
    # (Value added should be 0.19 ns, based on analysis from May 29,
    # 2023)
    cor_ts = pix_ts + 0.19e-9

    return cor_ts


def to_phase_center(meas_rho):
    """Shift the measured rho of the antenna to its phase center

    Parameters
    ----------
    meas_rho : float
        The measured rho of the antenna; measured from the front edge
        of the antenna stand, in [cm]

    Returns
    -------
    cor_rho : float
        The corrected rho of the antenna, i.e., the rho corresponding
        to the phase center of the antenna, in [cm]
    """

    # Add the length corresponding to the distance from the front edge
    # of the antenna stand to the phase center of the antenna, in [cm]
    # (Value added should be 2.4 cm, based on analysis from May 29,
    # 2023)
    cor_rho = meas_rho + 2.4

    return cor_rho
