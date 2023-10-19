"""
Tyson Reimer
University of Manitoba
November 7, 2018
"""

import numpy as np

###############################################################################

__GHz = 1e9  # Conversion factor from Hz to GHz

__VACUUM_SPEED = 3e8  # Speed of light in a vacuum

###############################################################################


def get_pix_xys(m_size, roi_rho):
    """Finds the x/y position of each pixel in the image-space

    Return arrays that contain the x-distances and y-distances of every
    pixel in the model.

    Parameters
    ----------
    m_size : int
        The number of pixels along one dimension of the model
    roi_rho : float
        The radius of the region of interest, in [cm]

    Returns
    -------
    x_dists : array_like
        A 2D arr. Each element in the arr contains the x-position of
        that pixel in the model, in [cm]
    y_dists : array_like
        A 2D arr. Each element in the arr contains the y-position of
        that pixel in the model, in [cm]
    """

    # Define the x/y points on each axis
    xs = np.linspace(-roi_rho, roi_rho, m_size)
    ys = np.linspace(roi_rho, -roi_rho, m_size)

    # Cast these to 2D for ease later
    x_dists, y_dists = np.meshgrid(xs, ys)

    return x_dists, y_dists


def get_pixdist_ratio(m_size, ant_rho):
    """Get the ratio between pixel number and physical distance

    Returns the pixel-to-distance ratio (physical distance, in meters)

    Parameters
    ----------
    m_size : int
        The number of pixels used along one-dimension for the model
        (the model is assumed to be square)
    ant_rho : float
        The radius of the antenna trajectory during the scan, in meters

    Returns
    -------
    pix_to_dist_ratio : float
        The number of pixels per physical meter
    """

    # Get the ratio between pixel and physical length
    pix_to_dist_ratio = m_size / (2 * ant_rho)

    return pix_to_dist_ratio


def get_ant_scan_xys(ant_rho, n_ants, ini_a_ang=-130.0):
    """Returns the x,y positions of each antenna position in the scan

    Returns two vectors, containing the x- and y-positions in meters of
    the antenna during a scan.

    Parameters
    ----------
    ant_rho : float
        The radius of the trajectory of the antenna during the scan,
        in meters
    n_ants : int
        The number of antenna positions used in the scan
    ini_a_ang : float
        The initial angle offset (in deg) of the antenna from the
        negative x-axis

    Returns
    -------
    ant_xs : array_like
        The x-positions in meters of each antenna position used in the
        scan
    ant_ys : array_like
        The y-positions in meters of each antenna position used in the
        scan
    """

    # Find the polar angles of each of the antenna positions used in the
    # scan; Antenna sweeps from 0deg to 355deg
    ant_angs = (np.linspace(0, np.deg2rad(355), n_ants)
                + np.deg2rad(ini_a_ang))
    ant_angs = np.flip(ant_angs)  # Antenna moves clockwise

    ant_xs = np.cos(ant_angs) * ant_rho  # Find the x-positions

    ant_ys = np.sin(ant_angs) * ant_rho  # Find the y-positions

    return ant_xs, ant_ys


def get_pix_ds(ant_rho, m_size, roi_rad, n_ant_pos=72,
               ini_ant_ang=-136.0):
    """Get one-way propagation distances to each pixel

    Parameters
    ----------
    ant_rho : float
        Antenna rho used in scan, in [cm]
    m_size : int
        Size of image-space along one dimension
    roi_rad : float
        Radius of the region of interest that will define the spatial
        extent of the image-space, in [cm]
    n_ant_pos : int
        Number of antenna positions used in the scan
    ini_ant_ang : float
        Polar angle of initial antenna position

    Returns
    -------
    p_ds : array_like, MxNxN
        One-way propagation distances for all pixels in the
        NxN image-space. M is the number of antenna positions.
        In units of [cm]
    """

    # Get antenna x/y positions during scan
    ant_xs, ant_ys = get_ant_scan_xys(ant_rho=ant_rho, n_ants=n_ant_pos,
                                      ini_a_ang=ini_ant_ang)

    # Create arrays of pixel x/y positions
    pix_xs, pix_ys = get_pix_xys(m_size=m_size, roi_rho=roi_rad)

    # Init array for storing pixel time-delays
    p_ds = np.zeros([n_ant_pos, m_size, m_size])

    for a_pos in range(n_ant_pos):  # For each antenna position

        # Find x/y position differences of each pixel from antenna
        x_diffs = pix_xs - ant_xs[a_pos]
        y_diffs = pix_ys - ant_ys[a_pos]

        # Calculate one-way time-delay of propagation from antenna to
        # each pixel
        p_ds[a_pos, :, :] = np.sqrt(x_diffs ** 2 + y_diffs ** 2)

    return p_ds


def get_pix_angs(ant_rho, m_size, roi_rho, n_ant_pos=72, ini_ant_ang=-136.0):
    """Get angle-off-antenna-boresight for each pixel

    Parameters
    ----------
    ant_rho : float
        Antenna rho used in scan, in [cm]
    m_size : int
        Size of image-space along one dimension
    roi_rho : float
        Radius of the region of interest that will define the spatial
        extent of the image-space, in [cm]
    n_ant_pos : int
        Number of antenna positions used in the scan
    ini_ant_ang : float
        Polar angle of initial antenna position

    Returns
    -------
    p_angs : array_like, MxNxN
        Angle of each pixel off of the boresight of the antenna,
        in [deg]
    """

    # Get antenna x/y positions during scan
    ant_xs, ant_ys = get_ant_scan_xys(ant_rho=ant_rho, n_ants=n_ant_pos,
                                      ini_a_ang=ini_ant_ang)

    # Convert antenna cartesian to polar coordinates
    ant_rhos = np.sqrt(ant_xs**2 + ant_ys**2)
    ant_phis = np.arctan2(ant_ys, ant_xs)

    # Create arrays of pixel x/y positions
    pix_xs, pix_ys = get_pix_xys(m_size=m_size, roi_rho=roi_rho)

    # Convert pixel cartesian to polar coordinates
    pix_rhos = np.sqrt(pix_xs**2 + pix_ys**2)
    pix_phis = np.arctan2(pix_ys, pix_xs)

    # Calculate pixel angle-off-boresight of antenna using cosine law
    p_angs = np.arccos(
        (ant_rhos[:, None, None] - pix_rhos[None, :, :]
         * np.cos(ant_phis[:, None, None] - pix_phis[None, :, :]))
        / np.sqrt(pix_rhos[None, :, :] ** 2 + ant_rhos[:, None, None] ** 2
                  - 2 * pix_rhos[None, :, :] * ant_rhos[:, None, None]
                  * np.cos(ant_phis[:, None, None] - pix_phis[None, :, :]))
    )

    # Convert from [rad] to [deg]
    p_angs = np.rad2deg(p_angs)

    return p_angs


def get_pix_ts(ant_rho, m_size, roi_rad, speed, n_ant_pos=72,
               ini_ant_ang=-136.0):
    """Get one-way pixel response times

    Parameters
    ----------
    ant_rho : float
        Antenna rho used in scan, in [cm]
    m_size : int
        Size of image-space along one dimension
    roi_rad : float
        Radius of the region of interest that will define the spatial
        extent of the image-space, in [cm]
    speed : float
        The estimated propagation speed of the signal, in [m/s]
    n_ant_pos : int
        Number of antenna positions used in the scan
    ini_ant_ang : float
        Polar angle of initial antenna position

    Returns
    -------
    p_ts : array_like, MxNxN
        One-way response times of all pixels in the NxN image-space.
        M is the number of antenna positions.
    """

    # Calculate one-way response times in units of [s], by
    # converting speed to units of [cm / s]
    p_ts = (get_pix_ds(ant_rho=ant_rho, m_size=m_size, roi_rad=roi_rad,
                       n_ant_pos=n_ant_pos, ini_ant_ang=ini_ant_ang)
            / (speed * 100))

    return p_ts


def get_pix_ts_ms(tx_xs, tx_ys, rx_xs, rx_ys, m_size, roi_rad, speed,
                  n_ant_pairs):
    """Get two-way pixel response times for a multistatic system

    Fatimah Eashour contributed to this function

    Parameters
    ----------
    tx_xs : array_like
        x-position of each transmit position, [cm]
    tx_ys : array_like
        y-position of each transmit position, [cm]
    rx_xs : array_like
        x-position of each receive position, [cm]
    rx_ys : array_like
        y-position of each receive position, [cm]
    m_size : int
        Size of image-space along one dimension
    roi_rad : float
        Radius of the region of interest that will define the spatial
        extent of the image-space, in [cm]
    speed : float
        The estimated propagation speed of the signal, in [m/s]
    n_ant_pairs : int
        Number of antenna pairs used in the scan
    ini_ant_ang : float
        Polar angle of initial antenna position

    Returns
    -------
    ts : array_like
        Two-way pixel response times for a multistatic system
    """

    # Create arrays of pixel x/y positions
    pix_xs, pix_ys = get_pix_xys(m_size=m_size, roi_rho=roi_rad)

    # Init array for storing pixel time-delays
    ts = np.zeros([n_ant_pairs, m_size, m_size])

    for a_pos in range(n_ant_pairs):  # For each antenna position

        # Find x/y position differences of each pixel from antenna
        x_diffs_tx = pix_xs - tx_xs[a_pos]
        y_diffs_tx = pix_ys - tx_ys[a_pos]
        x_diffs_rx = pix_xs - rx_xs[a_pos]
        y_diffs_rx = pix_ys - rx_ys[a_pos]

        # Calculate one-way time-delay of propagation from antenna to
        # each pixel
        ts[a_pos, :, :] = (
                (np.sqrt(x_diffs_tx**2 + y_diffs_tx**2) +
                 np.sqrt(x_diffs_rx**2 + y_diffs_rx**2))
                / (speed * 100))  # Convert speed to [cm / s]

    return ts



def get_fd_phase_factor(pix_ts):
    """Get phase factor required for multiple scattering computation

    Parameters
    ----------
    pix_ts : array_like, MxNxN
        One-way response times of all pixels in the NxN image-space.
        M is the number of antenna positions.

    Returns
    -------
    phase_fac : array_like, MxNxN
        Phase factor for multiple scattering. N is m_size and M is
        n_ant_pos
    """

    # Calculate the phase-factor for efficient computation of multiple
    # scattering
    phase_fac = np.exp(-1j * 2 * np.pi * pix_ts)

    return phase_fac
