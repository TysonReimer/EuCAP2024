"""
Tyson Reimer
University of Manitoba
September 23, 2023

Includes contributions by Fatimah Eashour
"""

import os
import numpy as np
import matplotlib
matplotlib.use('agg')

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle, load_birrs_folder

from umbms.recon.extras import get_pix_ts, get_fd_phase_factor
from umbms.recon.algos import fd_das

from umbms.plot.imgs import plot_img

from umbms.hardware import apply_sys_cor
from umbms.hardware.antenna import apply_ant_t_delay, to_phase_center

from umbms.tdelay.propspeed import estimate_speed

###############################################################################

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/bed/")

__O_DIR = os.path.join(get_proj_path(), "output/eucap2024/bed/")
verify_path(__O_DIR)

__INI_F = 2e9  # Initial freq: 700 MHz
__FIN_F = 9e9  # Final freq: 8 GHz
__N_FS = 1001  # Number of frequencies

# Antenna polar rho coordinate, in [cm], after shifting to phase center
__ANT_RHO = to_phase_center(21.0)
__N_ANTS = 72  # Number of antennas
__ROI_RHO = 8  # ROI radius, in [cm]
# If in-air measurement...
__SPEED = 299792458  # Speed in [m/s]

# Elif in-cylinder phantom measurement...
# __SPEED = estimate_speed(adi_rad=5.5, ant_rho=__ANT_RHO)
__FS = np.linspace(__INI_F, __FIN_F, __N_FS)  # Scan frequencies

__M_SIZE = int(__ROI_RHO * 2 / 0.1)  # Image size

###############################################################################


def setup_err_cor(metadata):
    """Correct for positionning setup error

    Parameters
    ----------
    metadata : array_like
        Target x/y positions, format x1, y1, x2, y2, [cm]

    Returns
    -------
    cor_x1 : array_like
        Corrected x-positions of target 1, [cm]
    cor_y1 : array_like
        Corrected y-positions of target 1, [cm]
    cor_x2 : array_like
        Corrected x-positions of target 2, [cm]
    cor_y2 : array_like
        Corrected y-positions of target 2, [cm]
    """

    # Get the target x/y positions
    x1, y1 = metadata[:, 0], metadata[:, 1]
    x2, y2 = metadata[:, 2], metadata[:, 3]

    # Correct for systematic setup positioning errors
    cor_x1, cor_y1 = apply_sys_cor(x1, y1)
    cor_x2, cor_y2 = apply_sys_cor(x2, y2)

    return cor_x1, cor_y1, cor_x2, cor_y2


def recon_two_tar(speed=299792458):
    """Reconstruct the two-target scans

    Parameters
    ----------
    speed : float
        Propagation speed to use, in [m/s]
    """

    # Data directory
    d_dir = os.path.join(__D_DIR, "2-tar/")

    # Output directory
    o_dir = os.path.join(__O_DIR, "2-tar/")
    verify_path(o_dir)

    # Load the frequency domain data
    fd = load_birrs_folder(folder_path=os.path.join(d_dir),
                           id_str="exp",
                           ref_idxs=[0],
                           load_s21=False,
                           ref_to_use=0)

    # Load metadata: Format of x1, y1, x2, y2, default in [mm] so
    # divide by 10 to convert to [cm]
    md = np.genfromtxt(os.path.join(d_dir, "metadata.csv"),
                       delimiter=",",
                       dtype=float,
                       skip_header=2
                       )[:, 1:] / 10

    # Correct for setup positioning errors
    xs_1, ys_1, xs_2, ys_2 = setup_err_cor(metadata=md)

    # Get 1-way pixel propagation time delays
    pix_ts = get_pix_ts(ant_rho=__ANT_RHO,
                        m_size=__M_SIZE,
                        roi_rad=__ROI_RHO,
                        speed=speed,
                        n_ant_pos=__N_ANTS
                        )

    pix_ts = apply_ant_t_delay(pix_ts)  # Correct for antenna t-delay

    # Get the phase factor for efficient computation
    phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

    # Init arr of images
    das_imgs = np.zeros([np.size(fd, axis=0), __M_SIZE, __M_SIZE],
                        dtype=complex)

    for ii in range(np.size(fd, axis=0)):  # For each scan

        logger.info('Exp [%2d / %2d]...' % (ii + 1, np.size(fd, axis=0)))

        das_imgs[ii, :, :] = fd_das(fd_data=fd[ii, :, :],
                                    phase_fac=phase_fac,
                                    freqs=__FS,
                                    n_cores=10)

        # Plot recon
        plot_img(img=np.abs(das_imgs[ii, :, :]),
                 tar_xs=[xs_1[ii], xs_2[ii]],
                 tar_ys=[ys_1[ii], ys_2[ii]],
                 tar_rads=[0.1, 0.1],
                 roi_rho=__ROI_RHO,
                 save_fig=True,
                 save_str=os.path.join(o_dir, "exp%d_das.png" % ii))

    save_pickle(das_imgs, os.path.join(o_dir, "das.pickle"))


def recon_noise(speed=estimate_speed(5.55, __ANT_RHO)):
    """Reconstruct noise scans

    Parameters
    ----------
    speed : float
        Propagation speed to use in the scan, in [m/s]; note the
        default value assumes phantom radius of 5.55 cm
    """

    # Output dir
    o_dir = os.path.join(__O_DIR, "noise/")
    verify_path(o_dir)

    # Load the frequency domain data
    fd = load_birrs_folder(folder_path=os.path.join(__D_DIR, "noise/"),
                           id_str="exp",
                           ref_idxs=[0],
                           load_s21=False,
                           ref_to_use=0)

    # Get 1-way pixel propagation time delays
    pix_ts = get_pix_ts(ant_rho=__ANT_RHO,
                        m_size=__M_SIZE,
                        roi_rad=__ROI_RHO,
                        speed=speed,
                        n_ant_pos=__N_ANTS
                        )

    pix_ts = apply_ant_t_delay(pix_ts)  # Correct for antenna t-delay

    # Get the phase factor for efficient computation
    phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

    # Init arr of images
    das_imgs = np.zeros([np.size(fd, axis=0), __M_SIZE, __M_SIZE],
                        dtype=complex)

    for ii in range(np.size(fd, axis=0)):  # For each scan

        logger.info('Exp [%2d / %2d]...' % (ii + 1, np.size(fd, axis=0)))

        das_imgs[ii, :, :] = fd_das(fd_data=fd[ii, :, :],
                                    phase_fac=phase_fac,
                                    freqs=__FS,
                                    n_cores=10)

        # Plot recon
        plot_img(img=np.abs(das_imgs[ii, :, :]),
                 tar_xs=[],
                 tar_ys=[],
                 tar_rads=[],
                 roi_rho=__ROI_RHO,
                 save_fig=True,
                 save_str=os.path.join(o_dir, "exp%d_das.png" % ii))

    save_pickle(das_imgs, os.path.join(o_dir, "das.pickle"))


def recon_psf(speed=299792458):
    """Reconstruct 1-target scans

        Parameters
    ----------
    speed : float
        Propagation speed to use, in [m/s]
    """

    # Data dir
    d_dir = os.path.join(__D_DIR, "psf/")

    # Output dir
    o_dir = os.path.join(__O_DIR, "psf/")
    verify_path(o_dir)

    # Load the frequency domain data
    fd = load_pickle(os.path.join(d_dir, "s11.pickle"))

    # Load metadata
    md = load_pickle(os.path.join(d_dir, "md.pickle"))

    xs, ys = md[:, 0], md[:, 1]  # Get xs/ys separately

    xs, ys = apply_sys_cor(xs=xs, ys=ys)  # Apply setup error correction

    # Get 1-way pixel propagation time delays
    pix_ts = get_pix_ts(ant_rho=__ANT_RHO,
                        m_size=__M_SIZE,
                        roi_rad=__ROI_RHO,
                        speed=speed,
                        n_ant_pos=__N_ANTS
                        )

    pix_ts = apply_ant_t_delay(pix_ts)  # Correct for antenna t-delay

    # Get the phase factor for efficient computation
    phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

    # Init arr of images
    das_imgs = np.zeros([np.size(fd, axis=0), __M_SIZE, __M_SIZE],
                        dtype=complex)

    for ii in range(np.size(fd, axis=0)):  # For each scan

        logger.info('Exp [%2d / %2d]...' % (ii + 1, np.size(fd, axis=0)))

        das_imgs[ii, :, :] = fd_das(fd_data=fd[ii, :, :],
                                    phase_fac=phase_fac,
                                    freqs=__FS,
                                    n_cores=10)

        # Plot recon
        plot_img(img=np.abs(das_imgs[ii, :, :]),
                 tar_xs=[xs[ii]],
                 tar_ys=[ys[ii]],
                 tar_rads=[0.1],
                 roi_rho=__ROI_RHO,
                 save_fig=True,
                 save_str=os.path.join(o_dir, "exp%d_das.png" % ii))

    save_pickle(das_imgs, os.path.join(o_dir, "das.pickle"))


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Do reconstructions
    recon_psf()
    recon_two_tar()
    recon_noise()
