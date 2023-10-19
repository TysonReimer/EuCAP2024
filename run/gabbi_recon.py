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

from umbms.loadsave import save_pickle

from umbms.recon.extras import get_fd_phase_factor, get_pix_ts
from umbms.recon.algos import fd_das

from umbms.plot.imgs import plot_img

from umbms.tdelay.propspeed import estimate_speed

###############################################################################

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/gabbi/")

__O_DIR = os.path.join(get_proj_path(), "output/eucap2024/gabbi/")
verify_path(__O_DIR)

__INI_F = 600e6  # Initial freq: 700 MHz
__FIN_F = 4.4e9  # Final freq: 8 GHz
__N_FS = 381  # Number of frequencies
# __ANT_T = 9.1e-10  # Antenna 1-way t-delay, [s]
__ANT_RHO = 10  # Antenna polar rho coordinate, in [cm]
__INI_ANT_ANG = -90.0  # Initial antenna angle, in [deg]
__N_ANTS = 24  # Number of antennas
__ROI_RHO = 8  # ROI radius, in [cm]
__SPEED = 299792458  # Speed in [m/s]
__FS = np.linspace(__INI_F, __FIN_F, __N_FS)  # Scan frequencies

__M_SIZE = int(__ROI_RHO * 2 / 0.1)  # Image size

__TWO_TAR_XS = np.array([
    3.314,
    -4.871,
    -9.239,
    -5.985,
    1.918,
    10.407,
    16.602,
    19.142,
    17.846,
    13.265,
    6.305,
    -2.034,
    -10.809,
    -19.22,
    -26.644,
]) / 10 # Convert from [mm] to [cm]

__TWO_TAR_YS = np.array([
    4.264,
    5.57,
    -1.731,
    -9.703,
    -13.262,
    -11.351,
    -5.208,
    3.155,
    11.81,
    19.277,
    24.605,
    27.324,
    27.341,
    24.83,
    20.141,
]) / 10  # Convert from [mm] to [cm]

###############################################################################


def recon_single_rod(speed=299792458):
    """Reconstruct 1-target scans

    Parameters
    ----------
    speed : float
        Propagation speed to use, in [m/s]
    """

    d_dir = os.path.join(__D_DIR, "psf/")  # Dir for single rod
    o_dir = os.path.join(__O_DIR, "psf/")
    verify_path(o_dir)

    all_fs = os.listdir(d_dir)  # All files to load

    tar_xs = np.zeros([len(all_fs), ])  # Target x-positions
    tar_ys = np.zeros([len(all_fs), ])  # Target y-positions
    fd = np.zeros([len(all_fs), __N_FS, __N_ANTS], dtype=complex)

    # Get 1-way pixel propagation time delays
    pix_ts = get_pix_ts(ant_rho=__ANT_RHO,
                        m_size=__M_SIZE,
                        roi_rad=__ROI_RHO,
                        speed=speed,
                        n_ant_pos=__N_ANTS,
                        ini_ant_ang=__INI_ANT_ANG)

    # Get the phase factor for efficient computation
    phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

    # Init arr of images
    das_imgs = np.zeros([len(all_fs), __M_SIZE, __M_SIZE], dtype=complex)

    for ii in range(len(all_fs)):  # For each file

        logger.info('Exp [%3d / %3d]...' % (ii + 1, len(all_fs)))

        # Load fd data
        fd[ii, :, :] = np.load(os.path.join(d_dir, all_fs[ii]))

        # Target x-position, in [cm]
        tar_xs[ii] = int(all_fs[ii].split('x')[1].split('mm')[0]) / 10

        # Target y-position, in [cm]
        tar_ys[ii] = int(all_fs[ii].split('y')[1].split('mm')[0]) / 10

        # Reconstruct DAS image
        das_imgs[ii, :, :] = fd_das(fd_data=fd[ii, :, :],
                                    phase_fac=phase_fac,
                                    freqs=__FS,
                                    n_cores=10)

        # Plot recon
        plot_img(img=np.abs(das_imgs[ii, :, :]),
                 tar_xs=[tar_xs[ii]],
                 tar_ys=[tar_ys[ii]],
                 tar_rads=[0.1],
                 roi_rho=__ROI_RHO,
                 save_fig=True,
                 save_str=os.path.join(o_dir, "exp%d_das.png" % ii))

    # Save to pickle
    save_pickle(das_imgs, os.path.join(o_dir, "das.pickle"))
    save_pickle(tar_xs, os.path.join(o_dir, "tar_xs.pickle"))
    save_pickle(tar_ys, os.path.join(o_dir, "tar_ys.pickle"))


def recon_two_tar(speed=299792458):
    """Reconstruct the two-target scans

    Parameters
    ----------
    speed : float
        Propagation speed to use, in [m/s]
    """

    d_dir = os.path.join(__D_DIR, "2-tar/")  # Dir for 2 rod data
    o_dir = os.path.join(__O_DIR, "two-tar/")  # Dir for 2 rod output
    verify_path(o_dir)

    all_fs = os.listdir(d_dir)  # All files to load

    fd = np.zeros([len(all_fs), __N_FS, __N_ANTS], dtype=complex)

    # Get 1-way pixel propagation time delays
    pix_ts = get_pix_ts(ant_rho=__ANT_RHO,
                        m_size=__M_SIZE,
                        roi_rad=__ROI_RHO,
                        speed=speed,
                        n_ant_pos=__N_ANTS,
                        ini_ant_ang=__INI_ANT_ANG)

    # Get the phase factor for efficient computation
    phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

    # Init arr of images
    das_imgs = np.zeros([len(all_fs), __M_SIZE, __M_SIZE], dtype=complex)
    tar_xs = np.zeros(len(all_fs))
    tar_ys = np.zeros(len(all_fs))

    for ii in range(len(all_fs)):  # For each file

        logger.info('Exp [%3d / %3d]...' % (ii + 1, len(all_fs)))

        # Target position index
        pos_idx = int(all_fs[ii].split('pos')[1].split('.')[0])

        tar_x = __TWO_TAR_XS[pos_idx // 2 - 1]
        tar_y = __TWO_TAR_YS[pos_idx // 2 - 1]

        tar_xs[ii] = tar_x
        tar_ys[ii] = tar_y

        # Load fd data
        fd[ii, :, :] = np.load(os.path.join(d_dir, all_fs[ii]))

        # Reconstruct DAS image
        das_imgs[ii, :, :] = fd_das(fd_data=fd[ii, :, :],
                                    phase_fac=phase_fac,
                                    freqs=__FS,
                                    n_cores=10)

        # Plot recon
        plot_img(img=np.abs(das_imgs[ii, :, :]),
                 tar_xs=[0, tar_x],
                 tar_ys=[0, tar_y],
                 tar_rads=[0.1, 0.1],
                 roi_rho=__ROI_RHO,
                 save_fig=True,
                 save_str=os.path.join(o_dir, "exp%d_das.png" % ii))

    # Save to pickle
    save_pickle(das_imgs, os.path.join(o_dir, "das.pickle"))
    save_pickle(tar_xs, os.path.join(o_dir, "tar_xs.pickle"))
    save_pickle(tar_ys, os.path.join(o_dir, "tar_ys.pickle"))


def recon_noise(speed=estimate_speed(5.55, __ANT_RHO)):
    """Reconstruct noise scans

    Parameters
    ----------
    speed : float
        Propagation speed to use in the scan, in [m/s]; note the
        default value assumes phantom radius of 5.55 cm
    """

    d_dir = os.path.join(__D_DIR, "noise/")  # Dir for single rod
    o_dir = os.path.join(__O_DIR, "noise/")
    verify_path(o_dir)

    adi_str = "AdiposeCylinder_"
    ref_str = "OpenChamber"

    all_fs = os.listdir(d_dir)  # All files to load

    fd = np.zeros([len(all_fs) - 2, __N_FS, __N_ANTS], dtype=complex)
    ref_fd = np.load(os.path.join(d_dir, "%s%d.npy" % (ref_str, 1)))

    # Get 1-way pixel propagation time delays
    pix_ts = get_pix_ts(ant_rho=__ANT_RHO,
                        m_size=__M_SIZE,
                        roi_rad=__ROI_RHO,
                        speed=speed,
                        n_ant_pos=__N_ANTS,
                        ini_ant_ang=__INI_ANT_ANG)

    # Get the phase factor for efficient computation
    phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

    # Init arr of images
    das_imgs = np.zeros([len(all_fs) - 2, __M_SIZE, __M_SIZE], dtype=complex)

    for ii in range(len(all_fs) - 2):  # For each file

        logger.info('Exp [%3d / %3d]...' % (ii + 1, len(all_fs) - 2))

        # Load fd data
        fd[ii, :, :] = (
            np.load(os.path.join(d_dir, "%s%d.npy" % (adi_str, ii + 1)))
            - ref_fd
        )

        # Reconstruct DAS image
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

    # Save to pickle
    save_pickle(das_imgs, os.path.join(o_dir, "das.pickle"))


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Reconstruct images
    recon_single_rod()
    recon_two_tar()
    recon_noise()



