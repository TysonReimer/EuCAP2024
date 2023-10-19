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

from umbms.loadsave import load_pickle, save_pickle

from umbms.recon.extras import get_pix_ts_ms, get_fd_phase_factor
from umbms.recon.algos import fd_das

from umbms.plot.imgs import plot_img

from umbms.tdelay.propspeed import estimate_speed

###############################################################################

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/fatimah-sys/")

__O_DIR = os.path.join(get_proj_path(), "output/eucap2024/fatimah-sys/")
verify_path(__O_DIR)

__INI_F = 700e6  # Initial freq: 700 MHz
__FIN_F = 8e9  # Final freq: 8 GHz
__N_FS = 1001  # Number of frequencies
__ANT_T = 9.1e-10  # Antenna 1-way t-delay, [s]
__ANT_RHO = 19.1  # Antenna polar rho coordinate, in [cm]
__INI_ANT_ANG = 7.5  # Initial antenna angle, in [deg]
__N_ANTS = 24  # Number of antennas
__ROI_RHO = 8  # ROI radius, in [cm]

__FS = np.linspace(__INI_F, __FIN_F, __N_FS)  # Scan frequencies

__M_SIZE = int(__ROI_RHO * 2 / 0.1)  # Image size

###############################################################################


def recon_psf(speed=299792458):
    """Reconstruct 1-target scans

        Parameters
    ----------
    speed : float
        Propagation speed to use, in [m/s]
    """

    # Load frequency domain data and metadata
    fd = load_pickle(os.path.join(__D_DIR, "psf_fd.pickle"))
    md = load_pickle(os.path.join(__D_DIR, "psf_md.pickle"))

    top_o_dir = os.path.join(__O_DIR, "psf/")
    verify_path(top_o_dir)

    # Load the calibration factors
    cal_factors = load_pickle(os.path.join(__D_DIR, "cal_factors.pickle"))

    loss_cal = cal_factors[0]  # Calibration factors for losses
    t_delay_cal = cal_factors[1]  # Calibration factors for t-delays

    n_scans = len(fd)  # Number scans performed

    # scan_types = ["multi", "mono"]  # Type of 'scan' data to reconstruct
    scan_types = ["mono"]

    for scan_type in scan_types:

        logger.info('Scan type:\t%s' % scan_type)

        if scan_type == 'multi':  # If multistatic scan
            n_proj = __N_ANTS ** 2  # Number of 'projections'
        elif scan_type == 'multi_trans':
            n_proj = __N_ANTS * (__N_ANTS - 1)
        else:  # If monostatic scan
            n_proj = __N_ANTS

        das_imgs = np.zeros([n_scans, __M_SIZE, __M_SIZE], dtype=complex)

        o_dir = os.path.join(top_o_dir, "%s/" % scan_type)
        verify_path(o_dir)

        for ii in range(n_scans):  # For each scan

            logger.info('Scan [%3d / %d]...' % (ii + 1, n_scans))

            if md[ii]['expt'] == 5:  # Fifth expt always empty ref

                continue

            for jj in range(n_scans):  # For each scan again...

                # If the two scans are from the same session *and*
                # the second scan is also of expt 5...
                if (md[ii]['session'] == md[jj]['session']
                        and md[jj]['expt'] == 5):
                    ref_data = fd[jj]
                    break

            for param in fd[ii]:  # For each s-parameter, do ref subtract
                fd[ii][param] = ((fd[ii][param] - ref_data[param])
                                 / loss_cal[param])

            # Init lists for storing everything here
            fd_here = []
            ant_xs_tx = []
            ant_ys_tx = []
            ant_xs_rx = []
            ant_ys_rx = []
            cal_ts = []

            for param in fd[ii]:  # For each s-param

                rx = param[1:param.index('_')]  # Index of rx port
                tx = param[param.index('_') + 1:]  # Index of tx port

                # If the tx and rx are different and scan_type is multi
                # trans
                # OR if
                if ((tx != rx and scan_type == 'multi_trans')
                    or (tx == rx and scan_type == 'mono')
                    or (scan_type == 'multi')):

                    fd_here.append(fd[ii][param])

                    # Tx antenna position
                    ant_ang_tx = int(tx) * 360 / __N_ANTS + __INI_ANT_ANG
                    ant_xs_tx.append(
                        __ANT_RHO * np.cos(np.deg2rad(ant_ang_tx)))
                    ant_ys_tx.append(
                        __ANT_RHO * np.sin(np.deg2rad(ant_ang_tx)))

                    # Rx antenna position
                    ant_ang_rx = int(rx) * 360 / __N_ANTS + __INI_ANT_ANG
                    ant_xs_rx.append(
                        __ANT_RHO * np.cos(np.deg2rad(ant_ang_rx)))
                    ant_ys_rx.append(
                        __ANT_RHO * np.sin(np.deg2rad(ant_ang_rx)))

                    cal_ts.append(
                        t_delay_cal[param] * np.ones([__M_SIZE, __M_SIZE]))

            # Get the two-way pixel response times
            pix_ts = get_pix_ts_ms(tx_xs=ant_xs_tx, tx_ys=ant_ys_tx,
                                   rx_xs=ant_xs_rx, rx_ys=ant_ys_rx,
                                   speed=speed,
                                   roi_rad=__ROI_RHO,
                                   m_size=__M_SIZE,
                                   n_ant_pairs=n_proj)

            # Compensate for the 1-way antenna t-delay
            pix_ts += 2 * __ANT_T

            # Add calibration response time
            pix_ts += np.array(cal_ts)

            # Get phase factor for efficient computation
            # but use 1-way time-delays instead of 2-way
            phase_fac = get_fd_phase_factor(pix_ts / 2)

            logger.info('\tStarting DAS...')

            # Make the DAS image
            das_imgs[ii, :, :] = fd_das(fd_data=np.array(fd_here).T,
                                        phase_fac=phase_fac,
                                        freqs=__FS,
                                        n_cores=10)

            plot_img(img=np.abs(das_imgs[ii, :, :]),
                     tar_xs=[md[ii]['x ']],
                     tar_ys=[md[ii]['y ']],
                     tar_rads=[0.1],
                     roi_rho=__ROI_RHO,
                     save_fig=True,
                     save_str=os.path.join(o_dir, "exp%d_das.png" % ii))

        save_pickle(das_imgs, os.path.join(o_dir, "das_%s.pickle" % scan_type))


def recon_2tar(speed=299792458):
    """Reconstruct the two-target scans

    Parameters
    ----------
    speed : float
        Propagation speed to use, in [m/s]
    """

    # Load frequency domain data and metadata
    fd = load_pickle(os.path.join(__D_DIR, "two_tar_fd.pickle"))
    md = load_pickle(os.path.join(__D_DIR, "two_tar_md.pickle"))

    top_o_dir = os.path.join(__O_DIR, "two-tar/")
    verify_path(top_o_dir)

    # Load the calibration factors
    cal_factors = load_pickle(os.path.join(__D_DIR, "cal_factors.pickle"))

    loss_cal = cal_factors[0]  # Calibration factors for losses
    t_delay_cal = cal_factors[1]  # Calibration factors for t-delays

    n_scans = len(fd)  # Number scans performed

    # scan_types = ["multi", "mono"]  # Type of 'scan' data to reconstruct
    scan_types = ["mono"]

    for scan_type in scan_types:

        logger.info('Scan type:\t%s' % scan_type)

        if scan_type == 'multi':  # If multistatic scan
            n_proj = __N_ANTS ** 2  # Number of 'projections'
        elif scan_type == 'multi_trans':
            n_proj = __N_ANTS * (__N_ANTS - 1)
        else:  # If monostatic scan
            n_proj = __N_ANTS

        das_imgs = np.zeros([n_scans, __M_SIZE, __M_SIZE], dtype=complex)

        o_dir = os.path.join(top_o_dir, "%s/" % scan_type)
        verify_path(o_dir)

        for ii in range(n_scans):  # For each scan

            logger.info('Scan [%3d / %d]...' % (ii + 1, n_scans))

            if md[ii]['expt'] == 5:  # Fifth expt always empty ref

                continue

            for jj in range(n_scans):  # For each scan again...

                # If the two scans are from the same session *and*
                # the second scan is also of expt 5...
                if (md[ii]['session'] == md[jj]['session']
                        and md[jj]['expt'] == 5):
                    ref_data = fd[jj]
                    break

            for param in fd[ii]:  # For each s-parameter, do ref subtract
                fd[ii][param] = ((fd[ii][param] - ref_data[param])
                                 / loss_cal[param])

            # Init lists for storing everything here
            fd_here = []
            ant_xs_tx = []
            ant_ys_tx = []
            ant_xs_rx = []
            ant_ys_rx = []
            cal_ts = []

            for param in fd[ii]:  # For each s-param

                rx = param[1:param.index('_')]  # Index of rx port
                tx = param[param.index('_') + 1:]  # Index of tx port

                # If the tx and rx are different and scan_type is multi
                # trans
                # OR if
                if ((tx != rx and scan_type == 'multi_trans')
                    or (tx == rx and scan_type == 'mono')
                    or (scan_type == 'multi')):

                    fd_here.append(fd[ii][param])

                    # Tx antenna position
                    ant_ang_tx = int(tx) * 360 / __N_ANTS + __INI_ANT_ANG
                    ant_xs_tx.append(
                        __ANT_RHO * np.cos(np.deg2rad(ant_ang_tx)))
                    ant_ys_tx.append(
                        __ANT_RHO * np.sin(np.deg2rad(ant_ang_tx)))

                    # Rx antenna position
                    ant_ang_rx = int(rx) * 360 / __N_ANTS + __INI_ANT_ANG
                    ant_xs_rx.append(
                        __ANT_RHO * np.cos(np.deg2rad(ant_ang_rx)))
                    ant_ys_rx.append(
                        __ANT_RHO * np.sin(np.deg2rad(ant_ang_rx)))

                    cal_ts.append(
                        t_delay_cal[param] * np.ones([__M_SIZE, __M_SIZE]))

            # Get the two-way pixel response times
            pix_ts = get_pix_ts_ms(tx_xs=ant_xs_tx, tx_ys=ant_ys_tx,
                                   rx_xs=ant_xs_rx, rx_ys=ant_ys_rx,
                                   speed=speed,
                                   roi_rad=__ROI_RHO,
                                   m_size=__M_SIZE,
                                   n_ant_pairs=n_proj)

            # Compensate for the 1-way antenna t-delay
            pix_ts += 2 * __ANT_T

            # Add calibration response time
            pix_ts += np.array(cal_ts)

            # Get phase factor for efficient computation
            # but use 1-way time-delays instead of 2-way
            phase_fac = get_fd_phase_factor(pix_ts / 2)

            logger.info('\tStarting DAS...')

            # Make the DAS image
            das_imgs[ii, :, :] = fd_das(fd_data=np.array(fd_here).T,
                                        phase_fac=phase_fac,
                                        freqs=__FS,
                                        n_cores=10)

            plot_img(img=np.abs(das_imgs[ii, :, :]),
                     tar_xs=[md[ii]['x _1'] / 10, md[ii]['x_2'] / 10],
                     tar_ys=[md[ii]['y_1 '] / 10, md[ii]['y_2'] / 10],
                     tar_rads=[0.1, 0.1],
                     roi_rho=__ROI_RHO,
                     save_fig=True,
                     save_str=os.path.join(o_dir, "exp%d_das.png" % ii))

        save_pickle(das_imgs, os.path.join(o_dir, "das_%s.pickle" % scan_type))


def recon_noise(speed=estimate_speed(5.55, __ANT_RHO)):
    """Reconstruct noise scans

    Parameters
    ----------
    speed : float
        Propagation speed to use in the scan, in [m/s]; note the
        default value assumes phantom radius of 5.55 cm
    """

    # Load frequency domain data and metadata
    fd = load_pickle(os.path.join(__D_DIR, "noise_fd.pickle"))
    md = load_pickle(os.path.join(__D_DIR, "noise_md.pickle"))

    top_o_dir = os.path.join(__O_DIR, "noise/")
    verify_path(top_o_dir)

    # Load the calibration factors
    cal_factors = load_pickle(os.path.join(__D_DIR, "cal_factors.pickle"))

    loss_cal = cal_factors[0]  # Calibration factors for losses
    t_delay_cal = cal_factors[1]  # Calibration factors for t-delays

    n_scans = len(fd)  # Number scans performed

    scan_types = ["multi", "mono"]  # Type of 'scan' data to reconstruct

    for scan_type in scan_types:

        logger.info('Scan type:\t%s' % scan_type)

        if scan_type == 'multi':  # If multistatic scan
            n_proj = __N_ANTS ** 2  # Number of 'projections'
        elif scan_type == 'multi_trans':
            n_proj = __N_ANTS * (__N_ANTS - 1)
        else:  # If monostatic scan
            n_proj = __N_ANTS

        das_imgs = np.zeros([n_scans, __M_SIZE, __M_SIZE], dtype=complex)

        o_dir = os.path.join(top_o_dir, "%s/" % scan_type)
        verify_path(o_dir)

        for ii in range(n_scans):  # For each scan

            logger.info('Scan [%3d / %d]...' % (ii + 1, n_scans))

            if md[ii]['expt'] == 5:  # Fifth expt always empty ref

                continue

            for jj in range(n_scans):  # For each scan again...

                # If the two scans are from the same session *and*
                # the second scan is also of expt 5...
                if (md[ii]['session'] == md[jj]['session']
                        and md[jj]['expt'] == 5):
                    ref_data = fd[jj]
                    break

            for param in fd[ii]:  # For each s-parameter, do ref subtract
                fd[ii][param] = ((fd[ii][param] - ref_data[param])
                                 / loss_cal[param])

            # Init lists for storing everything here
            fd_here = []
            ant_xs_tx = []
            ant_ys_tx = []
            ant_xs_rx = []
            ant_ys_rx = []
            cal_ts = []

            for param in fd[ii]:  # For each s-param

                rx = param[1:param.index('_')]  # Index of rx port
                tx = param[param.index('_') + 1:]  # Index of tx port

                # If the tx and rx are different and scan_type is multi
                # trans
                # OR if
                if ((tx != rx and scan_type == 'multi_trans')
                    or (tx == rx and scan_type == 'mono')
                    or (scan_type == 'multi')):

                    fd_here.append(fd[ii][param])

                    # Tx antenna position
                    ant_ang_tx = int(tx) * 360 / __N_ANTS + __INI_ANT_ANG
                    ant_xs_tx.append(
                        __ANT_RHO * np.cos(np.deg2rad(ant_ang_tx)))
                    ant_ys_tx.append(
                        __ANT_RHO * np.sin(np.deg2rad(ant_ang_tx)))

                    # Rx antenna position
                    ant_ang_rx = int(rx) * 360 / __N_ANTS + __INI_ANT_ANG
                    ant_xs_rx.append(
                        __ANT_RHO * np.cos(np.deg2rad(ant_ang_rx)))
                    ant_ys_rx.append(
                        __ANT_RHO * np.sin(np.deg2rad(ant_ang_rx)))

                    cal_ts.append(
                        t_delay_cal[param] * np.ones([__M_SIZE, __M_SIZE]))

            # Get the two-way pixel response times
            pix_ts = get_pix_ts_ms(tx_xs=ant_xs_tx, tx_ys=ant_ys_tx,
                                   rx_xs=ant_xs_rx, rx_ys=ant_ys_rx,
                                   speed=speed,
                                   roi_rad=__ROI_RHO,
                                   m_size=__M_SIZE,
                                   n_ant_pairs=n_proj)

            # Compensate for the 1-way antenna t-delay
            pix_ts += 2 * __ANT_T

            # Add calibration response time
            pix_ts += np.array(cal_ts)

            # Get phase factor for efficient computation
            # but use 1-way time-delays instead of 2-way
            phase_fac = get_fd_phase_factor(pix_ts / 2)

            logger.info('\tStarting DAS...')

            print(np.shape(np.array(fd_here).T))
            # Make the DAS image
            das_imgs[ii, :, :] = fd_das(fd_data=np.array(fd_here).T,
                                        phase_fac=phase_fac,
                                        freqs=__FS,
                                        n_cores=10)

            plot_img(img=np.abs(das_imgs[ii, :, :]),
                     tar_xs=[],
                     tar_ys=[],
                     tar_rads=[],
                     roi_rho=__ROI_RHO,
                     save_fig=True,
                     save_str=os.path.join(o_dir, "exp%d_das.png" % ii))

        save_pickle(das_imgs, os.path.join(o_dir, "das_%s.pickle" % scan_type))


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Do reconstructions
    recon_psf()
    recon_2tar()
    recon_noise()
