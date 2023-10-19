"""
Tyson Reimer
University of Manitoba
September 23, 2023

Includes contributions by Fatimah Eashour
"""

import os
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle

from umbms.sigproc.sigproc import iczt

###############################################################################

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/gabbi/psf/")

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


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Dir where DAS .pickle is
    img_dir = os.path.join(__O_DIR, "psf/")

    o_dir = os.path.join(__O_DIR, "res-dqm/")
    verify_path(o_dir)

    # Load DAS images
    imgs = load_pickle(os.path.join(img_dir, "das.pickle"))

    # Target x/y positions
    xs = load_pickle(os.path.join(__O_DIR, "psf/tar_xs.pickle"))
    ys = load_pickle(os.path.join(__O_DIR, "psf/tar_ys.pickle"))

    tar_rhos = np.sqrt(xs ** 2 + ys ** 2)
    tar_rhos[np.isnan(tar_rhos)] = 100  # Remove nans

    # ---- Analysis below -----------------------------------------------------

    # Index of scan with target closest to (0, 0)
    tar_idx = np.argmin(tar_rhos)

    # Load this frequency domain S11
    tar_fd = np.load(os.path.join(__D_DIR, os.listdir(__D_DIR)[tar_idx]))

    ini_t = 0
    fin_t = 5e-9
    n_ts = 1400

    # Convert to the time domain...
    tar_td = iczt(fd_data=tar_fd,
                  ini_f=__INI_F,
                  fin_f=__FIN_F,
                  ini_t=ini_t,
                  fin_t=fin_t,
                  n_ts=n_ts,
                  )

    ts = np.linspace(ini_t, fin_t, n_ts)

    td_sig = np.abs(tar_td)  # Take abs-value

    fwhm = np.zeros([np.size(tar_td, axis=1),])  # Init arr

    for ii in range(np.size(tar_td, axis=1)):  # For each antenna

        max_peak = np.argmax(td_sig[:, ii])  # Find max peak location

        # Intensity at half max
        half_max_intensity = 0.5 * td_sig[max_peak, ii]

        left_idx = np.where(td_sig[:, ii][:max_peak]
                            < half_max_intensity)[0][-1]
        right_idx = np.where(td_sig[:, ii][max_peak:]
                             < half_max_intensity)[0][0] + max_peak

        fwhm[ii] = ts[right_idx] - ts[left_idx]

        print("dx:\t%.3f mm" % (299792458 * 1e3 * fwhm[ii] / 2))

        plt.figure()
        plt.plot(ts, td_sig[:, ii])
        plt.fill_between(x=ts[left_idx: right_idx],
                         y1=np.zeros_like(ts[left_idx: right_idx]),
                         y2=td_sig[:, ii][left_idx: right_idx],
                         color='g',
                         alpha=0.3)
        plt.savefig(os.path.join(o_dir, "sig_%d.png" % ii))
        plt.close()

    fwhm_d = 299792458 * 1e3 * fwhm / 2
    fwhm_d = fwhm_d[fwhm_d <= 100]

    logger.info('FWHM: (%.2f +/- %.2f) mm'
                % (np.mean(fwhm_d), np.std(fwhm_d)))
