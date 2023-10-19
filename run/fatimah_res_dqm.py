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
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle

from umbms.sigproc.sigproc import iczt


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


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    scan_type = "mono"

    # Dir where DAS .pickle is
    img_dir = os.path.join(__O_DIR, "psf/%s/" % scan_type)

    # Dir to save figs etc.
    o_dir = os.path.join(__O_DIR, "res-dqm-%s/" % scan_type)
    verify_path(o_dir)

    # Load DAS images
    imgs = load_pickle(os.path.join(img_dir, "das_%s.pickle" % scan_type))

    md = load_pickle(os.path.join(__D_DIR, "psf_md.pickle"))

    xs = np.array([ii['x '] for ii in md])
    ys = np.array([ii['y '] for ii in md])

    tar_rhos = np.sqrt(xs**2 + ys**2)
    tar_rhos[np.isnan(tar_rhos)] = 100

    # Load the frequency domain data
    fd = load_pickle(os.path.join(__D_DIR, "psf_fd.pickle"))

    # ---- Analysis below -----------------------------------------------------

    # Load the calibration factors
    cal_factors = load_pickle(os.path.join(__D_DIR, "cal_factors.pickle"))

    loss_cal = cal_factors[0]  # Calibration factors for losses
    t_delay_cal = cal_factors[1]  # Calibration factors for t-delays

    # Index of scan with target closest to (0, 0)
    tar_idx = np.argmin(tar_rhos)
    ref_idx = 34  # Corresponding reference scan

    tar_fd = fd[tar_idx]  # The S parameters for this scan

    # The S_nn parameters only
    s_nn_fd = np.zeros([__N_ANTS, __N_FS], dtype=complex)
    cal_ts = np.zeros([__N_ANTS,])  # Arr for cal_t

    for ii in range(__N_ANTS):  # For each antenna

        # Get the S_nn after reference-subtraction
        s_nn_fd[ii, :] = ((tar_fd["S%d_%d" % (ii, ii)]
                          - fd[ref_idx]["S%d_%d" % (ii, ii)])
                          / loss_cal["S%d_%d" % (ii, ii)])

        # Store the t-delay here
        cal_ts[ii] = t_delay_cal["S%d_%d" % (ii, ii)]

    ini_t = 2 * __ANT_T + np.min(cal_ts)
    fin_t = 5.5e-9 + 2 * __ANT_T + np.mean(cal_ts)
    n_ts = 1400

    # Convert to the time domain...
    tar_td = iczt(fd_data=s_nn_fd.T,
                  ini_f=__INI_F,
                  fin_f=__FIN_F,
                  ini_t=ini_t,
                  fin_t=fin_t,
                  n_ts=n_ts,
                  )

    ts = np.linspace(ini_t, fin_t, n_ts)

    td_sig = np.abs(tar_td)  # Take abs-value

    fwhm = np.zeros([np.size(tar_td, axis=1),])
    for ii in range(np.size(tar_td, axis=1)):

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

    fwhm_d = fwhm_d[fwhm_d <= 60]  # Remove outlier

    logger.info('FWHM: (%.2f +/- %.2f) mm'
                % (np.mean(fwhm_d), np.std(fwhm_d)))
