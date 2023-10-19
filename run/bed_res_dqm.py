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

from umbms.hardware import apply_sys_cor

from umbms.sigproc.sigproc import iczt

###############################################################################

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/bed/")

__O_DIR = os.path.join(get_proj_path(), "output/eucap2024/bed/")
verify_path(__O_DIR)

__INI_F = 2e9  # Initial freq: 700 MHz
__FIN_F = 9e9  # Final freq: 8 GHz
__N_FS = 1001  # Number of frequencies


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Dir for fd / md
    d_dir = os.path.join(__D_DIR, "psf/")

    o_dir = os.path.join(__O_DIR, "res-dqm/")
    verify_path(o_dir)

    # Load metadata
    md = load_pickle(os.path.join(d_dir, "md.pickle"))

    xs, ys = md[:, 0], md[:, 1]  # Get xs/ys separately

    xs, ys = apply_sys_cor(xs=xs, ys=ys)  # Apply setup error correction

    tar_rhos = np.sqrt(xs**2 + ys**2)  # Target rho position, [cm]

    # Load the frequency domain data
    fd = load_pickle(os.path.join(d_dir, "s11.pickle"))

    # ---- Analysis below -----------------------------------------------------

    # Index of scan with target closest to (0, 0)
    tar_idx = np.argmin(tar_rhos)

    tar_fd = fd[tar_idx, :, :]  # This frequency domain S11

    ini_t = 0.5e-9
    fin_t = 5.5e-9
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

    fwhm = np.zeros([np.size(tar_td, axis=1),])
    for ii in range(np.size(tar_td, axis=1)):

        max_peak = np.argmax(td_sig[:, ii])  # Find max peak location

        # Intensity at half max
        half_max_intensity = 0.5 * td_sig[max_peak, ii]

        # Find the left/right sides of the FWHM
        left_idx = np.where(td_sig[:, ii][:max_peak]
                            < half_max_intensity)[0][-1]
        right_idx = np.where(td_sig[:, ii][max_peak:]
                             < half_max_intensity)[0][0] + max_peak

        # Identify the FWHM in units of time
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

    logger.info('FWHM: (%.2f +/- %.2f) mm'
                % (np.mean(fwhm_d), np.std(fwhm_d)))

