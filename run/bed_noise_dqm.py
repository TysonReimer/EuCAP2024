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
import seaborn as sns
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_birrs_folder

###############################################################################

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/bed/")

__O_DIR = os.path.join(get_proj_path(), "output/eucap2024/bed/")
verify_path(__O_DIR)

###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    o_dir = os.path.join(__O_DIR, "noise/")
    verify_path(o_dir)

    # Load the frequency domain data
    fd = load_birrs_folder(folder_path=os.path.join(__D_DIR, "noise/"),
                           id_str="exp",
                           ref_idxs=[0],
                           load_s21=False,
                           ref_to_use=0)

    # Exclude the index where a new session was started (phantom move
    idxs_to_excl = [10, 20]

    # Init arr
    all_snrs = np.zeros(np.size(fd, axis=0) - len(idxs_to_excl) - 1)
    cc = 0  # Init counter

    for ii in range(1, np.size(fd, axis=0)):  # For each scan

        if not (ii in idxs_to_excl):

            fd_here = np.abs(fd[ii, :, :])

            fd_diff = fd_here - np.abs(fd[ii - 1, :, :])

            snr = 10 * np.log10(np.abs(fd_here / fd_diff))

            logger.info('ii = %d\tSNR (dB): %.2f +/- %.2f'
                        % (ii, np.mean(snr), np.std(snr)))

            plt.figure()
            plt.imshow(snr, cmap='inferno', aspect='auto')
            plt.colorbar()
            plt.savefig(os.path.join(o_dir, "temp/snr_fd_%d.png" % ii))
            plt.close()

            all_snrs[cc] = np.mean(snr)
            cc += 1

            plt.figure(figsize=(12, 6))
            plt.title("SNR: %.2f +/- %.2f"
                      % (np.mean(snr), np.std(snr)))
            sns.histplot(fd_diff.flatten(), kde=True)
            plt.savefig(os.path.join(o_dir, "temp/diff_fd_distro_%d.png" % ii))
            plt.close()

    logger.info("SNR (dB): %.3f +/- %.3f"
                % (np.mean(all_snrs), np.std(all_snrs)))

