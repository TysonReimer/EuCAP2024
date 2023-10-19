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

from umbms.loadsave import load_pickle


###############################################################################

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/bed/")

__O_DIR = os.path.join(get_proj_path(), "output/eucap2024/bed/")
verify_path(__O_DIR)


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    o_dir = os.path.join(__O_DIR, "noise/")
    verify_path(o_dir)

    das_imgs = load_pickle(os.path.join(o_dir, "das.pickle"))

    # Exclude the index where a new session was started (phantom move
    idxs_to_excl = [10, 20]

    # Init arr
    all_snrs = np.zeros(np.size(das_imgs, axis=0) - len(idxs_to_excl) - 1)
    cc = 0

    for ii in range(1, np.size(das_imgs, axis=0)):  # For each scan

        if not (ii in idxs_to_excl):

            img = np.abs(das_imgs[ii, :, :])**2  # This image

            # Difference vs previous image
            img_diff = (np.abs(das_imgs[ii, :, :])**2
                        - np.abs(das_imgs[ii - 1, :, :])**2)

            # SNR map
            snr = 10 * np.log10(np.abs(img / img_diff))

            plt.figure()
            plt.imshow(snr, cmap='inferno')
            plt.colorbar()
            plt.savefig(os.path.join(o_dir, "temp/snr_img_%d.png" % ii))
            plt.close()

            all_snrs[cc] = np.mean(snr)
            cc += 1

            plt.figure(figsize=(12, 6))
            plt.title("SNR: %.2f +/- %.2f"
                      % (np.mean(snr), np.std(snr)))
            sns.histplot(img_diff.flatten(), kde=True)
            plt.savefig(os.path.join(o_dir, "temp/diff_distro_%d.png" % ii))
            plt.close()

    logger.info("SNR (dB): %.3f +/- %.3f"
                % (np.mean(all_snrs), np.std(all_snrs)))
