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

from scipy.ndimage import rotate
from scipy.signal import find_peaks

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle

from umbms.recon.extras import get_pix_xys



###############################################################################

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/fatimah-sys/")

__O_DIR = os.path.join(get_proj_path(), "output/eucap2024/fatimah-sys/")
verify_path(__O_DIR)

__ROI_RHO = 8  # ROI radius, in [cm]

__M_SIZE = int(__ROI_RHO * 2 / 0.1)  # Image size

###############################################################################


def _img_to_mtf2d(img):
    """Convert image to 2D modulation transfer function

    Parameters
    ----------
    img : array_like
        A 2D image

    Returns
    -------
    mtf_2d : array_like
        A 2D modulation transfer function of the image
    """

    # Convert to frequency domain and shift so 0 freq at center
    img_fft = np.fft.fftshift(np.fft.fft2(img))

    # Get the 2D MTF by normalizing
    mtf_2d = np.abs(img_fft) / np.max(np.abs(img_fft))

    return mtf_2d


def _mtf2d_to_1d(mtf_2d, dx):
    """Extract a 1D representative MTF from the 2D MTF

    Parameters
    ----------
    mtf_2d : array_like
        The 2D MTF of an images
    dx :

    Returns
    -------
    mtf_1d : array_like
        The 1D MTF, averaged over coordinate rho in freq space
    mtf_1d_uncty : array_like
        The uncertainty in the 1D MTF values
    plt_freqs : array_like
        The spatial frequencies used to index the 1D MTF
    """

    # Get the spatial frequencies corresponding to the MTF
    spatial_fs = np.fft.fftshift(np.fft.fftfreq(n=__M_SIZE, d=dx))

    # Define the polar radii for numerical integration
    rads_for_plt = np.linspace(0, __ROI_RHO, np.sum(spatial_fs >= 0))
    d_rad = np.diff(rads_for_plt)[0]  # The distance between each radii

    # Get the pixel x/y positions
    pix_xs, pix_ys = get_pix_xys(m_size=__M_SIZE, roi_rho=__ROI_RHO)

    # Init arr for storing the 1D MTF
    mtf_1d = np.zeros_like(rads_for_plt)
    mtf_1d_uncty = np.zeros_like(rads_for_plt)
    plt_freqs = np.zeros_like(rads_for_plt)

    # For each ring used for integration
    for jj in range(len(rads_for_plt)):

        # For the DC component of the frequencies and 1D MTF...
        if jj == 0:

            mtf_1d[jj] = 1  # Set first entry to unity
            mtf_1d_uncty[jj] = 0
            plt_freqs[jj] = 0

        else:  # For the non-DC components...

            # Find the pixels within this ring
            tar_pix = np.logical_and(
                np.sqrt(pix_xs ** 2 + pix_ys ** 2) >= rads_for_plt[jj] - d_rad / 2,
                np.sqrt(pix_xs ** 2 + pix_ys ** 2) < rads_for_plt[jj] + d_rad / 2,
            )

            # Obtain the mean and stdev of the MTF values for the
            # pixels in this ring
            mtf_1d[jj] = np.mean(mtf_2d[tar_pix])
            mtf_1d_uncty[jj] = np.std(mtf_2d[tar_pix])

            # Target pixels for the spatial frequencies of this ring
            fs_tar_pix = np.logical_and(
                np.abs(pix_xs[0, :]) >= rads_for_plt[jj] - d_rad / 2,
                np.abs(pix_xs[0, :]) < rads_for_plt[jj] + d_rad / 2
            )

            # Store the corresponding spatial frequencies
            plt_freqs[jj] = np.mean(np.abs(spatial_fs[fs_tar_pix]))

    return mtf_1d, mtf_1d_uncty, plt_freqs


def do_mtf_resolution(scan_type):
    """Do MTF-based resolution analysis
    """

    assert scan_type in ["mono", "multi"], "Error: invalid scan_type"

    # Dir where DAS .pickle is
    img_dir = os.path.join(__O_DIR, "psf/%s/" % scan_type)

    # Dir to save figs etc.
    o_dir = os.path.join(__O_DIR, "resolution-%s/" % scan_type)
    verify_path(o_dir)

    # Load DAS images
    imgs = load_pickle(os.path.join(img_dir, "das_%s.pickle" % scan_type))

    # Find the width of a pixel in the image, in units of [cm]
    dx = np.diff(np.linspace(-__ROI_RHO, __ROI_RHO, __M_SIZE))[0]

    mtf_cutoff = 0.05  # Cutoff value for MTF

    # Init lists for storing
    all_resolutions = []
    mtfs_1d = []
    mtfs_1d_uncty = []

    for ii in range(np.size(imgs, axis=0)):  # For each image

        if (ii + 1) % 5 != 0:  # The 5th scans are reference scans

            # Get the 2D MTF of the image
            mtf_2d = _img_to_mtf2d(img=np.abs(imgs[ii, :, :])**2)

            # Extract the 1D MTF from the 2D MTF
            mtf_1d, mtf_1d_uncty, plt_freqs = _mtf2d_to_1d(mtf_2d=mtf_2d,
                                                           dx=dx)

            # Store results in our ever-growing lists
            mtfs_1d.append(mtf_1d)
            mtfs_1d_uncty.append(mtf_1d_uncty)

            # Calculate the predicted resolution given the MTF cutoff
            the_res = 1 / plt_freqs[np.argmin(np.abs(mtf_1d - mtf_cutoff))]

            all_resolutions.append(the_res)  # Store result for later

    # Find the average MTF across all images
    mean_mtf = np.mean(np.array(mtfs_1d), axis=0)

    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=18)
    plt.errorbar(plt_freqs, mean_mtf,
                 yerr=3 * np.std(np.array(mtfs_1d), axis=0),
                 capsize=5,
                 color='k',
                 marker='o',
                 linestyle='--',
                 )
    plt.xlabel(r'Spatial Frequency (cm$^{\mathdefault{-1}}$)', fontsize=22)
    plt.ylabel('Modulation Transfer Function', fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(o_dir, "mtf.png"), dpi=300)

    # Report the average resolution across all scans
    logger.info('Resolution:\t%.3f +/- %.3f'
                % (np.mean(np.array(all_resolutions)),
                   np.std(np.array(all_resolutions))))


###############################################################################
# ------------ Two-target resolution analysis below --------------------------


def do_two_tar_resolution(scan_type):
    """Do two-target resolution analysis
    """

    assert scan_type in ["mono", "multi"], "Error: invalid scan_type"

    # Dir where DAS .pickle is
    img_dir = os.path.join(__O_DIR, "two-tar/%s/" % scan_type)

    # Dir to save figs etc.
    o_dir = os.path.join(__O_DIR, "resolution-%s/" % scan_type)
    verify_path(o_dir)

    # Load DAS images
    das_imgs = load_pickle(os.path.join(img_dir, "das_%s.pickle" % scan_type))

    # Load metadata
    md = load_pickle(os.path.join(__D_DIR, "two_tar_md.pickle"))

    xs_1 = np.array([ii['x _1'] for ii in md]) / 10
    ys_1 = np.array([ii['y_1 '] for ii in md]) / 10
    xs_2 = np.array([ii['x_2'] for ii in md]) / 10
    ys_2 = np.array([ii['y_2'] for ii in md]) / 10

    # Get pixel x/y positions, in [cm]
    pix_xs, pix_ys = get_pix_xys(m_size=__M_SIZE, roi_rho=__ROI_RHO)

    dx = pix_xs[0, 1] - pix_xs[0, 0]  # Pixel size

    for ii in range(np.size(das_imgs, axis=0)):

        if (ii + 1) % 5 != 0:  # Skip 5th scans, they are references

            img = np.abs(das_imgs[ii, :, :])**2  # Target image

            # Define circular ROIs for each target
            tar1_roi = np.sqrt(
                (xs_1[ii] - pix_xs) ** 2 + (ys_1[ii] - pix_ys) ** 2) < 1
            tar2_roi = np.sqrt(
                (xs_2[ii] - pix_xs) ** 2 + (ys_2[ii] - pix_ys) ** 2) < 1

            # Create image with the response from target 1 suppressed
            img_tar1_suppressed = np.copy(img)
            img_tar1_suppressed[tar1_roi] = 0

            # Create image with the response from target 2 suppressed
            img_tar2_suppressed = np.copy(img)
            img_tar2_suppressed[tar2_roi] = 0

            # Find the location of the max responses of the two objects
            tar1_max_loc = np.unravel_index(np.argmax(img_tar2_suppressed.T),
                                            shape=np.shape(img))
            tar2_max_loc = np.unravel_index(np.argmax(img_tar1_suppressed.T),
                                            shape=np.shape(img))

            # Slope of the line intersecting the two maxima
            slope = ((tar2_max_loc[1] - tar1_max_loc[1])
                     / (tar2_max_loc[0] - tar1_max_loc[0]))

            ang_for_rot = np.arctan(slope)  # Find angle for rotating image

            # NOTE: If ang_for_rot is nan, because tar1/tar2 max at
            # same pixel, i.e., not in the suppressed ROIs

            if ~np.isnan(ang_for_rot):  # If valid angle

                # Create rotated image
                rot_img = rotate(img, angle=np.rad2deg(ang_for_rot))

            else:  # If the angle is invalid, i.e. only 1 peak in img
                rot_img = img

            # Normalize
            rot_img /= np.max(rot_img)

            # Find new max location
            max_loc = np.unravel_index(np.argmax(rot_img),
                                       shape=np.shape(rot_img))

            # Get the 1D slice of the image of interest
            tar_img_slice = rot_img[max_loc[0], :]

            # Find indices of the two target responses
            peaks = find_peaks(tar_img_slice)

            # Sort the peaks so that the two target peaks are first two
            sorted_peaks = peaks[0][np.flip(np.argsort(tar_img_slice[peaks[0]]))]

            if len(sorted_peaks) >= 2:

                # Target peaks
                tar_peaks = np.array([sorted_peaks[0], sorted_peaks[1]])

                # Pixels between two peaks
                roi_between = tar_img_slice[np.min(tar_peaks): np.max(tar_peaks)]

                # Ratio between smaller of the two target peaks
                # and the minimum value between them
                peak_ratio = (np.min(np.array([tar_img_slice[sorted_peaks[1]],
                                              tar_img_slice[sorted_peaks[0]]]))
                              / np.min(roi_between))

                plt_xs = (np.size(rot_img, axis=1) * dx / 2 -
                          np.arange(np.size(rot_img, axis=1)) * dx)

                plt.figure()
                plt.rc("font", family='Times New Roman')
                plt.tick_params(labelsize=14)
                plt.title('Peak Ratio: %.3f\nSeparation: %.3f cm'
                          % (1 / peak_ratio,
                             np.sqrt((xs_1[ii] - xs_2[ii])**2
                                     + (ys_1[ii] - ys_2[ii])**2)))
                plt.plot(plt_xs, tar_img_slice, 'k')
                plt.scatter(plt_xs[tar_peaks], tar_img_slice[tar_peaks], color='b',
                            marker='x')
                plt.xlabel('Position Along Intersection (cm)', fontsize=22)
                plt.ylabel("Image Intensity", fontsize=22)
                plt.xlim([-__ROI_RHO, __ROI_RHO])
                plt.tight_layout()
                plt.savefig(os.path.join(o_dir, 'x_section_expt_%d.png' % (ii)),
                            dpi=300, transparent=False)
            else:
                logger.info('Error idx %d:\tOnly 1 peak' % ii)


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    do_mtf_resolution(scan_type="mono")
    do_mtf_resolution(scan_type="multi")
    do_two_tar_resolution(scan_type="mono")
    do_two_tar_resolution(scan_type="multi")
