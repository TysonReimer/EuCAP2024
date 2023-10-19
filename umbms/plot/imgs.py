"""
Tyson Reimer
University of Manitoba
June 18, 2019
"""

import numpy as np
import matplotlib.pyplot as plt

import umbms.recon.breastmodels as breastmodels


###############################################################################


# TODO: Revise plot_img() func to accommodate new antennas
def plot_img(img, tar_xs=None, tar_ys=None, tar_rads=None,
             phant_rad=0.0, roi_rho=8,
             crop_img=True,
             cmap='inferno', cbar_fmt='%.1f',
             title='',
             save_str='', save_fig=False,
             transparent=False, dpi=150, save_close=True):
    """

    Parameters
    ----------
    img : array_like
        The 2D image to be reconstructed
    tar_xs : list, optional
        List of the x-positions of the targets, in [cm]
    tar_ys : list, optional
        List of the y-positions of the targets, in [cm]
    tar_rads : list, optional
        List of the radii of the targets, in [cm]
    phant_rad : float, optional
        Radius of the (presumably circular) phantom, in [cm]
    roi_rho : float, optional
        Radius of the region of interest that will be reconstructed,
        in [cm]
    crop_img : bool, optional
        If True, crops the image so that only pixels in the circular
        region of interest are non-nan
    cmap : str, optional
        The colormap to use; defualt is 'inferno'
    cbar_fmt : str, optional
        Format str for the colorbar tick labels
    title : str, optional
        The title of the plot
    save_str : str, optional
        The full path and file name if saving the fig as a .png
    save_fig : bool, optional
        If True, will save the figure as a .png file
    transparent : bool, optional
        If True, and if save_fig, the fig will be saved with a
        transparent background
    dpi : int, optional
        The DPI to be used for saving
    save_close : bool, optional
        If True, will close the plot after saving

    """

    if tar_xs is None:
        tar_xs = []
    if tar_ys is None:
        tar_ys = []
    if tar_rads is None:
        tar_rads = []

    assert len(tar_xs) == len(tar_ys), "tar_xs, tar_ys diff lengths"
    assert len(tar_xs) == len(tar_rads), 'tar_xs, tar_rads diff lengths'

    img_to_plt = img * np.ones_like(img)  # Copy the image for plot

    # If cropping the image at the antenna-trajectory boundary
    if crop_img:

        # Find pixels in ROI
        in_roi = breastmodels.get_roi(roi_rho=roi_rho,
                                      m_size=np.size(img_to_plt, axis=0),
                                      arr_rho=roi_rho)

        # Set the pixels outside the antenna trajectory to NaN
        img_to_plt[np.logical_not(in_roi)] = np.NaN

    # Define angles for plot the tissue geometry
    draw_angs = np.linspace(0, 2 * np.pi, 1000)

    # Define the x/y coordinates of the approximate breast outline
    breast_xs, breast_ys = (phant_rad * np.cos(draw_angs),
                            phant_rad * np.sin(draw_angs))

    # Bounds for x/y axes ticks in plt
    tick_bounds = [-roi_rho, roi_rho, -roi_rho, roi_rho]

    # Set the font to times new roman
    plt.rc('font', family='Times New Roman')
    plt.figure()  # Make the figure window

    # Set the size of the axis tick labels
    plt.tick_params(labelsize=14)

    # Plot the image
    plt.imshow(img_to_plt, cmap=cmap, extent=tick_bounds, aspect='equal')

    # # Set the x/y-ticks at multiples of 5 cm
    # plt.gca().set_xticks([-8, -4, 0, 4, 8])
    # plt.gca().set_yticks([-10, -5, 0, 5, 10])

    # Specify the colorbar tick format and size
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=14)

    # Set the x/y axes limits
    plt.xlim([-roi_rho, roi_rho])
    plt.ylim([-roi_rho, roi_rho])

    # Plot breast (circular) outline
    plt.plot(breast_xs, breast_ys, 'w--', linewidth=2)

    for ii in range(len(tar_xs)):  # For each target

        # Get the x/y coords for plotting boundary
        plt_xs = tar_rads[ii] * np.cos(draw_angs) + tar_xs[ii]
        plt_ys = tar_rads[ii] * np.sin(draw_angs) + tar_ys[ii]

        # Plot the circular boundary
        plt.plot(plt_xs, plt_ys, 'w', linewidth=2.0)

    plt.title(title, fontsize=20)  # Make the plot title
    plt.xlabel('x-axis (cm)', fontsize=16)  # Make the x-axis label
    plt.ylabel('y-axis (cm)', fontsize=16)  # Make the y-axis label
    plt.tight_layout()  # Remove excess whitespace in the figure

    # If saving the image, save it to the save_str path and close it
    if save_fig:
        plt.savefig(save_str, transparent=transparent, dpi=dpi,
                    bbox_inches='tight')

        if save_close:  # If wanting to close the fig after saving
            plt.close()
