"""
Tyson Reimer
University of Mantioba
"""

import os
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from umbms import get_proj_path, get_script_logger

from umbms.loadsave import load_pickle

from umbms.hardware import apply_sys_cor

from umbms.recon.breastmodels import get_roi

###############################################################################

__F_DIR = os.path.join(get_proj_path(),
                       "output/eucap2024/fatimah-sys/two-tar/")

__B_DIR = os.path.join(get_proj_path(),
                       "output/eucap2024/bed/2-tar/")

__G_DIR = os.path.join(get_proj_path(),
                       "output/eucap2024/gabbi/two-tar/")

__ROI_RHO = 8  # ROI radius, in [cm]


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

    x1, y1 = metadata[:, 0], metadata[:, 1]
    x2, y2 = metadata[:, 2], metadata[:, 3]

    cor_x1, cor_y1 = apply_sys_cor(x1, y1)
    cor_x2, cor_y2 = apply_sys_cor(x2, y2)

    return cor_x1, cor_y1, cor_x2, cor_y2


def _plt_one_subplt(img, ax, roi):

    img_norm = img / np.max(img)
    img_norm[~roi] = np.nan
    plt_img = ax.imshow(img_norm**2,
                        cmap='inferno',
                        extent=[-__ROI_RHO, __ROI_RHO, -__ROI_RHO, __ROI_RHO],
                        aspect='equal')
    ax.set_xlim(-__ROI_RHO + 3, __ROI_RHO - 3)
    ax.set_ylim(-__ROI_RHO + 3, __ROI_RHO - 3)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])

    return plt_img


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    scan_type = "mono"  # Scan type for Fatimah data

    # Load all images
    b_imgs = load_pickle(os.path.join(__B_DIR, "das.pickle"))
    f_imgs = load_pickle(os.path.join(__F_DIR, "%s/das_%s.pickle"
                                      % (scan_type, scan_type)))
    g_imgs = load_pickle(os.path.join(__G_DIR, "das.pickle"))

    # Load x/y positions of targets ------------

    # Load metadata: Format of x1, y1, x2, y2, default in [mm] so
    # divide by 10 to convert to [cm]
    bed_md = np.genfromtxt(os.path.join(get_proj_path(),
                                        "data/eucap2024/bed/2-tar/",
                                        "metadata.csv"),
                           delimiter=",",
                           dtype=float,
                           skip_header=2
                           )[:, 1:] / 10

    # Correct for setup positioning errors
    b_xs_1, b_ys_1, b_xs_2, b_ys_2 = setup_err_cor(metadata=bed_md)

    # Load Fatimah metadata
    f_md =load_pickle(os.path.join(get_proj_path(),
                                   "data/eucap2024/fatimah-sys/",
                                   "two_tar_md.pickle"))
    f_xs_1 = np.array([ii['x _1'] for ii in f_md]) / 10
    f_ys_1 = np.array([ii['y_1 '] for ii in f_md]) / 10
    f_xs_2 = np.array([ii['x_2'] for ii in f_md]) / 10
    f_ys_2 = np.array([ii['y_2'] for ii in f_md]) / 10

    # Load Gabbi metadata
    g_xs_2 = load_pickle(os.path.join(__G_DIR, "tar_xs.pickle"))
    g_ys_2 = load_pickle(os.path.join(__G_DIR, "tar_ys.pickle"))
    g_xs_1 = np.zeros_like(g_xs_2)
    g_ys_1 = np.zeros_like(g_ys_2)

    g_idx = np.argsort(np.sqrt(g_xs_2**2 + g_ys_2**2))

    # Indices of images for fig
    b_tar_idxs = [6, 7]
    f_tar_idxs = [12, 13]
    g_tar_idxs = [g_idx[-1], g_idx[-2]]

    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)

    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=16)

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, 0])
    # ax4 = fig.add_subplot(gs[1, 1])
    # ax5 = fig.add_subplot(gs[2, :])

    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax5 = plt.subplot2grid((3, 2), (2, 0))
    ax6 = plt.subplot2grid((3, 2), (2, 1))

    # ROI for suppressing external responses
    plt_roi = get_roi(roi_rho=__ROI_RHO,
                      m_size=np.size(b_imgs, axis=1),
                      arr_rho=__ROI_RHO)
    draw_angs = np.linspace(0, 2 * np.pi, 300)

    _ = _plt_one_subplt(img=np.abs(b_imgs[b_tar_idxs[0], :, :]),
                        ax=ax1,
                        roi=plt_roi)
    ax1.plot(0.1 * np.cos(draw_angs) + b_xs_1[b_tar_idxs[0]],
             0.1 * np.sin(draw_angs) + b_ys_1[b_tar_idxs[0]],
             'w',
             linewidth=1.0)
    ax1.plot(0.1 * np.cos(draw_angs) + b_xs_2[b_tar_idxs[0]],
             0.1 * np.sin(draw_angs) + b_ys_2[b_tar_idxs[0]],
             'w',
             linewidth=1.0)
    ax1.text(0.5, -0.7, "(a)", transform=ax1.transAxes,
             fontsize=14,
             va='center',
             ha='center')
    ax1.set_xlabel("x-axis (cm)", fontsize=12)
    ax1.set_ylabel("y-axis (cm)", fontsize=12)

    _ = _plt_one_subplt(img=np.abs(b_imgs[b_tar_idxs[1], :, :]),
                        ax=ax2,
                        roi=plt_roi)
    ax2.plot(0.1 * np.cos(draw_angs) + b_xs_1[b_tar_idxs[1]],
             0.1 * np.sin(draw_angs) + b_ys_1[b_tar_idxs[1]],
             'w',
             linewidth=1.0)
    ax2.plot(0.1 * np.cos(draw_angs) + b_xs_2[b_tar_idxs[1]],
             0.1 * np.sin(draw_angs) + b_ys_2[b_tar_idxs[1]],
             'w',
             linewidth=1.0)
    ax2.text(0.5, -0.7, "(b)", transform=ax2.transAxes,
             fontsize=14,
             va='center',
             ha='center')

    _ = _plt_one_subplt(img=np.abs(f_imgs[f_tar_idxs[0], :, :]),
                        ax=ax3,
                        roi=plt_roi)
    ax3.plot(0.1 * np.cos(draw_angs) + f_xs_1[f_tar_idxs[0]],
             0.1 * np.sin(draw_angs) + f_ys_1[f_tar_idxs[0]],
             'w',
             linewidth=1.0)
    ax3.plot(0.1 * np.cos(draw_angs) + f_xs_2[f_tar_idxs[0]],
             0.1 * np.sin(draw_angs) + f_ys_2[f_tar_idxs[0]],
             'w',
             linewidth=1.0)
    ax3.text(0.5, -0.7, "(c)", transform=ax3.transAxes,
             fontsize=14,
             va='center',
             ha='center')

    plt_img = _plt_one_subplt(img=np.abs(f_imgs[f_tar_idxs[1], :, :]),
                              ax=ax4,
                              roi=plt_roi)
    ax4.plot(0.1 * np.cos(draw_angs) + f_xs_1[f_tar_idxs[1]],
             0.1 * np.sin(draw_angs) + f_ys_1[f_tar_idxs[1]],
             'w',
             linewidth=1.0)
    ax4.plot(0.1 * np.cos(draw_angs) + f_xs_2[f_tar_idxs[1]],
             0.1 * np.sin(draw_angs) + f_ys_2[f_tar_idxs[1]],
             'w',
             linewidth=1.0)
    ax4.text(0.5, -0.7, "(d)", transform=ax4.transAxes,
             fontsize=14,
             va='center',
             ha='center')

    _ = _plt_one_subplt(img=np.abs(g_imgs[g_tar_idxs[0], :, :]),
                        ax=ax5,
                        roi=plt_roi)
    ax5.plot(0.1 * np.cos(draw_angs) + g_xs_1[g_tar_idxs[0]],
             0.1 * np.sin(draw_angs) + g_ys_1[g_tar_idxs[0]],
             'w',
             linewidth=1.0)
    ax5.plot(0.1 * np.cos(draw_angs) + g_xs_2[g_tar_idxs[0]],
             0.1 * np.sin(draw_angs) + g_ys_2[g_tar_idxs[0]],
             'w',
             linewidth=1.0)
    ax5.text(0.5, -0.7, "(e)", transform=ax5.transAxes,
             fontsize=14,
             va='center',
             ha='center')

    _ = _plt_one_subplt(img=np.abs(g_imgs[g_tar_idxs[1], :, :]),
                        ax=ax6,
                        roi=plt_roi)
    ax6.plot(0.1 * np.cos(draw_angs) + g_xs_1[g_tar_idxs[1]],
             0.1 * np.sin(draw_angs) + g_ys_1[g_tar_idxs[1]],
             'w',
             linewidth=1.0)
    ax6.plot(0.1 * np.cos(draw_angs) + g_xs_2[g_tar_idxs[1]],
             0.1 * np.sin(draw_angs) + g_ys_2[g_tar_idxs[1]],
             'w',
             linewidth=1.0)
    ax6.text(0.5, -0.7, "(f)", transform=ax6.transAxes,
             fontsize=14,
             va='center',
             ha='center')

    plt.subplots_adjust(wspace=-0.5, hspace=0.9)

    # cax = plt.subplot2grid((3, 3), (2, 2))
    # cbar = plt.colorbar(plt_img,
    #                     cax=cax,
    #                     orientation='vertical')
    cbar = plt.colorbar(plt_img,
                        ax=[ax1, ax2, ax3, ax4, ax5, ax6],
                        pad=0.05,
                        aspect=50
                        )
    cbar.ax.tick_params(labelsize=12)
    plt.show()
    plt.savefig(os.path.join(get_proj_path(),
                             "output/eucap2024/figs/",
                             "two_tar_recons.png"),
                dpi=300,
                transparent=False,
                bbox_inches="tight")