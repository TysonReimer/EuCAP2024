"""
Tyson Reimer
University of Manitoba
June 4th, 2019
"""

import numpy as np
import multiprocessing as mp

from functools import partial

###############################################################################


def fd_das(fd_data, phase_fac, freqs, n_cores=2):
    """Compute frequency-domain DAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension
    freqs : array_like, Nx1
        The frequencies used in the scan
    n_cores : int
        Number of cores used for parallel processing

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """

    n_fs = np.size(freqs)  # Find number of frequencies used

    # Correct for to/from propagation
    new_phase_fac = phase_fac**(-2)

    # Create func for parallel computation
    parallel_func = partial(_parallel_fd_das_func, fd_data, new_phase_fac,
                            freqs)

    workers = mp.Pool(n_cores)  # Init worker pool

    iterable_idxs = range(n_fs)  # Indices to iterate over

    # Store projections from parallel processing
    back_projections = np.array(workers.map(parallel_func, iterable_idxs))

    # Reshape
    back_projections = np.reshape(back_projections,
                                  [n_fs, np.size(phase_fac, axis=1),
                                   np.size(phase_fac, axis=2)])

    workers.close()  # Close worker pool

    # Sum over all frequencies
    img = np.sum(back_projections, axis=0)

    return img


def _parallel_fd_das_func(fd_data, new_phase_fac, freqs, ff):
    """Compute projection for given frequency ff

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    new_phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension, corrected for DAS
    ff : int
        Frequency index

    Returns
    -------
    this_projection : array_like, KxK
        Back-projection of this particular frequency-point
    """

    # Get phase factor for this frequency
    this_phase_fac = new_phase_fac ** freqs[ff]

    # Sum over antenna positions
    this_projection = np.sum(this_phase_fac * fd_data[ff, :, None, None],
                             axis=0)

    return this_projection
