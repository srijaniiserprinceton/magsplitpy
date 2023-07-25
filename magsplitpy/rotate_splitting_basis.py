# code to rotate the spherical harmonic basis to a different basis 
# as compared to the one in which it was computed. This is mainly
# important when, for example, the rotation and magnetic axes are not aligned.

import numpy as np
from math import factorial as factorial

fac = np.vectorize(factorial)

def Wigner_d_matrix(ell,beta):
    """
    Function to calculate the Wigner d-matrix for a certain specified $\ell$ and $\\beta$.
    
    Parameters:
    -----------
    ell : int, scalar
          The angular degree of spherical harmonic or the mode of interest.
    
    beta : float, scalar
           Relative angle of inclination between the two coordinate axes in radians.

    Returns:
    --------
    wigner_d_matrix : ndarray, shape (2*ell+1 x 2*ell+1)
                      The Wigner d-matrix.
    """
    # investigating the factorials, it is easy to see that
    # the lowest value of s can be 0 (otherwise the s! term has a negatve factorial)
    # and the highest possible value of s can be 2 * ell
    s_arr = np.arange(0, 2*ell+1)

    # making the m and m_ arrays and meshes
    m_arr = np.arange(-ell, ell+1)

    mm, mm_, ss = np.meshgrid(m_arr, m_arr, s_arr, indexing='ij')

    # making the matrix with an extra s dimension which will be summed over at the end
    d_matrix = np.zeros((2*ell+1, 2*ell+1, len(s_arr)))

    # making the factorial terms arguments
    fac1_arg = ell + mm_ - ss
    fac2_arg = ss
    fac3_arg = mm - mm_ + ss
    fac4_arg = ell - mm - ss

    # finding masks for where any of these factorial arguments is negative
    mask_neg_fac_arg = (fac1_arg < 0) + (fac2_arg < 0) + (fac3_arg < 0) + (fac4_arg < 0)

    # extracting only the valid entries for the factorial
    fac1_arg = fac1_arg[~mask_neg_fac_arg]
    fac2_arg = fac2_arg[~mask_neg_fac_arg]
    fac3_arg = fac3_arg[~mask_neg_fac_arg]
    fac4_arg = fac4_arg[~mask_neg_fac_arg]

    # computing on only the non-negative factorial arguments
    d_matrix[~mask_neg_fac_arg] = (np.power(-1, np.abs(mm - mm_ + ss)) * np.power(np.cos(0.5 * beta), np.abs(2*ell + mm_ - mm - 2*ss)) *\
                                   np.power(np.sin(0.5 * beta), np.abs(mm - mm_ + 2*ss)))[~mask_neg_fac_arg] \
                                   / (fac(fac1_arg) * fac(fac2_arg) * fac(fac3_arg) * fac(fac4_arg))
    
    # summing the s dimension
    d_s_summed = np.sum(d_matrix, axis=2)

    # multiplying the s-independent prefactor
    # redefining the mm and mm_ meshgrid without an s dimension this time
    mm, mm_ = np.meshgrid(m_arr, m_arr, indexing='ij')
    s_indep_prefac = np.sqrt(fac(ell+mm) * fac(ell-mm) * fac(ell+mm_) * fac(ell-mm_))

    # total Wigner d-matrix for a specific ell and beta. Shape (2*ell+1 x 2*ell+1)
    wigner_d_matrix = s_indep_prefac * d_s_summed
    
    return wigner_d_matrix


if __name__ == "__main__":
    ell = 2
    beta = np.pi/4.0

    D = Wigner_d_matrix(ell, beta)