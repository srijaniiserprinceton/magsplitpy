import numpy as np
from magsplitpy import sph_transforms as spht
from magsplitpy import misc_funcs as fn

NAX = np.newaxis

# extracting the GSH coefficients corresponding to a general 3D magnetic field
def get_B_GSHcoeffs_from_B(B, nmax=5, mmax=5):
    """
    Given a 3D magnetic field B on a spherical surface with all the three vector components 
    $(B_r, B_{\\theta}, B_{\phi})$, this function calculates the corresponding
    GSH coefficients. 

    Parameters:
    -----------
    B : array_like of floats, shape (Ntheta x Nphi)
        3D magnetic field on a spherical shell with all the vector components.

    nmax : int, optional
           The maximum angular degree of GSH transform.

    mmax : int, optional
           The maximum azimuthal order of GSH transform.

    Returns:
    --------
    B_mu_st_r : ndarray, shape (3, nmax, mmax)
                Array of cofficients for each spherical shell. Retains
                the same number of points in radius.
    """
    Br, Btheta, Bphi = B[0], B[1], B[2]
    # converting the Br(r,theta,phi) field to B^0_st(r)
    B0_st_r = spht.field2coef(np.array([Br]))

    # converting the Btheta(r,theta,phi) and Bphi(r,theta,phi) to
    # Bv_{st}(r) and Bw_{st}(r) of the Jackson convention
    Bvec = np.array([Btheta, Bphi])
    B_Jackson_coefs = spht.field2coef(Bvec)
    BvJ_st_r, BwJ_st_r = B_Jackson_coefs.scoef1, B_Jackson_coefs.scoef2

    # converting to the DT convention first before converting to GSH components
    BvDT_st_r, BwDT_st_r = spht.convert_coeffs_Jackson2DT(BvJ_st_r, BwJ_st_r)
    # converting to the B^{-}_{st}(r) and B^{+}_{st}(r) convention
    Bm_st_r, Bp_st_r = spht.make_GSH_from_VSH(BvDT_st_r, BwDT_st_r)

    # creating B^{mu}_st(r) array
    B_mu_st_r = np.array([Bm_st_r._array_2d_repr(),
                          B0_st_r._array_2d_repr(),
                          Bp_st_r._array_2d_repr()])

    # returning B^0_st_r(r), 
    return B_mu_st_r
 
# using the GSH coefficients of B to build GSH coefficients of BB
def make_BB_GSH_from_B_GSH(B_mu_st_obj, sB_max=5):
    """
    Returns the GSH component of Lorentz Stress from the GSH components of B-field.

    Parameters:
    -----------
    B_mu_st_r : ndarray, shape (3, nmax, mmax, r)
                Array of cofficients for B.

    Returns:
    --------
    h_mu_nu_st_r : ndarray, shape (3, 3, nmax, mmax, r)
                   Array of cofficients for BB.
    """
    # setting up spherical harmonic quantum numbers
    mu = np.array([-1,0,1])
    nu = np.array([-1,0,1])

    # quantum numbers for B
    sB_arr = np.arange(0, sB_max+1)
    tB_arr = np.arange(-sB_max, sB_max+1)
    
    # quantum numbers for BB
    sBB_max = 2 * sB_max   # from wigner 3j selection rules
    sBB_arr = np.arange(0, sBB_max+1)
    tBB_arr = np.arange(-sBB_max, sBB_max+1)

    # vectorizing the wigner calculation
    # wig_calc = np.vectorize(fn.w3j_vecm)
    wig_calc = np.vectorize(fn.wig)

    # the array to store the h^{mu,nu}_{st} components
    h_mu_nu_st_r = np.zeros((3, 3, sBB_max+1, 2*sBB_max+1, B_mu_st_r.shape[-1]),
                             dtype = complex)

    # making meshgrid to make less loops
    mumu,nunu,ss_BB,tt_BB = np.meshgrid(mu,nu,sBB_arr,tBB_arr,indexing='ij')

    # looping over the dimensions to be summed over in Eqn(D56) in Das 2020
    for s1_idx, s1 in enumerate(sB_arr):
        for s2_idx, s2 in enumerate(sB_arr):
            # wigner does not depend on t1 and t2
            wig1 = wig_calc(s1,ss_BB,s2,mumu,-(mumu+nunu),nunu)
            # prefactor not dependent on t1 and t2
            prefac = (-1)**(np.abs(tt_BB+mumu+nunu)) * np.sqrt((2*s1+1)*(2*s2+1)*(2*ss_BB+1)/(4*np.pi))
            for t1_idx, t1 in enumerate(tB_arr):
                for t2_idx, t2 in enumerate(tB_arr):
                    print(s1,s2,t1,t2)
                    # implementing Eqn(D56) in Das 2020
                    wig2 = wig_calc(s1,ss_BB,s2,t1,-tt_BB,t2)
                    h_mu_nu_st_r += B_mu_st_r[:,NAX,s1_idx,t1_idx,NAX,NAX,:] *\
                                    B_mu_st_r[NAX,:,s2_idx,t2_idx,NAX,NAX,:] *\
                                    (prefac * wig1 * wig2)[:,:,:,:,NAX]

    return h_mu_nu_st_r


if __name__ == "__main__":
    sB_max = 5
    # B_mu_st_r = np.random.rand(3, sB_max+1, 2*sB_max+1, 1000)

    # constructing a generic 3D magnetic field
    B = np.random.rand(3, 180, 360, 10)

    # creating the array for B_mu_st_r
    len_r = 10
    B_mu_st_r = np.zeros((3, sB_max+1, 2*sB_max+1, len_r))

    # creating a list of spherepy objects for each slice in radius
    # this is mainly done to be efficient in memory management
    B_r_mu_st = []

    # extracting the GSH components of the generic 3D B field one radial slice at a time
    for r_ind in range(B.shape[-1]):
        print(r_ind)
        B_r_mu_st.append(get_B_GSHcoeffs_from_B(B[:,:,:,r_ind]))

    # moving the radius dimension form the first to the very end
    B_mu_st_r = np.moveaxis(np.asarray(B_r_mu_st), 0, -1)

    # getting the BB GSH components from the B GSH components
    h_mu_nu_st_r = make_BB_GSH_from_B_GSH(B_mu_st_r, sB_max=sB_max)
    