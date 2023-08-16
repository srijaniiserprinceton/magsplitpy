import numpy as np
import h5py
from tqdm import tqdm

from magsplitpy import misc_funcs as fn
from magsplitpy import synthetic_B_profiles as B_profiles
from magsplitpy import mag_GSH_funcs

NAX = np.newaxis

def get_h_specific_Bs0t0(B_mu_s0t0_r, sB, tB, sBB_arr, tBB_arr):
    # mu and un matrices
    mu, nu = np.arange(-1, 2), np.arange(-1, 2)

    mumu, nunu, ss, tt = np.meshgrid(mu, nu, sBB_arr, tBB_arr, indexing='ij')

    h_mu_nu_st_r_specific = ((-1)**(np.abs(mumu + nunu + tt)) *\
                            (2 * sB + 1) * np.sqrt((2*ss + 1) / (4 * np.pi)) *\
                            fn.w3j_vecm(sB, ss, sB, mumu, -(mumu+nunu), nunu) *\
                            fn.w3j_vecm(sB, ss, sB, tB, tt, tB))[:,:,:,:,NAX] *\
                            (B_mu_s0t0_r[:, NAX, :] * B_mu_s0t0_r[NAX, :, :])[:,:,NAX,NAX,:]

    return h_mu_nu_st_r_specific


if __name__ == '__main__':
    # loading the corresponding file
    eigfile = h5py.File(f'../Vincent_Eig/mode_h.2_-2_hz.h5')
    # the radius grid
    r_norm_Rstar = eigfile['x'][()]

    # getting a t = 0 magnetic field from Bugnet 2021
    make_B = B_profiles.synthetic_B()
    r, rho = np.loadtxt('../sample_eigenfunctions/r_rho.txt').T
    B = make_B.make_Bugnet2021_field(r/r.max(), rho, B0=1, stretch_radius=True,
                                     tointerpolate=False, r_interp=r_norm_Rstar)
    

    # choosing a low number of splines since this is just a unit test
    make_B.create_bsplines(5)
    c_arr = make_B.get_radial_spline_coefs(B)

    # making h_mu_nu_st_r
    num_knots = c_arr.shape[-1]
    sB_max = 1

    # empty list for magnetic field coefficients
    B_j_mu_st = []

    print("Extracting the VSH coefficients for the splined B.")
    # extracting the GSH components of the generic 3D B field one radial slice at a time
    for knot_ind in tqdm(range(num_knots)):
        Bcoefs_numerical = mag_GSH_funcs.get_B_GSHcoeffs_from_B(c_arr[:,:,:,knot_ind],
                                                                nmax=sB_max,mmax=sB_max)
        B_j_mu_st.append(Bcoefs_numerical)

    # moving the radius dimension form the first to the very end
    B_mu_st_j = np.moveaxis(np.asarray(B_j_mu_st), 0, -1)
    ___, len_s, len_t, len_knots = B_mu_st_j.shape
    B_mu_st_j += np.random.rand(3, len_s, len_t, len_knots)

    #converting back to radius in grid
    print("Reconstructing the B_mu_st_r from the spline coefficients.")
    B_mu_st_r = make_B.reconstruct_field_from_spline(B_mu_st_j)

    # getting the dimension length of s and t for B_mu_st_r
    len_sB = B_mu_st_r.shape[1]
    sB_arr = np.arange(0, len_sB)
    tmaxB_arr = np.arange(-sB_arr.max(), sB_arr.max() + 1)

    # creating the sBB and tBB arrays
    sBB_arr = np.arange(0, 2 * sB_arr.max() + 1)
    tBB_arr = np.arange(-sBB_arr.max(), sBB_arr.max() + 1)

    for sB_ind, sB in enumerate(sB_arr):
        tB_arr = np.arange(-sB, sB + 1)
        for tB_ind, tB in enumerate(tB_arr):
            print(f"s0 = {sB}, t0 = {tB}")
            print("==============")
            t_ind_in_B_mu_st_r = np.argmin(np.abs(tmaxB_arr - tB))
            B_mu_st_r_specific_s0t0 = np.zeros_like(B_mu_st_r)
            B_mu_st_r_specific_s0t0[:, sB_ind, t_ind_in_B_mu_st_r, :] =\
                            B_mu_st_r[:, sB_ind, t_ind_in_B_mu_st_r, :]
            B_mu_s0t0_r = B_mu_st_r[:, sB_ind, t_ind_in_B_mu_st_r, :]
            # getting the BB GSH components from the B GSH components
            print('Start h_mu_nu_st_r computation')
            h_mu_nu_st_r = mag_GSH_funcs.make_BB_GSH_from_B_GSH(B_mu_st_r_specific_s0t0, sB_max=sB_max)
            print('End h_mu_nu_st_r computation\n')
            # generating the h_mu_nu_st_r for a specfic s and t of B
            h_mu_nu_st_r_specific = get_h_specific_Bs0t0(B_mu_s0t0_r, sB, tB, sBB_arr, tBB_arr)
            # and comparing it with the generric h_mu_nu_st_r generation

            np.testing.assert_array_almost_equal(h_mu_nu_st_r, h_mu_nu_st_r_specific)

            print(f"h_general_abssum : {np.sum(np.abs(h_mu_nu_st_r))}")
            print(f"h_specific_abssum : {np.sum(np.abs(h_mu_nu_st_r_specific))}")
