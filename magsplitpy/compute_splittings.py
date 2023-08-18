import numpy as np
import scipy.integrate as integrate
import h5py
import sys
import re
import os
from statistics import mode
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()

from magsplitpy import synthetic_B_profiles as B_profiles
from magsplitpy import magkerns
from magsplitpy import mag_GSH_funcs
from magsplitpy import misc_funcs as fn
from magsplitpy import rotate_splitting_basis as rotate

def compute_mag_coupling(kern, h, t_arr, r):
    # reading off the m array from the dimension of kern
    m_max_kern = kern.shape[3]//2
    m = np.arange(-m_max_kern, m_max_kern + 1)
    m_ = m

    # creating the m'-m matrix to find t_index
    mm, mm_ = np.meshgrid(m, m_, indexing='ij')
    mm__minus_mm = mm_ - mm

    # the array containing the index of t corresponding to m' and m
    mm__mm_mesh_2_t_map = np.zeros_like(mm__minus_mm)
    # filling in the array according to the selection rule t=m'-m
    for i in range(len(m_)):
        for j in range(len(m)):
            t_ind = np.argmin(np.abs(mm__minus_mm[i,j] - t_arr))
            mm__mm_mesh_2_t_map[i,j] = t_ind

    # h_mm__ should be of shape mu x nu x s x m x m' x r
    h_mm__ = h[:,:,:,mm__mm_mesh_2_t_map,:] 

    # Zmag = supermatrix for magnetism
    # integrand is the sum over mu, nu, s
    Z_integrand = np.sum(kern * h_mm__, axis=(0,1,2))

    # Zmag should have shape m x m' (for self coupling)
    Z = integrate.simpson(Z_integrand, r, axis=(-1))

    return Z

def plot_splittings_vs_freq(dom_ell1_m0, dom_ell1_m1, dom_ell2_m0, dom_ell2_m1, dom_ell2_m2):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    ax.plot(dom_ell1_m1[:,0], dom_ell1_m1[:,1], marker='*', color='black', label='$\ell = 1, m = 1$',linestyle = 'None')
    ax.plot(dom_ell1_m0[:,0], dom_ell1_m0[:,1], marker='*', color='grey', label='$\ell = 1, m = 0$',linestyle = 'None')
    ax.plot(dom_ell2_m2[:,0], dom_ell2_m2[:,1], marker='o', color='lime', label='$\ell = 2, m = 2$',linestyle = 'None')
    ax.plot(dom_ell2_m1[:,0], dom_ell2_m1[:,1], marker='o', color='green', label='$\ell = 2, m = 1$',linestyle = 'None')
    ax.plot(dom_ell2_m0[:,0], dom_ell2_m0[:,1], marker='o', color='red', label='$\ell = 2, m = 0$',linestyle = 'None')

    plt.legend()
    ax.set_xlim([0, 9])
    ax.set_ylim([0, 0.052])
    ax.set_xlabel('$\\nu_{n,\ell}$ in $d^{-1}$', fontsize=16)
    ax.set_ylabel('$\delta \\nu_{\mathrm{mag}}$ in $d^{-1}$', fontsize=16)
    plt.tight_layout()

if __name__ == "__main__":
    # loading the file with the number of radius point which appears the most often
    # in the set of multiplet files
    # accumulating the labels for all the files to be read
    filenames = np.asarray(os.listdir('../Vincent_Eig/mode_files/'))
    len_r = []

    for f in filenames:
        n_str = re.split('[_]+', f, flags=re.IGNORECASE)[2]
        ell_str = re.split('[.]+', re.split('[_]+', f, flags=re.IGNORECASE)[1])[1]
        eigfile = h5py.File(f'../Vincent_Eig/mode_files/mode_h.{ell_str}_{n_str}_hz.h5')
        len_r.append(eigfile['x'][()].shape[0])

    len_r = np.asarray(len_r)
    mode_len_r = mode(len_r)
    
    for f in filenames:
        n_str = re.split('[_]+', f, flags=re.IGNORECASE)[2]
        ell_str = re.split('[.]+', re.split('[_]+', f, flags=re.IGNORECASE)[1])[1]
        eigfile = h5py.File(f'../Vincent_Eig/mode_files/mode_h.{ell_str}_{n_str}_hz.h5')
        if(eigfile['x'][()].shape[0] == mode_len_r): break

    # the radius and rho grid for most multiplets
    r_norm_Rstar = eigfile['x'][()]   # reading it off a random file since its the same for all
    rho = eigfile['rho'][()]

    #--------------obtaining the Lorentz-stress GSh components for
    # a given magnetic field-------------------------------------#
    make_B = B_profiles.synthetic_B()
    r, rho = np.loadtxt('../sample_eigenfunctions/r_rho.txt').T
    B = make_B.make_Bugnet2021_field(r/r.max(), rho, B0=5e4, stretch_radius=True,
                                     tointerpolate=True, r_interp=r_norm_Rstar)

    print("3D magnetic field loaded.")

    # decomposing B into splines
    print("Decomposing B field into splines in radius.")
    make_B.create_bsplines(50)
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

    #converting back to radius in grid
    print("Reconstructing the B_mu_st_r from the spline coefficients.")
    B_mu_st_r = make_B.reconstruct_field_from_spline(B_mu_st_j)

    # getting the BB GSH components from the B GSH components
    print('Start h_mu_nu_st_r computation')
    h_mu_nu_st_r = mag_GSH_funcs.make_BB_GSH_from_B_GSH(B_mu_st_r, sB_max=sB_max)
    print('End h_mu_nu_st_r computation')

    # the t array for the largest range of t
    tmax = h_mu_nu_st_r.shape[3] // 2
    t_arr = np.arange(-tmax, tmax+1)

    # creating lists of delta_omega_mag with dummy initializations
    delta_omega_mag_ell1_m0 = np.asarray([[0, 0]])
    delta_omega_mag_ell1_m1 = np.asarray([[0, 0]])
    delta_omega_mag_ell2_m0 = np.asarray([[0, 0]])
    delta_omega_mag_ell2_m1 = np.asarray([[0, 0]])
    delta_omega_mag_ell2_m2 = np.asarray([[0, 0]])

    # running a loop over all the multiplets we are interested in
    for i in tqdm(range(len(filenames))):
        # desired mode
        n_str = re.split('[_]+', filenames[i], flags=re.IGNORECASE)[2]
        ell_str = re.split('[.]+', re.split('[_]+', filenames[i], flags=re.IGNORECASE)[1])[1]

        #---------------------making the kernel----------------------#
        # loading the corresponding file
        eigfile = h5py.File(f'../Vincent_Eig/mode_files/mode_h.{ell_str}_{n_str}_hz.h5')
        # parameters of the star
        R_star = eigfile.attrs['R_star']
        freq_nl = eigfile.attrs['freq'][0]

        # converting freq_nl to omega_nl
        omega_nl = 2 * np.pi * freq_nl

        # the radius grid
        r_norm_Rstar = eigfile['x'][()]   # reading it off a random file since its the same for all
        rho = eigfile['rho'][()]
        
        # loading eigenfunctions here
        Ui_raw = eigfile['xi_r']['re'][()]
        Vi_raw = eigfile['xi_h']['re'][()]
        # normalizing eigenfunctions at the outset
        eignorm = fn.eignorm(Ui_raw, Vi_raw, int(ell_str), r_norm_Rstar, rho)
        Ui_raw = Ui_raw / eignorm
        Vi_raw = Vi_raw / eignorm

        # converting n and ell back to integers
        n, ell = int(n_str), int(ell_str)

        # the desired s values
        s = np.arange(0, 3)

        # initializing the kernel class for a specific field geometry and a radial grid
        make_kern_s = magkerns.magkerns(s, r_norm_Rstar)

        # calling the generic kernel computation
        kern = make_kern_s.ret_kerns(n, ell, np.arange(-ell,ell+1), Ui_raw, Vi_raw)
        # calling the axisymmetric field kernel computation
        # kern = make_kern_s.ret_kerns_axis_symm(n, ell, np.arange(-ell,ell+1), Ui_raw, Vi_raw)

        # changing the kernel to mu x nu shape in first two dimensions
        kern = np.asarray(kern)
        kern_mu_nu = np.zeros((3, 3, 2*ell+1, 2*ell+1, len(s), len(r_norm_Rstar)))
        kern_mu_nu[0,0] = kern[0]
        kern_mu_nu[0,1] = kern[1]
        kern_mu_nu[0,2] = kern[3]
        kern_mu_nu[1,1] = kern[2]
        kern_mu_nu[1,2] = kern[4]
        kern_mu_nu[2,2] = kern[5]

        # filling in the other half by virtue of symmetry
        kern_mu_nu[1,0] = kern[1]
        kern_mu_nu[2,0] = kern[3]
        kern_mu_nu[2,1] = kern[4]

        # # for now cropping the size of radius [FOR FASTER TESTING]
        # kern_mu_nu = kern_mu_nu[:,:,:,:,:,:]

        # changing from shape (mu,nu,m.m_,sBB,r) to (mu,nu,sBB,m,m_,r)
        kern_mu_nu = np.moveaxis(kern_mu_nu, -2, 2)

        # interpolating the magnetic coefficients onto the grid of eigenfunctions, if needed
        if(len(make_B.r) == len(r_norm_Rstar)):
            h_mu_nu_st_r_interp = 1.0 * h_mu_nu_st_r
        else:
            bsp_basis_interp = make_B.interp_splines(r_norm_Rstar)
            B_mu_st_r = B_mu_st_j @ bsp_basis_interp
            h_mu_nu_st_r_interp = mag_GSH_funcs.make_BB_GSH_from_B_GSH(B_mu_st_r, sB_max=sB_max)

        #-----------finding the frequency splittings for the give mode
        # induced by the given magnetic field--------------------------#
        Z_mag = compute_mag_coupling(kern_mu_nu, h_mu_nu_st_r_interp, t_arr, r_norm_Rstar)

        # finding frequency splittings in omega (units radian per seconds)
        # the division with R_star^2 is to account for dimensions of length in numerator and denominator
        Z_mag = Z_mag / (2 * omega_nl * R_star**2)

        # rotating by 75 degrees
        D = rotate.Wigner_d_matrix(ell, 75. * np.pi / 180.)

        Z_mag = D @ Z_mag @ D.T

        # frequency splittings in the unrotated frame in day^{-1}.
        # The 2pi division is to convert from omega to nu.
        delta_omega_mag = np.diag(Z_mag).real * 3600 * 24 / (2 * np.pi)

        # converting to day^{-1}
        freq_nl *= 3600 * 24

        if(ell == 1):
            delta_omega_mag_ell1_m0 = np.append(delta_omega_mag_ell1_m0, np.array([[freq_nl, delta_omega_mag[1]]]), axis=0)
            delta_omega_mag_ell1_m1 = np.append(delta_omega_mag_ell1_m1, np.array([[freq_nl, delta_omega_mag[2]]]), axis=0)
        else:
            delta_omega_mag_ell2_m0 = np.append(delta_omega_mag_ell2_m0, np.array([[freq_nl, delta_omega_mag[2]]]), axis=0)
            delta_omega_mag_ell2_m1 = np.append(delta_omega_mag_ell2_m1, np.array([[freq_nl, delta_omega_mag[3]]]), axis=0)
            delta_omega_mag_ell2_m2 = np.append(delta_omega_mag_ell2_m2, np.array([[freq_nl, delta_omega_mag[4]]]), axis=0)
        
    # removing the first dummy entries [0,0]
    delta_omega_mag_ell1_m0 = delta_omega_mag_ell1_m0[1:]
    delta_omega_mag_ell1_m1 = delta_omega_mag_ell1_m1[1:]
    delta_omega_mag_ell2_m0 = delta_omega_mag_ell2_m0[1:]
    delta_omega_mag_ell2_m1 = delta_omega_mag_ell2_m1[1:]
    delta_omega_mag_ell2_m2 = delta_omega_mag_ell2_m2[1:]
    
    # plotting the frequencies
    plot_splittings_vs_freq(delta_omega_mag_ell1_m0, delta_omega_mag_ell1_m1,\
                            delta_omega_mag_ell2_m0, delta_omega_mag_ell2_m1, delta_omega_mag_ell2_m2)