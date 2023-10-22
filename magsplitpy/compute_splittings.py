import numpy as np
import scipy.integrate as integrate
import sys
import os
import re
import h5py
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
plt.ion()

font = {'size'   : 20}
rc('font', **font)

from magsplitpy import synthetic_B_profiles as B_profiles
from magsplitpy import magkerns
from magsplitpy import mag_GSH_funcs
from magsplitpy import misc_funcs as fn
from magsplitpy import rotate_splitting_basis as rotate

def delta_nu_mag(B_class_object, dir_eigfiles, mag_obliquity=0.0, freq_units='muHz'):
    """
    Front-end function to calculate freqeuncy splittings.

    Paramters:
    ----------
    B_class_object: object of type synthetic_B from script synthetic_B_profiles
                    Contains miscellaneous information about the synthetic model
                    magnetic field.
    dir_eigfiles: str
                  Path to directory containing the eigenfunctions for model star.
    mag_obliquity: float, optional
                   Relative inclination of the axisymmetric magnetic field
                   with respect to rotation axis. Uses Loi 2021 formula.
                   Default value is set at 0.0 (aligned magnetic and rotation axis).
    freq_units: str, optional
                Units in which to output frequency splittings. Currently setup
                for microhertz 'muHz' and day inverse 'day_inv'. Default is 'muHz'.
    """
    # the most commmon grid in radius
    r_grid_common = B_class_object.r

    # getting the sBB_max (the counting of sBB starts from zero so subtracting 1)
    sBB_max = B_class_object.h_mu_nu_st_r.shape[2] - 1

    # the t array for the largest range of t
    tmax = B_class_object.h_mu_nu_st_r.shape[3] // 2
    t_arr = np.arange(-tmax, tmax+1)

    # creating lists of delta_omega_mag with dummy initializations
    delta_omega_mag_ell1_m0 = np.asarray([[0, 0]])
    delta_omega_mag_ell1_m1 = np.asarray([[0, 0]])
    delta_omega_mag_ell2_m0 = np.asarray([[0, 0]])
    delta_omega_mag_ell2_m1 = np.asarray([[0, 0]])
    delta_omega_mag_ell2_m2 = np.asarray([[0, 0]])

    # checking the units in which we want the frequency and splittings
    # default unit is Hz
    if(freq_units == 'muHz'):
        unit_fac = 1e6
    if(freq_units == 'day_inv'):
        unit_fac = 3600 * 24

    filenames = np.asarray(os.listdir(f'{dir_eigfiles}/'))

    # running a loop over all the multiplets we are interested in
    for i in tqdm(range(len(filenames))):
        #---------------------making the kernel----------------------#
        # loading the corresponding file
        # eigfile = h5py.File(f'{dir_eigfiles}/mode_h.{ell_str}_{n_str}_hz.h5')
        eigfile = h5py.File(f'{dir_eigfiles}/' + filenames[i])

        # desired mode (read from metadata or GYRE file)
        n_str = str(eigfile.attrs['n_pg'])
        ell_str = str(eigfile.attrs['l'])

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

        # the desired s values for BB [0, ...., sBB_max]
        s = np.arange(0, sBB_max+1)

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
        if(len(r_grid_common) == len(r_norm_Rstar)):
            h_mu_nu_st_r_interp = 1.0 * B_class_object.h_mu_nu_st_r
        else:
            bsp_basis_interp = B_class_object.interp_splines(r_norm_Rstar)
            B_mu_st_r = B_class_object.B_mu_st_j @ bsp_basis_interp
            h_mu_nu_st_r_interp = mag_GSH_funcs.make_BB_GSH_from_B_GSH(B_mu_st_r, sB_max=make_B.sB_max)

        #-----------finding the frequency splittings for the give mode
        # induced by the given magnetic field--------------------------#
        Z_mag = compute_mag_coupling(kern_mu_nu, h_mu_nu_st_r_interp, t_arr, r_norm_Rstar)

        # finding frequency splittings in omega (units radian per seconds)
        # the division with R_star^2 is to account for dimensions of length in numerator and denominator
        Z_mag = Z_mag / (2 * omega_nl * R_star**2)

        # rotating by "mag_obliquity" degrees
        D = rotate.Wigner_d_matrix(ell, mag_obliquity * np.pi / 180.)

        # new rotated supermatrix as per Eqn(22) of Loi 2021
        Z_mag = D @ Z_mag @ D.T

        # frequency splittings in the unrotated frame in day^{-1}.
        # The 2pi division is to convert from omega to nu.
        delta_omega_mag = np.diag(Z_mag).real * unit_fac / (2 * np.pi)

        # converting to day^{-1}
        freq_nl *= unit_fac

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

    # returning magnetic frequency splittings in day^{-1}
    return delta_omega_mag_ell1_m0, delta_omega_mag_ell1_m1,\
           delta_omega_mag_ell2_m0, delta_omega_mag_ell2_m1, delta_omega_mag_ell2_m2


def compute_mag_coupling(kern, h, t_arr, r):
    """
    Function that computes the coupling matrix from the kernel,
    and Lorentz-stress tensor [Eqn(18) of Das et al 2020].

    Parameters:
    -----------
    kern: array-like of floats, shape (mu,nu,sBB,m,m_,r)
          Lorentz-stress kernels computed in delta_mu_nu()
    h: array-like of floats, shape (mu,nu,s,t,r)
       Lorentz-stress tensor in the GSH form
    t_arr: array-like of ints, shape 2*sBB_max + 1
           Contains the array of (-sBB_max, sBB_max).
    r: array-like of floats, shape (r,)
       Radial grid to integrate over.

    Returns:
    --------
    Z: array-like of comples128, shape m x m_
       Submatrix for the specific multiplet for magnetic perturbation.
    """
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

# function to plot the frequency splittings of different modes as a function of unperturbed frequencies
def plot_splittings_vs_freq(dom_ell1_m0, dom_ell1_m1, dom_ell2_m0, dom_ell2_m1, dom_ell2_m2, freq_units='muHz'):
    fig, ax = plt.subplots(1, 1, figsize=(10,7))

    ax.plot(dom_ell1_m1[:,0], dom_ell1_m1[:,1], marker='*', color='black', label='$\ell = 1, m = 1$',linestyle = 'None')
    ax.plot(dom_ell1_m0[:,0], dom_ell1_m0[:,1], marker='*', color='grey', label='$\ell = 1, m = 0$',linestyle = 'None')
    ax.plot(dom_ell2_m2[:,0], dom_ell2_m2[:,1], marker='o', color='lime', label='$\ell = 2, m = 2$',linestyle = 'None')
    ax.plot(dom_ell2_m1[:,0], dom_ell2_m1[:,1], marker='o', color='green', label='$\ell = 2, m = 1$',linestyle = 'None')
    ax.plot(dom_ell2_m0[:,0], dom_ell2_m0[:,1], marker='o', color='red', label='$\ell = 2, m = 0$',linestyle = 'None')

    plt.legend()
    # ax.set_xlim([0, 9])
    # ax.set_ylim([0, 0.052])
    if(freq_units == 'muHz'):
        ax.set_xlabel('$\\nu_{n,\ell}$ in $\mu$Hz')
        ax.set_ylabel('$\delta \\nu_{\mathrm{mag}}$ in $\mu$Hz')
    if(freq_units == 'day_inv'):
        ax.set_xlabel('$\\nu_{n,\ell}$ in $d^{-1}$')
        ax.set_ylabel('$\delta \\nu_{\mathrm{mag}}$ in $d^{-1}$')

    plt.tight_layout()

# function originally written to compare the difference in splitting between Das and Bugnet codes
def plot_splittings_vs_freq_compare(dom_ell1_m0, dom_ell1_m1, dom_ell2_m0, dom_ell2_m1, dom_ell2_m2, freq_units='muHz'):
    fig, ax = plt.subplots(1, 1, figsize=(10,7))

    ax.plot(dom_ell1_m1[:,0], dom_ell1_m1[:,1]*100, marker='*', color='black', label='$\ell = 1, m = 1$',linestyle = 'None')
    ax.plot(dom_ell1_m0[:,0], dom_ell1_m0[:,1]*100, marker='*', color='grey', label='$\ell = 1, m = 0$',linestyle = 'None')
    ax.plot(dom_ell2_m2[:,0], dom_ell2_m2[:,1]*100, marker='o', color='lime', label='$\ell = 2, m = 2$',linestyle = 'None')
    ax.plot(dom_ell2_m1[:,0], dom_ell2_m1[:,1]*100, marker='o', color='green', label='$\ell = 2, m = 1$',linestyle = 'None')
    ax.plot(dom_ell2_m0[:,0], dom_ell2_m0[:,1]*100, marker='o', color='red', label='$\ell = 2, m = 0$',linestyle = 'None')

    plt.legend()
    ax.set_xlim([0, 9])
    # ax.set_ylim([0, 0.052])
    if(freq_units == 'muHz'):
        ax.set_xlabel('$\\nu_{n,\ell}$ in $\mu$Hz')
    if(freq_units == 'day_inv'):
        ax.set_xlabel('$\\nu_{n,\ell}$ in $d^{-1}$')
    ax.set_ylabel('$(\delta \\nu_{\mathrm{mag}} - \delta \\nu_{\mathrm{mag}}^{\mathrm{Lisa}}) / \delta \\nu_{\mathrm{mag}}^{\mathrm{Lisa}} \\times$ 100')
    plt.tight_layout()

if __name__ == "__main__":
    # path to the directory containing the eigfiles
    dir_eigfiles = '../Vincent_Eig/mode_files'

    # extracting the most common r and rho grid
    r_norm_Rstar, rho = fn.find_mode_r_grid(dir_eigfiles)

    # units in which we want the splittings
    # currently setup for 'muHz' or 'day_inv'
    freq_units = 'day_inv'

    #--------------obtaining the Lorentz-stress GSh components for
    # a given magnetic field-------------------------------------#
    B_field_type = 'Bugnet_2021'
    B0 = 50000
    user_knot_num = 50
    sB_max = 1
    mag_obliquity = 75.0    # tilt of the magnetic axis wrt the rotation axis in degrees
    print("2. Loading 3D magnetic field and decomposing into B-splines in radius.")
    make_B = B_profiles.synthetic_B(r_norm_Rstar, rho=rho, custom_knot_num=user_knot_num,
                                    B0=B0, sB_max=sB_max, field_type=B_field_type)

    
    print("6. Computing kernels and splittings.")
    delta_omega_mag_ell1_m0, delta_omega_mag_ell1_m1,\
    delta_omega_mag_ell2_m0, delta_omega_mag_ell2_m1, delta_omega_mag_ell2_m2 = delta_nu_mag(make_B, dir_eigfiles,
                                                                mag_obliquity=mag_obliquity, freq_units=freq_units)
    
    # plotting the frequencies
    plot_splittings_vs_freq(delta_omega_mag_ell1_m0, delta_omega_mag_ell1_m1,\
                            delta_omega_mag_ell2_m0, delta_omega_mag_ell2_m1, delta_omega_mag_ell2_m2, freq_units=freq_units)

    
    #------------------------------main template code ends here---------------------------#


    '''
    # comparing with Lisa's splittings (after ordering)
    delta_omega_mag_ell1_m0_L, delta_omega_mag_ell1_m1_L = np.load('../Vincent_Eig/Lisa_splittings_ordered_ell1.npy')
    delta_omega_mag_ell2_m0_L, delta_omega_mag_ell2_m1_L, delta_omega_mag_ell2_m2_L =\
                                                           np.load('../Vincent_Eig/Lisa_splittings_ordered_ell2.npy')
    
    B_key = 'Bp_only_new'
    splittings_Lisa = json.load(open(f'../tests/Splittings_model56_l1_{B_key}.json'))
    freq_Srijan = delta_omega_mag_ell1_m0[np.argsort(delta_omega_mag_ell1_m0[:,0]),1]
    freq_Lisa = np.asarray(splittings_Lisa['shifting_m0'])[np.argsort(np.asarray(splittings_Lisa['f_modes']))]
    plt.plot(splittings_Lisa['f_modes'], splittings_Lisa['shifting_m0'])
    plt.savefig(f'../plots/{B_key}_splittings.pdf')

    # plotting the percent error to compare the two methods
    percent_diff = 100 * (freq_Srijan - freq_Lisa)/freq_Lisa
    freq = delta_omega_mag_ell1_m0[np.argsort(delta_omega_mag_ell1_m0[:,0]),0]
    plt.figure()
    plt.plot(freq, percent_diff, 'k')
    plt.title(f'{B_key}')
    plt.savefig(f'../plots/{B_key}_percent_diff.pdf')
    '''

    # delta_omega_mag_ell1_m0_L, delta_omega_mag_ell1_m1_L = splittings_Lisa['shifting_m0'], splittings_Lisa['shifting_m1']
    # delta_omega_mag_ell2_m0_L, delta_omega_mag_ell2_m1_L, delta_omega_mag_ell2_m2_L =\
    #                                     delta_omega_mag_ell2_m0, delta_omega_mag_ell2_m1, delta_omega_mag_ell2_m2
                                                           

    # # plotting the frequencies from Lisa
    # plot_splittings_vs_freq(delta_omega_mag_ell1_m0_L, delta_omega_mag_ell1_m1_L,\
    #                         delta_omega_mag_ell2_m0_L, delta_omega_mag_ell2_m1_L, delta_omega_mag_ell2_m2_L, key='Lisa')

    # # comparing with Lisa's frequencies
    # freq_diff_ell1_m0 = np.asarray([delta_omega_mag_ell1_m0[:,0],\
    #                     (delta_omega_mag_ell1_m0 - delta_omega_mag_ell1_m0_L)[:,1] / delta_omega_mag_ell1_m0_L[:,1]])
    # freq_diff_ell1_m1 = np.asarray([delta_omega_mag_ell1_m1[:,0],\
    #                     (delta_omega_mag_ell1_m1 - delta_omega_mag_ell1_m1_L)[:,1] / delta_omega_mag_ell1_m1_L[:,1]])
    # freq_diff_ell2_m0 = np.asarray([delta_omega_mag_ell2_m0[:,0],\
    #                     (delta_omega_mag_ell2_m0 - delta_omega_mag_ell2_m0_L)[:,1] / delta_omega_mag_ell2_m0_L[:,1]])
    # freq_diff_ell2_m1 = np.asarray([delta_omega_mag_ell2_m1[:,0],\
    #                     (delta_omega_mag_ell2_m1 - delta_omega_mag_ell2_m1_L)[:,1] / delta_omega_mag_ell2_m1_L[:,1]])
    # freq_diff_ell2_m2 = np.asarray([delta_omega_mag_ell2_m2[:,0],\
    #                     (delta_omega_mag_ell2_m2 - delta_omega_mag_ell2_m2_L)[:,1] / delta_omega_mag_ell2_m2_L[:,1]])

    # plot_splittings_vs_freq_compare(freq_diff_ell1_m0.T, freq_diff_ell1_m1.T, freq_diff_ell2_m0.T, freq_diff_ell2_m1.T, freq_diff_ell2_m2.T)

    