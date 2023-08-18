import os
import numpy as np
import json
from scipy.special import spherical_jn, spherical_yn
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.pyplot as plt
plt.ion()

os.environ["F90"] = "gfortran"
from avni.tools.bases import eval_splrem, eval_polynomial, eval_vbspl

NAX = np.newaxis

class synthetic_B:
    """
    Class to build the coefficients for simple B-field
    geometry from Das et al. 2020 as well as return the 
    Generalized Spherical Harmonic coefficients corresponding
    to these fields.
    """
    def __init__(self, Ntheta=180, Nphi=360, get_B_GSH=False):
        # grid in radius
        self.r = None

        # number of points in theta and phi directions
        self.Ntheta, self.Nphi = Ntheta, Nphi
        # constructing the theta and phi grids
        theta = np.linspace(0,np.pi,Ntheta)
        phi = np.linspace(0,2*np.pi-2*np.pi/Nphi,Nphi)

        self.thth, self.phph = np.meshgrid(theta, phi, indexing='ij')

        # whether to also return the GSH coefficients of the field
        self.get_B_GSH = get_B_GSH

        # rho is only needed for the profile of Bugnet 2021
        self.rho = None

        # bspline basis in radius (only computed if required)
        self.bsp_basis = None
        # this is supposed to be the radius within which we have high density of knots
        self.rshell = None  
        self.knot_ind_shell = None  
        self.knot_locs = None

    def dipolar_B(self):
        """
        Function to return the B field and GSH components for a purely dipolar field.

        Returns:
        --------
        B : ndarray
            The 3D array of the three vector components of a dipolar magnetic field.

        B_GSH_coefs : complex ndarray
                      The array of coefficients for the different GSH components of the 
                      dipolar magnetic field as proposed above Eqn.(D48) in Das et al 2020.
        """
        B = np.zeros((3, self.Ntheta, self.Nphi))

        # radial component
        B[0] += 2 * np.cos(self.thth)
        # theta component
        B[1] += np.sin(self.thth)
        # phi component
        B[2] += 0.0

        B_GSH_coefs = -1.0 * np.array([1.0, -2.0, 1.0]) / misc_funcs.gam(1.0)

        return B, B_GSH_coefs.astype('complex')

    def toroidal_B(self):
        """
        Function to return the B field and GSH components for a purely toroidal field.

        B : ndarray
            The 3D array of the three vector components of a toroidal magnetic field.

        B_GSH_coefs : complex ndarray
                      The array of coefficients for the different GSH components of the 
                      toroidal magnetic field as proposed above Eqn.(D48) in Das et al 2020.
        """
        B = np.zeros((3, self.Ntheta, self.Nphi))

        # radial component
        B[0] += 0.0
        # theta component
        B[1] += 0.0
        # phi component
        B[2] += -1.0 * np.sin(self.thth)

        B_GSH_coefs = -1j * np.array([1.0, 0.0, -1.0]) / misc_funcs.gam(1.0)

        return B, B_GSH_coefs.astype('complex')


    def make_B_D20(self, field_type='dipolar', alpha=1, beta=1):
        """
        Function to make the simple fields used Das et al 2020

        Parameters:
        -----------
        field_type : string, optional
                    The type of field configuration to be used: 'dipole',
                    'toroidal' or 'mixed'. The default is 'dipole'. For 'mixed',
                    the default is alpha = beta = 1. So, equal contribution from
                    both toroidal and dipolar.

        Returns:
        --------
        B : ndarray
            The 3D array of the three vector components of the magnetic field.

        B_GSH_coefs : complex ndarray
                      The array of coefficients for the different GSH components of the 
                      simple magnetic field as proposed in Eqn.(D48) of Das et al 2020.
        """

        match field_type:
            case 'dipolar':
                B, B_GSH_coefs = self.dipolar_B()

            case 'toroidal':
                B, B_GSH_coefs = self.toroidal_B()

            case 'mixed':
                # building the toroidal and dipolar components individually
                B1, B1_GSH_coefs = self.toroidal_B()
                B2, B2_GSH_coefs = self.dipolar_B()

                # constructing the complete field from the toroidal and dipolar components
                B = alpha * B1 + beta * B2
                B_GSH_coefs = alpha * B1_GSH_coefs + beta * B2_GSH_coefs

            case _: 
                print('Not a valid field type.')
                B, B_GSH_coefs = None, None

        if(self.get_B_GSH):
            return B, B_GSH_coefs
        else:
            return B

    
    def make_Bugnet2021_field(self, r, rho, B0=1.0, stretch_radius=False, toreturn1D=False, tointerpolate=False, r_interp=None):
        '''
        lam   = 2.80
        R_rad = 0.136391

        if(stretch_radius):
            r = r/R_rad
            R_rad = 1.0

        # rho interpolation function
        rho_interp = interp1d(r, rho)

        A_array = []
        Br_array = []

        def j1(r):
            arg = lam * r / R_rad
            return spherical_jn(1,arg)

        def y1(r):
            arg = lam * r / R_rad
            return spherical_yn(1,arg)

        def J_integral(r_in, r_out):
            x = np.linspace(r_in, r_out,len(r)-1)
            
            term1 = j1(x)
            term2 = rho_interp(x)
            term3 = x**3
            
            integral = simpson( term1 * term2 * term3 , x )
            return integral
            
        def Y_integral(r_in, r_out):
            x = np.linspace(r_in, r_out, len(r)-1)
            
            term1 = y1(x)
            term2 = rho_interp(x)
            term3 = x**3
    
            integral = simpson( term1 * term2 * term3 , x )
            return integral

        def A(r):
            term1a = - r * j1(r)
            term1b = Y_integral(r, R_rad)
            term1  = term1a * term1b
            
            term2a = - r * y1(r)
            term2b = J_integral(0, r)
            term2  = term2a * term2b
                
            return term1 + term2

        # making the radius grid
        my_r_vals = np.logspace(np.log10(1e-4),np.log10(1), len(r)-1)

        # A_vec = np.vectorize(A)
        # A_array = A_vec(my_r_vals)
        for r_val in my_r_vals:
            A_array.append( A(r_val) )

        mask2 = np.where((my_r_vals[:] >= R_rad))
        A_array = np.array(A_array)

        if(~stretch_radius):
            A_array[mask2] = 0.

        Br_array = 2 * A_array / my_r_vals**2
        Bt_array = - np.gradient( A_array , my_r_vals ) / my_r_vals
        Bp_array = lam/R_rad * A_array / my_r_vals

        Bt_array[0]=Bt_array[1]

        my_r_vals= np.append(0,my_r_vals)
        self.r = my_r_vals

        Br_array = np.append(Br_array[0],Br_array)
        Bt_array = np.append(Bt_array[0],Bt_array)
        Bp_array = np.append(Bp_array[0],Bp_array)

        # normalizing all components with the maximum 
        # in Br component
        Bt_array = Bt_array/np.max(Br_array)
        Bp_array = Bp_array/np.max(Br_array)
        Br_array = Br_array/np.max(Br_array)
        '''

        Bugnet_field_data = json.load(open('../tests/Field.json'))
        r, Br_array, Bt_array, Bp_array = np.asarray(Bugnet_field_data['r']),\
                                          B0 * np.asarray(Bugnet_field_data['Br']),\
                                          B0 * np.asarray(Bugnet_field_data['Bt']),\
                                          B0 * np.asarray(Bugnet_field_data['Bp'])
        self.r = r

        if(tointerpolate):
            Br_array = np.interp(r_interp, self.r, Br_array)
            Bt_array = np.interp(r_interp, self.r, Bt_array)
            Bp_array = np.interp(r_interp, self.r, Bp_array)

            self.r = r_interp

        if(toreturn1D):
            return np.array([Br_array, Bt_array, Bp_array])

        # introducing the theta and phi dependencies
        B = np.array([Br_array[NAX,NAX,:] * np.cos(self.thth)[:,:,NAX],
                      Bt_array[NAX,NAX,:] * np.sin(self.thth)[:,:,NAX],
                      Bp_array[NAX,NAX,:] * np.sin(self.thth)[:,:,NAX]])

        return B

    def plot_synthetic_B_field(self, B):
        plt.figure(figsize=(10,5))
        
        plt.plot(self.r, B[0], 'k', label='$B_r$')
        plt.plot(self.r, B[1], '--k', label='$B_{\\theta}$')
        plt.plot(self.r, B[2], ':k', label='$B_{\phi}$')

        # plt.ylim([-0.05, 0.05])
        plt.ylim([-1.2, 1.2])
        plt.xlim([0, 1])

        plt.legend()
        plt.grid()
        plt.tight_layout()

    def create_bsplines(self, custom_knot_num, rshell=2e-2):
        rmin, rmax = self.r.min(), self.r.max()
        self.rshell = rshell
        total_knot_num = int(np.round((rmax-rmin)/(1 - self.rshell))) \
                         * custom_knot_num
        total_knot_num += 4 - total_knot_num%4 - 1

        num_skip = len(self.r)//total_knot_num
        knot_locs_uniq = self.r[::num_skip][:total_knot_num-1]
        knot_locs_uniq = np.append(knot_locs_uniq, rmax)
        knot_ind_shell = np.argmin(abs(knot_locs_uniq - self.rshell))
        self.knot_ind_shell = knot_ind_shell - knot_ind_shell%4
        knotval_shell = knot_locs_uniq[self.knot_ind_shell]
        # print(f"knotlocsuniq shape = {knot_locs_uniq.shape}, {self.knot_ind_th}")

        knot_locs = np.hstack((knot_locs_uniq[:self.knot_ind_shell],
                               knot_locs_uniq[self.knot_ind_shell:]))
        self.knot_locs = knot_locs
        # print(f"knotlocs shape = {knot_locs.shape}")

        #----------ORIGINAL SPLINE CONFIGURATION FROM HELIOSEISMOLOGY-------------------#
        # vercof1, dvercof1 = eval_polynomial(self.r,
        #                                     [rmin, self.knot_locs[self.knot_ind_shell]],
        #                                     1, types= ['TOP','BOTTOM'])
        # vercof2, dvercof2 = eval_vbspl(self.r, knot_locs_uniq[:self.knot_ind_shell+1])
        # vercof3, dvercof3 = eval_polynomial(self.r,
        #                                     [self.knot_locs[self.knot_ind_shell], rmax],
        #                                     1, types= ['TOP','BOTTOM'])
        # vercof4, dvercof4 = eval_vbspl(self.r, knot_locs_uniq[self.knot_ind_shell:])
        
        # idx = np.where(vercof3[:, -1] > 0)[0][0]
        # vercof3[idx, -1] = 0.0

        # idx = np.where(vercof4[:, 0] > 0)[0][0]
        # vercof4[idx, 0] = 0.0

        # arranging the basis from left to right with st lines                                
        # bsp_basis = np.column_stack((vercof1[:, :],
        #                              vercof2[:, :],                                        
        #                              vercof3[:, :],
        #                              vercof4[:, :])) 

        #-----------SPLINE CONFIGURATION WITH NO BOUNDARY CONSIDERATION IN BETWEEN-------------#
        vercof1, dvercof1 = eval_polynomial(self.r, [rmin, rmax],
                                            1, types= ['TOP','BOTTOM'])
        vercof2, dvercof2 = eval_vbspl(self.r, knot_locs_uniq)

        idx = np.where(vercof1[:, -1] > 0)[0][0]
        vercof1[idx, -1] = 0.0                                         

        bsp_basis = np.column_stack((vercof1[:, :],
                                     vercof2[:, :]))

        # d_bsp_basis = np.column_stack((dvercof1[:, :],
        #                                dvercof2[:, :]))
        
        self.knot_ind_shell = self.knot_ind_shell + 4

        knot_locs = np.hstack((knot_locs_uniq[:self.knot_ind_shell+1],
                               knot_locs_uniq[self.knot_ind_shell:]))
        self.knot_locs = knot_locs

        # storing the analytically derived B-splines and it first derivatives
        # making them of shape (n_basis, r)
        self.bsp_basis = bsp_basis.T
        # self.d_bsp_basis = d_bsp_basis.T


    def get_radial_spline_coefs(self, B):
        # creating the carr corresponding to the DPT using custom knots
        Gtg = self.bsp_basis @ self.bsp_basis.T   # shape(n_basis, n_basis)

        # computing the coefficient arrays (c_arr)
        c_arr = np.tensordot(np.tensordot(B, self.bsp_basis, axes=([-1],[-1])),
                             np.linalg.inv(Gtg), axes=([-1],[-1]))

        # these coefficient array is of shape (3 x theta x phi x knots)
        return c_arr

    def reconstruct_field_from_spline(self, c_arr):
        B_rec = c_arr @ self.bsp_basis

        # reconstructed B has shape (3 x Ntheta x Nphi x len(r))
        return B_rec

    def interp_splines(self, r_new):
        bsp_basis_new = np.zeros((len(self.bsp_basis), len(r_new)))
        for i in range(len(self.bsp_basis)):
            bsp_basis_new[i] = np.interp(r_new, self.r, self.bsp_basis[i])

        return bsp_basis_new

    def plot_spline_basis(self):
        fig, ax = plt.subplots(1, 1, figsize=(10,6))

        for i in range(self.bsp_basis.shape[0]):
            ax.plot(self.r, self.bsp_basis[i])

        ax.set_xlim([0,1])

        plt.tight_layout()

    
    def compare_spline_efficiency(self, B_rec, B_true):
        plt.figure(figsize=(10,5))
        
        # plotting the true fueld
        plt.plot(self.r, B_true[0,0,0,:], 'k', label='$B_r$')
        plt.plot(self.r, B_true[1,90,0,:], 'k', label='$B_{\\theta}$')
        plt.plot(self.r, B_true[2,90,0,:], 'k', label='$B_{\phi}$')

        # plotting the reconstructed field
        plt.plot(self.r, B_rec[0,0,0,:], '--r', label='$B_r^{\mathrm{rec}}$')
        plt.plot(self.r, B_rec[1,90,0,:], '--r', label='$B_{\\theta}^{\mathrm{rec}}$')
        plt.plot(self.r, B_rec[2,90,0,:], '--r', label='$B_{\phi}^{\mathrm{rec}}$')

        # plt.ylim([-0.05, 0.05])
        plt.xlim([0, 1])

        plt.title(f'Number of spline knots: {self.knot_locs.shape[0]}')

        plt.legend()
        plt.grid()
        plt.tight_layout()

if __name__ == "__main__":
    make_B = synthetic_B()

    # reading a sample density profile
    r, rho = np.loadtxt('../sample_eigenfunctions/r_rho.txt').T
    B_Bugnet_1D = make_B.make_Bugnet2021_field(r/r.max(), rho, stretch_radius=True, toreturn1D=True)

    # plotting the field
    make_B.plot_synthetic_B_field(B_Bugnet_1D)

    B_Bugnet_3D = make_B.make_Bugnet2021_field(r/r.max(), rho, stretch_radius=True)

    # extracting the spline coefficients of the B-field for faster B_st(r) computation
    make_B.create_bsplines(50)
    c_arr = make_B.get_radial_spline_coefs(B_Bugnet_3D)

    # plotting to see how the B-spline basis looks like
    make_B.plot_spline_basis()

    # reconstructing B
    B_rec = make_B.reconstruct_field_from_spline(c_arr)

    # plotting the reconstructed B
    make_B.compare_spline_efficiency(B_rec, B_Bugnet_3D)