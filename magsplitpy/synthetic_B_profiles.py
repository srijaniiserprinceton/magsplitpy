import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.pyplot as plt
plt.ion()

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

    
    def make_Bugnet2021_field(self, r, rho):
        lam   = 2.80
        R_rad = 0.136391

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
            x = np.linspace(r_in, r_out,1000)
            
            term1 = j1(x)
            term2 = rho_interp(x)
            term3 = x**3
            
            integral = simpson( term1 * term2 * term3 , x )
            return integral
            
        def Y_integral(r_in, r_out):
            x = np.linspace(r_in, r_out,1000)
            
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
        my_r_vals = np.logspace(np.log10(1e-4),np.log10(1),1000)

        # A_vec = np.vectorize(A)
        # A_array = A_vec(my_r_vals)
        for r_val in my_r_vals:
            A_array.append( A(r_val) )

        mask2 = np.where((my_r_vals[:] >= R_rad))
        A_array = np.array(A_array)
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

        plt.ylim([-0.05, 0.05])
        plt.xlim([0, 0.15])

        plt.legend()
        plt.grid()
        plt.tight_layout()

if __name__ == "__main__":
    make_B = synthetic_B()

    # reading a sample density profile
    r, rho = np.loadtxt('../sample_eigenfunctions/r_rho.txt').T
    B_Bugnet = make_B.make_Bugnet2021_field(r/r.max(), rho)

    make_B.plot_synthetic_B_field(B_Bugnet)