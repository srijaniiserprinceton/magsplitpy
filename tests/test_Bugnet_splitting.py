# this code is to test if we recover the same GSH components 
# of Lorentz stress as the analytically calculated values in Das et al 2020.
# see Eqn.(D48) for the details of the coefficients.

import numpy as np

from magsplitpy import misc_funcs
from magsplitpy import mag_GSH_funcs

class B_field_D20:
    """
    Class to build the coefficients for simple B-field
    geometry from Das et al. 2020 as well as return the 
    Generalized Spherical Harmonic coefficients corresponding
    to these fields.
    """
    def __init__(self, Ntheta=180, Nphi=360):
        self.Ntheta, self.Nphi = Ntheta, Nphi
        theta = np.linspace(0,np.pi,Ntheta)
        phi = np.linspace(0,2*np.pi-2*np.pi/Nphi,Nphi)

        self.thth, self.phph = np.meshgrid(theta, phi, indexing='ij')

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

        return B, B_GSH_coefs

    def get_GSH_numerically(self, B):
        """
        Function to extract the GSH coefficients from the numerical
        calculation in magsplitpy.
        """
        B_GSH_coefs_numr = mag_GSH_funcs.get_B_GSHcoeffs_from_B(B)

        return B_GSH_coefs_numr

def compare_coefs_analytical_numerical(Bcoefs_analytical, Bcoefs_numerical_full):
    """
    Function to test the analytical and numerical coefficients.
    We can only compare the s=1, t=0 component of the numerical coeffients
    with the analytical coefficients.

    Parameters:
    -----------
    Bcoefs_analytical : complex ndarray, shape (3,)
                        Array containing the coefficients for s=1, t=0 
                        component of the simple field constructed from
                        Das et al. 2020.

    Bcoefs_numerical_full : 3 x spherepy.ScalarCoefs type
                            Coefficients of type spherepy.ScalarCoefs for
                            the three components of Generalized Spherical
                            Harmonic basis. It is sliced to extract the 
                            s=1, t=0 component to compare with
                            Bcoefs_analytical.
    """
    # extracting the s=1,t=0 component
    len_t_dim = Bcoefs_numerical_full.shape[-1]
    Bcoefs_numerical = np.array([0,0,0], dtype='complex')
    Bcoefs_numerical[0] += Bcoefs_numerical_full[0][1][len_t_dim//2]
    Bcoefs_numerical[1] += Bcoefs_numerical_full[1][1][len_t_dim//2]
    Bcoefs_numerical[2] += Bcoefs_numerical_full[2][1][len_t_dim//2]

    np.testing.assert_array_almost_equal(Bcoefs_analytical,
                                         Bcoefs_numerical)

    print('Analytically found coefficients in Das et al. 2020:')
    print('---------------------------------------------------')
    print(Bcoefs_analytical)
    print('\nNumerically evaluated coefficients from magsplitpy:')
    print('---------------------------------------------------')
    print(Bcoefs_numerical)


if __name__ == "__main__":
    # stipulating the field type: dipole/toroidal/mixed
    field_type = 'mixed'
    make_B_analytical = B_field_D20()
    B_analytical, Bcoefs_analytical = make_B_analytical.make_B_D20(field_type=field_type)
    Bcoefs_numerical = make_B_analytical.get_GSH_numerically(B_analytical)
    compare_coefs_analytical_numerical(Bcoefs_analytical, Bcoefs_numerical)

    # making h_mu_nu_st_r
    len_r = 10
    sB_max = 1

    # empty list for magnetic field coefficients
    B_r_mu_st = []

    # extracting the GSH components of the generic 3D B field one radial slice at a time
    for r_ind in range(len_r):
        print(r_ind)
        B_r_mu_st.append(Bcoefs_numerical)

    # moving the radius dimension form the first to the very end
    B_mu_st_r = np.moveaxis(np.asarray(B_r_mu_st), 0, -1)

    # getting the BB GSH components from the B GSH components
    h_mu_nu_st_r = mag_GSH_funcs.make_BB_GSH_from_B_GSH(B_mu_st_r, sB_max=sB_max)