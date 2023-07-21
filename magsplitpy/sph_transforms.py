import spherepy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
plt.ion()


def field2coef(B_field, ellmax=10, mmax=10):
    """
    Function to convert a given field to its SH or VSH components.
    The returned coefs Object is of spherepy's ScalarCoefs or VectorCoefs type. Note that
    you cannot use a field with all three vector components at the same time.
    Either provide just the radial component $B_r$ or the transverse components $(B_{\\theta}, B_{\phi})$.
    
    Parameters:
    -----------
    B_field : array_like of floats, shape (Ntheta x Nphi)
              Either the radial component or the transverse components of the field.

    ellmax : int, optional
           Maximum angular degree for truncating the spherical harmonic transform.

    mmax : int, optional
           Maximum aximuthal order for truncating the spherical harmonic transform.
           Has to be a positive number.

    Returns:
    --------
    B_coefs : spherepy.ScalarCoefs or spherepy.VectorCoefs type
              The coefficients after spherical harmonic transform using spherepy's
              spht (for scalar) or vspht (for vectors) transforms. 
    """
    # detects the number of dimensions and splits operations accordingly
    ndims = B_field.shape[0]

    # if 1D, then its regarded as Br only and uses ScalarPatternUniform
    if(ndims == 1):
        B_pattern = sp.ScalarPatternUniform(B_field[0])
        B_coefs = sp.spht(B_pattern, nmax=ellmax, mmax=mmax)

    # if 2D, then its rgraded as (Btheta, Bphi) and uses TransversePatternUniform
    elif(ndims == 2):
        B_pattern = sp.TransversePatternUniform(B_field[0], B_field[1])
        B_coefs = sp.vspht(B_pattern, nmax=ellmax, mmax=mmax)

    # cannot accept 3D to stick to the convention of spherepy
    else:
        print("You cannot pass 3D arrays. Make objects 1D (for radial) and 2D (for transverse).")

    return B_coefs


def coef2field(B_coefs, Ntheta=180, Nphi=360):
    """
    Function to convert given 1 or 2 dimensional VSH coefficents to its corresponding field.
    The input coefs Object is of spherepy's ScalarCoefs or VectorCoefs type.

    Parameters:
    -----------
    B_coefs : complex array, spherepy.ScalarCoefs or spherepy.VectorCoefs
              The magnetic field coefficients (for either radial or transverse) which are to be
              converted to its equivalent 2D pattern in $(\theta, \phi)$.

    Ntheta : int, optional


    Nphi : int, optional

    Returns:
    --------
    B_pattern : ndarray of floats, shape (Ntheta x Nphi)
                The 2D profile corresponding to either a scalar $B_r$ field or a
                vector $(B_{\\theta}, B_{\phi})$ field.
    """
    # if 1D, then its regarded as Br only
    if isinstance(B_coefs, sp.ScalarCoefs):
        # B_pattern = np.zeros((1, ntheta, nphi), dtype='complex128')
        B_pattern = sp.ispht(B_coefs, nrows=Ntheta, ncols=Nphi).array
        B_pattern = np.array([B_pattern])

    # if 2D, then its regarded as (Btheta, Bphi)
    elif isinstance(B_coefs, sp.VectorCoefs):
        # B_pattern = np.zeros((2, ntheta, nphi), dtype='complex128')
        B_pattern = sp.vispht(B_coefs, nrows=Ntheta, ncols=Nphi).array
        # B_pattern[1] = sp.ispht(B_coefs[1], nrows=ntheta, ncols=nphi).array

    else:
        print("You cannot pass 3D arrays. Make objects 1D (for radial) and 2D (for transverse).")

    return B_pattern


def convert_coeffs_DT2Jackson(v_coeff_DT, w_coeff_DT):
    """
    Function to convert the coefficients from the convention of DT98 to Jackson.
    Spherepy follows the convention of Jackson.

    Parameters:
    -----------
    v_coeff_DT : complex ndarray
                 The set of coefficients corresponding to basis function $\\nabla_1 Y_{\elll m}$ in Dahlen and Tromp 1998.
    w_coeff_DT : complex ndarray
                 The set of coefficients corresponding to basis function $\hat{r} \\times \\nabla_1 Y_{\ell m}$ in Dahlen and Tromp 1998.

    Returns:
    --------
    v_coeff_J : complex ndarray
                The set of coefficients corresponding to basis function X in Jackson.
    w_coeff_J : complex ndarray
                The set of coefficients corresponding to basis function Y in Jackson.
    """
    v_coeff_J = 1j * w_coeff_DT
    w_coeff_J = -1 * v_coeff_DT

    return v_coeff_J, w_coeff_J


def convert_coeffs_Jackson2DT(v_coeff_J, w_coeff_J):
    """
    Function to convert the coefficients from the convention of Jackson to DT98.
    Spherepy follows the convention of Jackson while DT98 does not.

    Parameters:
    -----------
    v_coeff_J : complex ndarray
                The set of coefficients corresponding to basis function X in Jackson.
    w_coeff_J : complex ndarray
                The set of coefficients corresponding to basis function Y in Jackson.

    Returns:
    --------
    v_coeff_DT : complex ndarray
                 The set of coefficients corresponding to basis function $\\nabla_1 Y_{\elll m}$ in Dahlen and Tromp 1998.
    w_coeff_DT : complex ndarray
                 The set of coefficients corresponding to basis function $\hat{r} \\times \\nabla_1 Y_{\ell m}$ in Dahlen and Tromp 1998.
    """
    v_coeff_DT = -1 * w_coeff_J
    w_coeff_DT = -1j * v_coeff_J

    return v_coeff_DT, w_coeff_DT


def make_GSH_from_VSH(v_st, w_st):
    """
    Converting the VSH coefficients of B field to its GSH coefficients.
    
    Parameters:
    -----------
    v_st : array_like of floats
           Array containing the coefficients of the first transverse vector basis
           according to the convention followed in Dahlen and Tromp 1998. 
    w_st : array_like of floats
           Array containing the coefficients of the second transverse vector basis
           according to the convention followed in Dahlen and Tromp 1998.

    Returns:
    --------
    B_m_st : complex ndarray
             The complex array of the coefficients corresponding to $\hat{e}_{-}$ basis.
    B_p_st : complex ndarray
             The complex array of the coefficients corresponding to $\hat{e}_{+}$ basis.
    """
    # building the u^{-}_st and u^{+}_st as per Eqns.(C.141)-(C.142) of DT98
    B_m_st = 1./np.sqrt(2) * (v_st - 1j * w_st)
    B_p_st = 1./np.sqrt(2) * (v_st + 1j * w_st)

    return B_m_st, B_p_st


def add_noise_to_field(B_field, noise_percent=5):
    """
    Function to add a random noise using np.random.rand() to a provided synthetic
    B_field at a desired percent of the noiseless field.

    Parameters:
    -----------
    B_field : array_like of floats
              The synthetic magnetic field to which we want to add noise. This has to have
              the shape (ND x Nr x Ntheta x Nphi) where ND = number of vector components
              considered. 

    noise_percent : float, optional
                    The percent level at which to add noise to each of the components. As 
                    of now this is a scalar and so each component has the same percent of noise
                    added.

    Returns:
    --------
    B_field_noisy : ndarray of float_like
                    The synthetic mangetic field along with the random noise added to it.
    """
    B_field_noisy = np.zeros_like(B_field)

    for i in range(B_field.shape[0]):
        B_field_noisy[i] = B_field[i] +\
                           np.max(np.abs(B_field[i])) * (noise_percent/100.) *\
                           np.random.rand(B_field[i].shape[0], B_field[i].shape[1])

    return B_field_noisy

def plot_Aitoff_projection(B_field, fig_title='Placeholder title'):
    """
    Plot different components of the magnetic field in a Aitoff projection.

    Parameters:
    -----------
    B_field : array_like of floats
              Contains a 1D, 2D or 3D field whose different vector components are to be 
              plotted.
    fig_title : string, optional
                The title for the figure to denote which plot corresponds to which components.
    """
    # number of dimensions of the B_field
    ndims = B_field.shape[0]

    theta = np.linspace(-np.pi/2,np.pi/2,180)
    phi = np.linspace(-np.pi,np.pi,360)
    # meshgrid
    phph, thth = np.meshgrid(phi,theta)

    figr = plt.figure()

    # plotting each component on Aitoff projection
    for i in range(ndims):
        B_comp = B_field[i]
        subplot_string = f'{ndims}1{i+1}'
        print(subplot_string)
        axr = figr.add_subplot(int(subplot_string), projection='aitoff')
        imr = axr.pcolormesh(phph, thth, B_comp.real, cmap=plt.cm.jet)
        axr.set_title(fig_title, pad=50)
        figr.colorbar(imr,ax=axr)

    plt.tight_layout()


def gYlm(ell, m, theta, phi):
    """
    Function to compute the two components of derivative of Ylm needed to make the different basis
    functions of VSH --- either the Dahlen and Tromp convention or the Jackson convention.

    Parameters:
    -----------
    ell : int

    m : int

    theta : 1D array_like of floats 
            Array contatining the 1D $\\theta$ mesh.

    phi : 1D array_like of floats
          Array contatining the 1D $\phi$ mesh.


    Returns:
    --------
    gYlm_theta : ndarray, shape (Ntheta x Nphi)
                 Numerically evaluates $\\frac{\partial Y_{\ell m}}{\partial \\theta} / \sqrt{\ell (\ell + 1)}$.

    gYlm_phi : ndarray, shape (Ntheta x Nphi)
               Numerically evaluates $\\frac{1}{\sin{\\theta}}\\frac{\partial Y_{\ell m}}{\partial \phi} / \sqrt{\ell (\ell + 1)}$.
    """
    # meshgrid
    thth, phph = np.meshgrid(theta, phi, indexing='ij')

    # making the Ylm from scipy.special
    Ylm = special.sph_harm(m, ell, phph, thth)

    # calculating the gradients with respect to theta and phi
    # including the factor 1/(ell * (ell + 1)) to be consistent with 
    # conventions in DT98 and Jackson
    gYlm_theta = np.gradient(Ylm, theta , axis=0, edge_order=2) / np.sqrt(ell*(ell+1))
    gYlm_phi = 1./np.sin(thth) * np.gradient(Ylm, phi, axis=1, edge_order=2) / np.sqrt(ell*(ell+1))

    return gYlm_theta, gYlm_phi

def gYlm_2DTbasis(gYlm_theta, gYlm_phi):
    """
    Function to return the transverse vector spherical harmonic 
    basis functions in the convention of Dahlen and Tromp 1998.

    Parameters:
    -----------
    gYlm_theta : 2D complex array_like, shape (Ntheta x Nphi)
                 The $\\frac{\partial Y_{\ell m}}{\partial \\theta} / \sqrt(\ell (\ell+1))$ term.
    gYlm_phi : 2D complex array_like, shape (Ntheta x Nphi)
               The $\\frac{1}{\sin{\\theta}}\\frac{\partial Y_{\ell m}}{\partial \\theta} / \sqrt(\ell (\ell+1))$ term.

    Returns:
    --------
    B1_DT : ndarray, shape (Ntheta x Nphi)
            The component corresponding to the first transverse basis function of VSH in Dahlen and Tromp 1998.
    B2_DT : ndarray, shape (Ntheta x Nphi)
            The component corresponding to the second transverse basis function of VSH in Dahlen and Tromp 1998.
    """
    B1_DT = np.array([gYlm_theta, gYlm_phi])
    B2_DT = np.array([-gYlm_phi, gYlm_theta])

    return B1_DT, B2_DT

def gYlm_2Jbasis(gYlm_theta, gYlm_phi):
    """
    Function to return the transverse vector spherical harmonic 
    basis functions in the convention of Jackson Electrodynamics.

    Parameters:
    -----------
    gYlm_theta : 2D complex array_like, shape (Ntheta x Nphi)
                 The $\\frac{\partial Y_{\ell m}}{\partial \\theta} / \sqrt(\ell (\ell+1))$ term.

    gYlm_phi : 2D complex array_like, shape (Ntheta x Nphi)
               The $\\frac{1}{\sin{\\theta}}\\frac{\partial Y_{\ell m}}{\partial \\theta} / \sqrt(\ell (\ell+1))$ term.

    Returns:
    --------
    B1_J : ndarray, shape (Ntheta x Nphi)
           The component corresponding to the first transverse basis function of VSH in Jackson.

    B2_J : ndarray, shape (Ntheta x Nphi)
           The component corresponding to the second transverse basis function of VSH in Jackson.
    """
    B1_J = -1j * np.array([-gYlm_phi, gYlm_theta])
    B2_J = -1 * np.array([gYlm_theta, gYlm_phi])

    return B1_J, B2_J

# to run some of the tests of the functions
if __name__ == "__main__":
    # the max angular and azimuthal degrees we want to deal with
    ellmax, mmax = 10, 10

    # generating a scalar pattern (Br)
    # Building random coefficents for the field
    Br_st = sp.random_coefs(ellmax, mmax, coef_type = sp.scalar)
    Br = coef2field(Br_st)

    # now pretending that this is the given field and adding a bit of noise
    Br_synth_raw = add_noise_to_field(Br, noise_percent=5)

    # converting the raw data to coefs
    Br_coefs = field2coef(Br_synth_raw, ellmax=ellmax, mmax=mmax)

    # converting the coefs back to fields
    Br_field_rec = coef2field(Br_coefs)

    # plotting the raw and reconstructed fields
    plot_Aitoff_projection(Br_synth_raw, fig_title='Br_synth_raw')
    plot_Aitoff_projection(Br_field_rec, fig_title='Br_field_rec')

    # generating a vector pattern (Btheta, Bphi) for the transverse components
    # Building random coefficents for the two components of the vector field
    Bvec_st = sp.random_coefs(ellmax, mmax, coef_type = sp.vector)
    Bvec_tuple = coef2field(Bvec_st)
    Bvec = np.array([Bvec_tuple[0], Bvec_tuple[1]])

    # now pretending that this is the given field and adding a bit of noise
    Bvec_synth_raw = add_noise_to_field(Bvec, noise_percent=5)

    # converting the raw data to coefs
    Bvec_coefs = field2coef(Bvec_synth_raw, ellmax=ellmax, mmax=mmax)

    # converting the coefs back to fields
    Bvec_field_rec_tuple = coef2field(Bvec_coefs)
    Bvec_field_rec = np.array([Bvec_field_rec_tuple[0],
                               Bvec_field_rec_tuple[1]])

    # plotting the raw and reconstructed fields
    plot_Aitoff_projection(Bvec_synth_raw, fig_title='Bvec_synth_raw')
    plot_Aitoff_projection(Bvec_field_rec, fig_title='Bvec_field_rec')
