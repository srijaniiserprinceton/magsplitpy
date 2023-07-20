import spherepy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
plt.ion()

# function to convert a given 1 or 2 dimensional field to its SH or VSH components
# the returned coefs Object is of ScalarCoefs or VectorCoefs type
def field2coef(B_field, nmax=10, mmax=10):
    # detects the number of dimensions and splits operations accordingly
    ndims = B_field.shape[0]

    # if 1D, then its regarded as Br only and uses ScalarPatternUniform
    if(ndims == 1):
        B_pattern = sp.ScalarPatternUniform(B_field[0])
        B_coefs = sp.spht(B_pattern, nmax=nmax, mmax=mmax)

    # if 2D, then its rgraded as (Btheta, Bphi) and uses TransversePatternUniform
    elif(ndims == 2):
        B_pattern = sp.TransversePatternUniform(B_field[0], B_field[1])
        B_coefs = sp.vspht(B_pattern, nmax=nmax, mmax=mmax)

    # cannot accept 3D to stick to the convention of spherepy
    else:
        print("You cannot pass 3D arrays. Make objects 1D (for radial) and 2D (for transverse).")

    return B_coefs

# function to convert given 1 or 2 dimensional VSH coefficents to its corresponding field
# the input coefs Object is of ScalarCoefs or VectorCoefs type
def coef2field(B_coefs, ntheta=180, nphi=360):

    # if 1D, then its regarded as Br only
    if isinstance(B_coefs, sp.ScalarCoefs):
        # B_pattern = np.zeros((1, ntheta, nphi), dtype='complex128')
        B_pattern = sp.ispht(B_coefs, nrows=ntheta, ncols=nphi).array
        B_pattern = np.array([B_pattern])

    # if 2D, then its regarded as (Btheta, Bphi)
    elif isinstance(B_coefs, sp.VectorCoefs):
        # B_pattern = np.zeros((2, ntheta, nphi), dtype='complex128')
        B_pattern = sp.vispht(B_coefs, nrows=ntheta, ncols=nphi).array
        # B_pattern[1] = sp.ispht(B_coefs[1], nrows=ntheta, ncols=nphi).array

    else:
        print("You cannot pass 3D arrays. Make objects 1D (for radial) and 2D (for transverse).")

    return B_pattern

# function to convert the coefficients from the convention of DT98 to Jackson
# spherepy follows the convention of Jackson
def convert_coeffs_DT2Jackson(v_coeff_DT, w_coeff_DT):
    v_coeff_J = 1j * w_coeff_DT
    w_coeff_J = -1 * v_coeff_DT

    return v_coeff_J, w_coeff_J

# function to convert the coefficients from the convention of Jackson to DT98
# spherepy follows the convention of Jackson while DT98 does not 
def convert_coeffs_Jackson2DT(v_coeff_J, w_coeff_J):
    v_coeff_DT = -1 * w_coeff_J
    w_coeff_DT = -1j * v_coeff_J

    return v_coeff_DT, w_coeff_DT

# converting the VSH coefficients of B field to its GSH coefficients
def make_GSH_from_VSH(v_st, w_st):
    # building the u^{-}_st and u^{+}_st as per Eqns.(C.141)-(C.142) of DT98
    B_m_st = 1./np.sqrt(2) * (v_st - 1j * w_st)
    B_p_st = 1./np.sqrt(2) * (v_st + 1j * w_st)

    return B_m_st, B_p_st

def add_noise_to_field(B_field, noise_percent=5):
    B_field_noisy = np.zeros_like(B_field)

    for i in range(B_field.shape[0]):
        B_field_noisy[i] = B_field[i] +\
                           np.max(np.abs(B_field[i])) * (noise_percent/100.) *\
                           np.random.rand(B_field[i].shape[0], B_field[i].shape[1])

    return B_field_noisy

def plot_Aitoff_projection(B_field, fig_title='Placeholder title'):
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
    """
    B1_DT = np.array([gYlm_theta, gYlm_phi])
    B2_DT = np.array([-gYlm_phi, gYlm_theta])

    return B1_DT, B2_DT

def gYlm_2Jbasis(gYlm_theta, gYlm_phi):
    """
    Function to return the transverse vector spherical harmonic 
    basis functions in the convention of Jackson Electrodynamics.
    """
    B1_J = -1j * np.array([-gYlm_phi, gYlm_theta])
    B2_J = -1 * np.array([gYlm_theta, gYlm_phi])

    return B1_J, B2_J

# to run some of the tests of the functions
if __name__ == "__main__":
    # the max angular and azimuthal degrees we want to deal with
    nmax, mmax = 10, 10

    # generating a scalar pattern (Br)
    # Building random coefficents for the field
    Br_st = sp.random_coefs(nmax, mmax, coef_type = sp.scalar)
    Br = coef2field(Br_st)

    # now pretending that this is the given field and adding a bit of noise
    Br_synth_raw = add_noise_to_field(Br, noise_percent=5)

    # converting the raw data to coefs
    Br_coefs = field2coef(Br_synth_raw)

    # converting the coefs back to fields
    Br_field_rec = coef2field(Br_coefs)

    # plotting the raw and reconstructed fields
    plot_Aitoff_projection(Br_synth_raw, fig_title='Br_synth_raw')
    plot_Aitoff_projection(Br_field_rec, fig_title='Br_field_rec')

    # generating a vector pattern (Btheta, Bphi) for the transverse components
    # Building random coefficents for the two components of the vector field
    Bvec_st = sp.random_coefs(nmax, mmax, coef_type = sp.vector)
    Bvec_tuple = coef2field(Bvec_st)
    Bvec = np.array([Bvec_tuple[0], Bvec_tuple[1]])

    # now pretending that this is the given field and adding a bit of noise
    Bvec_synth_raw = add_noise_to_field(Bvec, noise_percent=5)

    # converting the raw data to coefs
    Bvec_coefs = field2coef(Bvec_synth_raw)

    # converting the coefs back to fields
    Bvec_field_rec_tuple = coef2field(Bvec_coefs)
    Bvec_field_rec = np.array([Bvec_field_rec_tuple[0],
                               Bvec_field_rec_tuple[1]])

    # plotting the raw and reconstructed fields
    plot_Aitoff_projection(Bvec_synth_raw, fig_title='Bvec_synth_raw')
    plot_Aitoff_projection(Bvec_field_rec, fig_title='Bvec_field_rec')
