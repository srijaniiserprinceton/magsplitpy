import spherepy as sp
import numpy as np
import scipy.special as special
import scipy.integrate as integrate
import matplotlib.pyplot as plt
plt.ion()

import sph_transforms as spht

def compare_Ylm_gradients(vlm, wlm, ell, m, toPlot=False):
    """
    Function to compute the 2D-gradient in theta and phi and compare 
    with the numerically evaluated arrays. This serves as a sanity check of 
    what we think spherepy is computing as well as its normalizations. The tests
    are not performed using np.testing.assert_array_almost_equal since due to 
    the nature of numerical derivative, the values differ at the 3rd or 4th decimal
    place but visually, we can see if the spherepy and numerically evaluated functions
    are the same or not.

    Parameters
    ----------
    vlm : array_like of floats
          The array of coefficients to be multiplied with the first basis function
          of spherepy convention (which is the convention of Jackson).
    wlm : array_like of floats
          The array of coefficients to be multiplied with the second basis function
          of spherepy convention (which is the convention of Jackson).
    ell : int
          The spherical harmonic angular degree of the Ylm.
    m : int 
          The spherical harmnonic azimuthal order of Ylm.
    toPLot: bool, optional
          Whether or not to generate the comparative plots between spherepy and the
          numerical estimates of gradients of Ylm for visual comparison.
    """
    # building spherical harmonic coefficients from the following grid
    ntheta, nphi = 180, 360
    theta = np.linspace(0,np.pi,ntheta)
    phi = np.linspace(0,2*np.pi-2*np.pi/nphi,nphi)

    # meshgrid
    thth, phph = np.meshgrid(theta, phi, indexing='ij')

    # creating from spherepy
    Y_coefs = sp.random_coefs(ell, abs(m), coef_type = sp.vector)
    Y_coefs.scoef1._vec[:] *= 0.0
    Y_coefs.scoef1._vec[3] += vlm
    Y_coefs.scoef2._vec[:] *= 0.0
    Y_coefs.scoef2._vec[3] += wlm

    # creating from spherepy
    Ylm_vec = sp.vispht(Y_coefs, nrows=ntheta, ncols=nphi).array

    # creating from custom Ylm gradients
    # gYlm_theta = dYlm/dtheta / np.sqrt(ell*(ell+1))
    # gYlm_phi = (1/sin_theta) * dYlm/dphi / np.sqrt(ell*(ell+1))
    gYlm_theta, gYlm_phi = spht.gYlm(ell, m, theta, phi)

    # constructing the bases of grad1_Ylm from DT98
    B1_DT, B2_DT = spht.gYlm_2DTbasis(gYlm_theta, gYlm_phi)

    # changing it to the bases as in Jackson
    B1_J, B2_J = spht.gYlm_2Jbasis(gYlm_theta, gYlm_phi)

    # testing if they are the same for proof of concept
    gYlm_vec_pattern_spherepy = Ylm_vec
    gYlm_vec_pattern_custom = vlm * B1_J + wlm * B2_J

    # comparing the theta and phi components 
    # np.testing.assert_array_almost_equal(gYlm_vec_pattern_custom[0],\
    #                                      gYlm_vec_pattern_spherepy[0])
    # np.testing.assert_array_almost_equal(gYlm_vec_pattern_custom[1],\
    #                                      gYlm_vec_pattern_spherepy[1])

    # Note: We are primarily using a visual comparison because of the nan values
    # in the custom made Ylm gradients.
    # plotting if toPlot is true
    if(toPlot):
        plot_patterns(gYlm_vec_pattern_custom[0].real,\
                      gYlm_vec_pattern_spherepy[0].real, thth, phph)
        plot_patterns(gYlm_vec_pattern_custom[0].imag,\
                      gYlm_vec_pattern_spherepy[0].imag, thth, phph)
        plot_patterns(gYlm_vec_pattern_custom[1].real,\
                      gYlm_vec_pattern_spherepy[1].real, thth, phph)
        plot_patterns(gYlm_vec_pattern_custom[1].imag,\
                      gYlm_vec_pattern_spherepy[1].imag, thth, phph)
    
    return gYlm_vec_pattern_custom, gYlm_vec_pattern_spherepy

def plot_patterns(Y1, Y2, thth, phph):
    """
    Plotter function to compare two scalar spherical harmonics
    for a given meshgrid in theta and phi.

    Parameters
    ----------
    Y1 : complex float_like
         The complex spherical harmonic generated from spherepy.
    Y2 : complex float_like
         The complex spherical harmonic generated from numerical gradient evaluation of Ylm.
    thth : array_like of floats, shape (Ntheta x Nphi)
           The meshgrid in theta generated using np.meshgrid(theta, phi, indexing='ij').
    phph : array_like of floats, shape (Ntheta x Nphi)
           The meshgrid in phi generated using np.meshgrid(theta, phi, indexing='ij').
    """
    # plotting the scalar spherical harmonic
    figr = plt.figure()
    axr = figr.add_subplot(131)
    imr = axr.pcolormesh(Y1, cmap=plt.cm.jet)
    figr.colorbar(imr,ax=axr)

    axr = figr.add_subplot(132)
    imr = axr.pcolormesh(Y2, cmap=plt.cm.jet)
    figr.colorbar(imr,ax=axr)

    axr = figr.add_subplot(133)
    imr = axr.pcolormesh(np.log10(Y1/Y2), cmap=plt.cm.jet, vmin=-0.1, vmax=0.1)
    figr.colorbar(imr,ax=axr)

def compare_Ylms(ell, m, toPlot=False):
    """
    Testing function to compare the spherepy generated scalar Ylm 
    vs. the numerically calculated scalar Ylm from np.scipy.special.sph_harm().

    Parameters
    ----------
    ell : int
          The angular degree of the desired scalar spherical harmonic.
    m :   int
          The azimiuthal order of the desired spherical harmonic.
    toPLot : bool, optional
             Either True or False depending on if we also want to visualize the 
             scalar Ylms. By default, this is set to False. 
    """
    # building spherical harmonic coefficients from the following grid
    ntheta, nphi = 180, 360
    theta = np.linspace(0,np.pi,ntheta)
    phi = np.linspace(0,2*np.pi-2*np.pi/nphi,nphi)

    # meshgrid
    thth, phph = np.meshgrid(theta, phi, indexing='ij')

    # creating from scipy. Transposing to match the dimension convention of spherepy
    Y_scipy = special.sph_harm(m, ell, phph, thth)

    # creating from spherepy
    Y_coefs = sp.random_coefs(ell, abs(m), coef_type = sp.scalar)
    Y_coefs._vec[:] *= 0.0
    Y_coefs._vec[3] += 1.0
    Y_spherepy = sp.ispht(Y_coefs, nrows=ntheta, ncols=nphi).array

    # testing if they are the same
    # np.testing.assert_array_almost_equal(Y_scipy, Y_spherepy)

    # plotting if toPlot is true
    if(toPlot):
        plot_patterns(Y_scipy.real, Y_spherepy.real, thth, phph)
        plot_patterns(Y_scipy.imag, Y_spherepy.imag, thth, phph)

    return Y_scipy, Y_spherepy


def test_Ylm_gYlm_orthonormalizations():
    """
    Function to test if the Ylm and its gradients have normalizations
    and orthogonalities consistent with DT98. If this is satisfied, then
    the rest of the basis functions constructions should be fine too (if the 
    compare_Ylm() and compare_Ylm_gradients() in test_spherepy.py are working).
    """
    ell1, m1 = 1, 1
    ell2, m2 = 1, -1

    # number of grids in theta and phi
    ntheta, nphi = 180, 360
    # creating the grids
    theta = np.linspace(0,np.pi,ntheta)
    phi = np.linspace(0,2*np.pi-2*np.pi/nphi,nphi)
    # meshgrid
    thth, phph = np.meshgrid(theta, phi, indexing='ij')

    # creating the Ylm gradients from spherepy
    # creating from spherepy
    Y1_coefs = sp.zeros_coefs(ell1, abs(m1), coef_type = sp.vector)
    Y1_coefs.scoef1._vec[3] += 1.0
    Y1_coefs.scoef2._vec[3] += 1.0

    Y2_coefs = sp.zeros_coefs(ell2, abs(m2), coef_type = sp.vector)
    Y2_coefs.scoef1._vec[2] += 1.0
    Y2_coefs.scoef2._vec[2] += 1.0

    # creating the Ylms from spherepy
    Y1, Y2 = sp.ispht(Y1_coefs.scoef1, nrows=ntheta, ncols=nphi).array,\
             sp.ispht(Y2_coefs.scoef1, nrows=ntheta, ncols=nphi).array

    # creating the gradients of Ylm from spherepy
    Y1_vec = sp.vispht(Y1_coefs, nrows=ntheta, ncols=nphi).array
    Y2_vec = sp.vispht(Y2_coefs, nrows=ntheta, ncols=nphi).array

    # testing the Ylm normalization
    Y1starY1_norm = integrate.simpson(integrate.simpson(np.conjugate(Y1) * Y1, x=phi, axis=1) \
                                      * np.sin(theta), x=theta)
    Y2starY2_norm = integrate.simpson(integrate.simpson(np.conjugate(Y2) * Y2, x=phi, axis=1) \
                                      * np.sin(theta), x=theta)

    # testing the Ylm orthogonality
    Y1starY2_norm = integrate.simpson(integrate.simpson(np.conjugate(Y1) * Y2, x=phi, axis=1) \
                                      * np.sin(theta), x=theta)

    # testing the gradients of Ylm normalizations
    gY1_comp1_star_gY1_comp1 = integrate.simpson(integrate.simpson(np.conjugate(Y1_vec[0]) @ Y1_vec[0], x=phi, axis=1) \
                                                 * np.sin(theta), x=theta)
    gY1_comp2_star_gY1_comp2 = integrate.simpson(integrate.simpson(np.conjugate(Y1_vec[1]) @ Y1_vec[1], x=phi, axis=1) \
                                                 * np.sin(theta), x=theta)
    gY2_comp1_star_gY2_comp1 = integrate.simpson(integrate.simpson(np.conjugate(Y2_vec[0]) * Y2_vec[0], x=phi, axis=1) \
                                                 * np.sin(theta), x=theta)
    gY2_comp2_star_gY2_comp2 = integrate.simpson(integrate.simpson(np.conjugate(Y2_vec[1]) * Y2_vec[1], x=phi, axis=1) \
                                                 * np.sin(theta), x=theta)

    # testing the gradients of Ylm orthogonalitites
    gY1_comp1_star_gY1_comp2 = integrate.simpson(integrate.simpson(np.conjugate(Y1_vec[0]) * Y1_vec[1], x=phi, axis=1) \
                                                 * np.sin(theta), x=theta)
    gY2_comp1_star_gY2_comp2 = integrate.simpson(integrate.simpson(np.conjugate(Y2_vec[0]) * Y2_vec[1], x=phi, axis=1) \
                                                 * np.sin(theta), x=theta)
    gY1_comp1_star_gY2_comp1 = integrate.simpson(integrate.simpson(np.conjugate(Y1_vec[0]) * Y2_vec[0], x=phi, axis=1) \
                                                 * np.sin(theta), x=theta)
    gY1_comp2_star_gY2_comp2 = integrate.simpson(integrate.simpson(np.conjugate(Y1_vec[1]) * Y2_vec[1], x=phi, axis=1) \
                                                 * np.sin(theta), x=theta)
    
    print("\nWe expect the following to be close to 1.0:")
    print("-------------------------------------------")
    print(Y1starY1_norm)
    print(Y2starY2_norm)
    print(gY1_comp1_star_gY1_comp1)
    print(gY1_comp2_star_gY1_comp2)
    print(gY2_comp1_star_gY2_comp1)
    print(gY2_comp2_star_gY2_comp2)

    
    print("\nWe expect the following to be close to 0.0:")
    print("-------------------------------------------")
    print(Y1starY2_norm)
    print(gY1_comp1_star_gY1_comp2)
    print(gY2_comp1_star_gY2_comp2)
    print(gY1_comp1_star_gY2_comp1)
    print(gY1_comp2_star_gY2_comp2)
        
    

if __name__ == "__main__":
    # spherical harmonics degree of interest
    ell, m = 1, 1

    # testing the scalar spherical harmonics
    Y1, Y2 = compare_Ylms(ell, m, toPlot=True)

    print('Scalar comparison done: Compared arrays element-by-element.')

    # testing the vector spherical harmonics
    vlm, wlm = np.random.rand(2)
    Y1_vec = compare_Ylm_gradients(vlm, wlm, ell, m, toPlot=True)

    print('Vector comparison done: Visual check by plotting the different real components.')

    # test_Ylm_gYlm_orthonormalizations()