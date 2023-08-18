import os
import h5py
import numpy as np
from sympy.physics.wigner import wigner_3j
from sympy import N as sympy_eval
import scipy.special as special
from statistics import mode
import sympy as sy
import py3nj
import re

P_j = np.array([])

def wig(l1,l2,l3,m1,m2,m3):
	"""returns numerical value of wigner3j symbol"""
	if (np.abs(m1) > l1 or np.abs(m2) > l2 or np.abs(m3) > l3):
	    return 0.
	return(sympy_eval(wigner_3j(l1,l2,l3,m1,m2,m3)))

def w3j_vecm(l1, l2, l3, m1, m2, m3):
    """Computes the wigner-3j symbol for given l1, l2, l3, m1, m2, m3.

    Inputs:
    -------
    l1, l2, l3 - int
    m1, m2, m3 - np.ndarray(ndim=1, dtype=np.int32)

    Returns:
    --------
    wigvals - np.ndarray(ndim=1, dtype=np.float32)
    """
    l1 = int(2*l1)
    # l2 = int(2*l2)
    l3 = int(2*l3)
    # l1 = 2*l1.astype('int')
    l2 = 2*l2.astype('int')
    # l3 = 2*l3.astype('int')
    m1 = 2*m1
    m2 = 2*m2
    m3 = 2*m3
    wigvals = py3nj.wigner3j(l1, l2, l3, m1, m2, m3, ignore_invalid=True)
    return wigvals

def omega(l,n):
	"""returns numerical value of \Omega_l^n"""
	if (np.abs(n) > l):	
		return 0.
	return np.sqrt(0.5*(l+n)*(l-n+1.))

def gam(s):
    return np.sqrt((2.*s+1.)/ (4.* np.pi))
    
def P_a(l,i):
    global P_j

    P_l_vec = np.vectorize(special.legendre(i))
    L = np.sqrt(l*(l+1))
    m = np.arange(-l,l+1,1)

    #returns 2*l+1 values of P_j^l(m)
    P = np.zeros(2*l+1)
    
    if (l == 0):
        print("l can't be zero in discretised Legendre P")
        return None
    
    #P_0^l(m) = l
    if(i==0): 
        P += l
    
    #creating P''(m) for all m's belonging to l. Needed for c_ij
    P_pp_i = L*P_l_vec((1.*m)/L)
    P_p_i = np.zeros(2*l+1)

    for j in range(0,i,1):
        c_ij_num = 0.0
        c_ij_denom = 0.0 
        P_j_l = P_j[j]  #using pre-computed P_j^l(m)'s
        c_ij_num = np.sum(P_pp_i*P_j_l)
        c_ij_denom = np.sum(P_j_l**2)
        c_ij = c_ij_num/c_ij_denom
        P_p_i -= c_ij*P_j_l
    
    P_p_i += P_pp_i

    #returns an array of length (2*l+1) 
    P = l*P_p_i/P_p_i[-1]

    if(i==0):
        P_j = np.append(P_j,P)
        P_j = np.reshape(P_j,(1,2*l+1))
    else: P_j = np.append(P_j,np.reshape(P,(1,2*l+1)),axis=0)
    return P
    
def a_coeff(del_om, l, jmax, plot_switch = False):
    """a[0] is actually a_1"""

    #this part is just for plotting the basis P's
    if(plot_switch):
        for j in range(jmax+1): P_a(l,j)

        m = np.arange(-l,l+1,1)
        for i in range(jmax+1):
            plt.plot(m,P_j[i],label='j = %i'%i)
        plt.legend()
        plt.ylabel('$\mathcal{P}_{j}^{(%i)}$'%l)
        plt.xlabel('m')
        plt.show()
        return 0

    #this is where the a-coeffs are computed
    a = np.zeros(jmax+1)
    for j in range(jmax+1):
        for m in np.arange(-l,l+1,1):
            a[j] += del_om[m+l] * P_a(l,j)
        a[j] *= (j+0.5) / l**3

    return a

def a_coeff_matinv(del_om, l, jmax):
    """Inverting for a-coeff from matrix inversion. AC = B. A contains coeffs"""

    P_a_vec = np.vectorize(P_a)

    A_i = np.zeros(jmax+1)
    j = np.arange(0,jmax+1,1)
    m = np.arange(-l,l+1,1)
    jj,mm = np.meshgrid(j,m,indexing='ij')

    P_j_m = P_a_vec(mm,l,jj)
    C_j_i = np.matmul(P_j_m,np.transpose(P_j_m))
    B_j = np.matmul(P_j_m,del_om)

    A_i = np.linalg.solve(C_j_i,B_j)
    return A_i

#finding a-coefficients using GSO
def a_coeff_GSO(del_om,l,jmax):
    """a[0] is actually a_1"""
    global P_j

    a = np.ones(jmax+1)
    a_num = np.zeros(jmax+1)
    a_denom = np.zeros(jmax+1)
    for j in range(jmax+1): 
        P_l_j =  P_a(l,j)
        a_num[j] += np.sum(del_om * P_l_j)
        a_denom[j] += np.sum(P_l_j**2)
    a = a_num/a_denom

    P_j = np.array([])

    return a


# finding the normalization value of eigenfunctions
def eignorm(U,V,ell,r,rho):
    return np.sqrt(np.trapz(rho * (U**2 + ell*(ell+1) * V**2) * r**2, r) * 4 * np.pi)

# finding the grid that appears most frequently in a given set of GYRE eigenfunctions
def find_mode_r_grid(dir_eigfiles):
    print("1. Scanning through modefiles to find the most common grid.")
    filenames = np.asarray(os.listdir(f'{dir_eigfiles}/'))
    len_r = []

    for f in filenames:
        n_str = re.split('[_]+', f, flags=re.IGNORECASE)[2]
        ell_str = re.split('[.]+', re.split('[_]+', f, flags=re.IGNORECASE)[1])[1]
        eigfile = h5py.File(f'{dir_eigfiles}/mode_h.{ell_str}_{n_str}_hz.h5')
        len_r.append(eigfile['x'][()].shape[0])

    len_r = np.asarray(len_r)
    mode_len_r = mode(len_r)
    
    for f in filenames:
        n_str = re.split('[_]+', f, flags=re.IGNORECASE)[2]
        ell_str = re.split('[.]+', re.split('[_]+', f, flags=re.IGNORECASE)[1])[1]
        eigfile = h5py.File(f'{dir_eigfiles}/mode_h.{ell_str}_{n_str}_hz.h5')
        if(eigfile['x'][()].shape[0] == mode_len_r): break

    # the radius and rho grid for most multiplets
    r_norm_Rstar = eigfile['x'][()]   # reading it off a random file since its the same for all
    rho = eigfile['rho'][()]

    return r_norm_Rstar, rho