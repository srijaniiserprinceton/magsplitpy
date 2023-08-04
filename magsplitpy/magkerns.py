# code where the Lorentz-stress kernels are compouted from eigenfunctions
import matplotlib.pyplot as plt
plt.ion()
import numpy as np

from magsplitpy import misc_funcs as fn

NAX = np.newaxis

class magkerns:
    def __init__(self, s, r, rho, axis_symm=True):
        self.s = s
        # self.t = np.arange(-self.s, self.s+1)
        self.r = r
        self.rho = rho

        #ss_in is s X r dim (inner)
        self.ss_i,__ = np.meshgrid(s,self.r, indexing = 'ij')
        self.mm_, self.mm, self.ss_o = None, None, None

        self.axis_symm = axis_symm

        # the mode parameters
        self.n, self.l, self.m = None, None, None
        self.n_, self.l_, self.m_ = None, None, None

        # eigenfunctions
        self.Ui, self.Vi = None, None
        self.Ui_, self.Vi_ = None, None


    def wig_red_o(self, m1, m2, m3):
        '''3j symbol with upper row fixed (outer)'''
        wig_vect = np.vectorize(fn.wig,otypes=[float])
        return wig_vect(self.l_,self.ss_o,self.l,m1,m2,m3)

    def wig_red(self, m1, m2, m3):
        '''3j symbol with upper row fixed (inner)'''
        wig_vect = np.vectorize(fn.wig,otypes=[float])
        return wig_vect(self.l_,self.ss_i,self.l,m1,m2,m3)

    def ret_kerns(self, n, l, m, n_=None, l_=None, m_=None, smoothen=False):    
        # loading the eigenfunctions
        # Ui, Vi = 
        # nl = fn.find_nl(n,l)
        # nl_ = fn.find_nl(n_,l_)

        # load eigenfunctions here
        Ui_raw = np.loadtxt('../sample_eigenfunctions/Un-162.txt')
        Vi_raw = np.loadtxt('../sample_eigenfunctions/Vn-162.txt')

        self.Ui = Ui_raw
        self.Vi = Vi_raw

        # for self-coupling
        if(l_==None):
            n_, l_, m_ = n, l, m
            self.n, self.l, self.m = n, l, m
            self.n_, self.l_, self.m_ = n, l, m
            self.Ui_, self.Vi_ = self.Ui, self.Vi
        # for cross-coupling
        else: 
            self.n, self.l, self.m = n, l, m
            self.n_, self.l_, self.m_ = n_, l_, m_

        # creating meshgrid for a generic field
        self.mm_, self.mm, self.ss_o = np.meshgrid(m_, m, self.s, indexing = 'ij')            

        len_m, len_m_, len_s = np.shape(self.ss_o)

        #Savitsky golay filter for smoothening
        window = 45  #must be odd
        order = 3

        om = np.vectorize(fn.omega,otypes=[float])
        parity_fac = (-1)**(l+l_+self.ss_o) #parity of selected modes

        # the 4pi cancels a 4pi from the denominator
        prefac = np.sqrt((2*l_+1.) * (2*self.ss_o+1.) * (2*l+1.) \
                 / (4.* np.pi)) * self.wig_red_o(-self.mm_,self.mm_-self.mm,self.mm)


        #----------------EIGENFUCNTION DERIVATIVES--------------------#
        #interpolation params
        if(smoothen == True):
            npts = 300
            r_new = np.linspace(np.amin(self.r),np.amax(self.r),npts)
            self.ss_i,__ = np.meshgrid(self.s,r_new, indexing = 'ij')

            Ui,dUi,d2Ui = fn.smooth(self.Ui,self.r,window,order,npts)
            Vi,dVi,d2Vi = fn.smooth(self.Vi,self.r,window,order,npts)

            Ui_,dUi_,d2Ui_ = fn.smooth(self.Ui_,self.r,window,order,npts)
            Vi_,dVi_,d2Vi_ = fn.smooth(self.Vi_,self.r,window,order,npts)

            rho_sm, __, __ = fn.smooth(self.rho,self.r,window,order,npts)
            #re-assigning with smoothened variables
            r = r_new
            rho = rho_sm

        
        ###############################################################
        #no smoothing
        else: 
            r = self.r
            rho = self.rho
            Ui = self.Ui 
            Vi = self.Vi 
            Ui_ = self.Ui_ 
            Vi_ = self.Vi_ 

            dUi, dVi = np.gradient(Ui,r), np.gradient(Vi,r)
            d2Ui,d2Vi = np.gradient(dUi,r), np.gradient(dVi,r)
            dUi_, dVi_ = np.gradient(Ui_,r), np.gradient(Vi_,r)
            d2Ui_,d2Vi_ = np.gradient(dUi_,r), np.gradient(dVi_,r)

        #################################################################3

        #making U,U_,V,V_,dU,dU_,dV,dV_,d2U,d2U_,d2V,d2V_ of same shape
        U = np.tile(Ui,(len_s,1))
        V = np.tile(Vi,(len_s,1))
        dU = np.tile(dUi,(len_s,1))
        dV = np.tile(dVi,(len_s,1))
        d2U = np.tile(d2Ui,(len_s,1))
        d2V = np.tile(d2Vi,(len_s,1))
        U_ = np.tile(Ui_,(len_s,1))
        V_ = np.tile(Vi_,(len_s,1))
        dU_ = np.tile(dUi_,(len_s,1))
        dV_ = np.tile(dVi_,(len_s,1))
        d2U_ = np.tile(d2Ui_,(len_s,1))
        d2V_ = np.tile(d2Vi_,(len_s,1))
        r = np.tile(r,(len_s,1))

        # Initializing the omegas which will be used very often. Might cut down on some computational time
        om0 = om(l,0)
        om0_ = om(l_,0)
        om2 = om(l,2)
        om2_ = om(l_,2)
        om3 = om(l,3)
        om3_ = om(l_,3)

        #B-- EXPRESSION
        Bmm = self.wig_red(2,-2,0)*om0_*om2_*(V_*(3.*U-2.*om0**2 *V + 3.*r*dU) - r*U*dV_)
        Bmm += self.wig_red(0,-2,2)*om0*om2*(V*(3.*U_-2.*om0_**2 *V_ + 3.*r*dU_) - r*U_*dV)
        Bmm += self.wig_red(1,-2,1)*om0_*om0*(3.*U*V_+3.*U_*V-2.*om0_**2 *V_*V - 2.*om0**2 *V_*V \
                + om2_**2*V_*V + om2**2 *V_*V + r*V*dU_ + r*V_*dU - r*U*dV_ - r*U_*dV - 2.*U_*U)
        Bmm += self.wig_red(3,-2,-1)*om0*om0_*om2_*om3_*V_*V
        Bmm += self.wig_red(-1,-2,3)*om0_*om0*om2*om3*V_*V                

        Bmm = 0.5*(((-1)**np.abs(1+self.mm_))*prefac)[:,:,:,np.newaxis] \
                 * Bmm[np.newaxis,:,:]
    

        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)*om0_*(4.*om0**2 *V_*V + U_*(8.*U-5.*om0**2* V) - 3.*r*om0**2*V*dV_ \
                + 2*r**2*dU*dV_ - r*om0**2 *V_*dV + r**2 *V_*d2U + U*((-6.-2.*om0_**2+om0**2)*V_ + r*(4.*dV_-r*d2V_)))
        B0m += self.wig_red(0,-1,1)*om0*(4.*om0_**2 *V*V_ + U*(8.*U_-5.*om0_**2 *V_) - 3.*r*om0_**2 *V_*dV \
                + 2*r**2*dU_*dV - r*om0_**2 *V*dV_ + r**2 *V*d2U_ + U_*((-6.-2.*om0**2+om0_**2)*V + r*(4.*dV-r*d2V)))
        B0m += self.wig_red(-1,-1,2)*om0*om0_*om2*(U*V_ + V*(U_-4.*V_+3.*r*dV_) + r*V_*dV)
        B0m += self.wig_red(2,-1,-1)*om0_*om0*om2_*(U_*V + V_*(U-4.*V+3.*r*dV) + r*V*dV_)

        B0m = (0.25*((-1)**np.abs(self.mm_))*prefac)[:,:,:,np.newaxis] \
                * B0m[np.newaxis,:,:]


        #B00 EXPRESSION
        B00 = self.wig_red(0,0,0)*2.*(-2.*r*U*dU_ - 2.*r*U_*dU + om0**2*r*V*dU_ + om0_**2*r*V_*dU - 5.*om0_**2*V_*U \
                - 5.*om0**2*V*U_ + 4.*om0**2*om0_**2*V_*V + om0_**2*r*U*dV_ + om0**2*r*U_*dV + 6.*U_*U)
        B00 += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*(-1.*om0_*om0)*(-U_*V-U*V_+2.*V_*V+r*V*dU_\
                +r*V_*dU-2.*r*V*dV_-2*r*V_*dV+r*U*dV_+r*U_*dV+2*r**2 *dV_*dV)

        B00 = (0.5*((-1)**np.abs(self.mm_))*prefac)[:,:,:,np.newaxis] \
                * B00[np.newaxis,:,:]

        #B+- EXPRESSION
        Bpm = self.wig_red(0,0,0)*2.*(-2.*r*dU_*U-2.*r*dU*U_+om0**2*r*dU_*V+om0_**2*r*dU*V_-2.*r**2*dU_*dU \
                -om0_**2*U*V_-om0**2*U_*V+om0_**2*r*U*dV_+om0**2*r*U_*dV + 2.*U_*U)
        Bpm += (self.wig_red(-2,0,2)+self.wig_red(2,0,-2))*(-4.*om0*om0_*om2*om2_*V_*V)
        Bpm += (self.wig_red(-1,0,1)+self.wig_red(1,0,-1))*(om0*om0_)*(-r*V*dU_-r*V_*dU-V_*U-V*U_+r*U*dV_+r*U_*dV+2.*U_*U)


        Bpm = (0.25*((-1)**np.abs(self.mm_))*prefac)[:,:,:,np.newaxis] \
                * Bpm[np.newaxis,:,:]

        #constructing the other two components of the kernel
        Bpp = parity_fac[:,:,:,np.newaxis]*Bmm
        Bp0 = parity_fac[:,:,:,np.newaxis]*B0m

        return Bmm,B0m,B00,Bpm,Bp0,Bpp
        
    def ret_kerns_axis_symm(self, n, l, m, n_=None, l_=None, smoothen = False, a_coeffkerns = False):
        # load eigenfunctions here
        Ui_raw = np.loadtxt('../sample_eigenfunctions/Un-162.txt')
        Vi_raw = np.loadtxt('../sample_eigenfunctions/Vn-162.txt')

        self.Ui = Ui_raw
        self.Vi = Vi_raw

        # for self-coupling
        if(l_==None):
            n_, l_, m_ = n, l, m
            self.n, self.l, self.m = n, l, m
            self.n_, self.l_, self.m_ = n, l, m
            self.Ui_, self.Vi_ = self.Ui, self.Vi
        # for cross-coupling
        else: 
            self.n, self.l, self.m = n, l, m
            self.n_, self.l_, self.m_ = n_, l_, m  # note that m = m_ since axisymmetric field

        # creating meshgrid for axisymmetric field
        self.mm, self.ss_o = np.meshgrid(m, self.s, indexing = 'ij')
        
        len_m, len_s = np.shape(self.ss_o)

        #Savitsky golay filter for smoothening
        window = 45  #must be odd
        order = 3

        om = np.vectorize(fn.omega,otypes=[float])
        # parity of selected modes
        parity_fac = (-1)**(l+l_+self.ss_o) 
        if(a_coeffkerns == True):
            prefac = ((-1.)**l)/(4.* np.pi) * np.sqrt((2*l_+1.) * (2*self.ss_o+1.) * (2*l+1.) \
                    / (4.* np.pi)) * self.wig_red_o(-l,0,l) / l 
        else:
            # the 4pi cancels a 4pi from the denominator
            prefac = np.sqrt((2*l_+1.) * (2*self.ss_o+1.) * (2*l+1.) \
                    / (4.* np.pi)) * self.wig_red_o(-self.mm, np.zeros_like(self.mm), self.mm)

        #-------------------EIGENFUCNTION DERIVATIVES----------------------#
        # smoothing
        # interpolation params
        if(smoothen == True):
            npts = 300      #should be less than the len(r) in r.dat
            r_new = np.linspace(np.amin(self.r),np.amax(self.r),npts)
            self.ss_i,__ = np.meshgrid(self.s,r_new, indexing = 'ij')

            Ui,dUi,d2Ui = fn.smooth(self.Ui,self.r,window,order,npts)
            Vi,dVi,d2Vi = fn.smooth(self.Vi,self.r,window,order,npts)

            Ui_,dUi_,d2Ui_ = fn.smooth(self.Ui_,self.r,window,order,npts)
            Vi_,dVi_,d2Vi_ = fn.smooth(self.Vi_,self.r,window,order,npts)

            rho_sm, __, __ = fn.smooth(self.rho,self.r,window,order,npts)
            #re-assigning with smoothened variables
            r = r_new
            rho = rho_sm
        
        #----------------------no smoothing------------------------------#

        else:
            r = self.r
            rho = self.rho
            Ui = self.Ui 
            Vi = self.Vi 
            Ui_ = self.Ui_ 
            Vi_ = self.Vi_ 

            dUi, dVi = np.gradient(Ui,r), np.gradient(Vi,r)
            d2Ui,d2Vi = np.gradient(dUi,r), np.gradient(dVi,r)
            print(Ui_.shape, Vi_.shape, r.shape)
            dUi_, dVi_ = np.gradient(Ui_,r), np.gradient(Vi_,r)
            d2Ui_,d2Vi_ = np.gradient(dUi_,r), np.gradient(dVi_,r)

        #########################################################################

        #making U,U_,V,V_,dU,dU_,dV,dV_,d2U,d2U_,d2V,d2V_ of same shape

        U = np.tile(Ui,(len_s,1))
        V = np.tile(Vi,(len_s,1))
        dU = np.tile(dUi,(len_s,1))
        dV = np.tile(dVi,(len_s,1))
        d2U = np.tile(d2Ui,(len_s,1))
        d2V = np.tile(d2Vi,(len_s,1))
        U_ = np.tile(Ui_,(len_s,1))
        V_ = np.tile(Vi_,(len_s,1))
        dU_ = np.tile(dUi_,(len_s,1))
        dV_ = np.tile(dVi_,(len_s,1))
        d2U_ = np.tile(d2Ui_,(len_s,1))
        d2V_ = np.tile(d2Vi_,(len_s,1))
        r = np.tile(r,(len_s,1))

        # Initializing the omegas which will be used very often. Might cut down on some computational time
        om0 = om(l,0)
        om0_ = om(l_,0)
        om2 = om(l,2)
        om2_ = om(l_,2)
        om3 = om(l,3)
        om3_ = om(l_,3)

        #B-- EXPRESSION
        Bmm = self.wig_red(2,-2,0)*om0_*om2_*(V_*(3.*U-2.*om0**2 *V + 3.*r*dU) - r*U*dV_)
        Bmm += self.wig_red(0,-2,2)*om0*om2*(V*(3.*U_-2.*om0_**2 *V_ + 3.*r*dU_) - r*U_*dV)
        Bmm += self.wig_red(1,-2,1)*om0_*om0*(3.*U*V_+3.*U_*V-2.*om0_**2 *V_*V - 2.*om0**2 *V_*V \
                + om2_**2*V_*V + om2**2 *V_*V + r*V*dU_ + r*V_*dU - r*U*dV_ - r*U_*dV - 2.*U_*U)
        Bmm += self.wig_red(3,-2,-1)*om0*om0_*om2_*om3_*V_*V
        Bmm += self.wig_red(-1,-2,3)*om0_*om0*om2*om3*V_*V   

        Bmm = 0.5*(((-1)**np.abs(1+self.mm))*prefac)[:,:,np.newaxis] \
                 * Bmm[np.newaxis,:,:]

        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)*om0_*(4.*om0**2 *V_*V + U_*(8.*U-5.*om0**2* V) - 3.*r*om0**2*V*dV_ \
                + 2*r**2*dU*dV_ - r*om0**2 *V_*dV + r**2 *V_*d2U + U*((-6.-2.*om0_**2+om0**2)*V_ + r*(4.*dV_-r*d2V_)))
        B0m += self.wig_red(0,-1,1)*om0*(4.*om0_**2 *V*V_ + U*(8.*U_-5.*om0_**2 *V_) - 3.*r*om0_**2 *V_*dV \
                + 2*r**2*dU_*dV - r*om0_**2 *V*dV_ + r**2 *V*d2U_ + U_*((-6.-2.*om0**2+om0_**2)*V + r*(4.*dV-r*d2V)))
        B0m += self.wig_red(-1,-1,2)*om0*om0_*om2*(U*V_ + V*(U_-4.*V_+3.*r*dV_) + r*V_*dV)
        B0m += self.wig_red(2,-1,-1)*om0_*om0*om2_*(U_*V + V_*(U-4.*V+3.*r*dV) + r*V*dV_)

        B0m = (0.25*((-1)**np.abs(self.mm))*prefac)[:,:,np.newaxis] \
                * B0m[np.newaxis,:]
        

        #B00 EXPRESSION
        B00 = self.wig_red(0,0,0)*2.*(-2.*r*U*dU_ - 2.*r*U_*dU + om0**2*r*V*dU_ + om0_**2*r*V_*dU - 5.*om0_**2*V_*U \
                - 5.*om0**2*V*U_ + 4.*om0**2*om0_**2*V_*V + om0_**2*r*U*dV_ + om0**2*r*U_*dV + 6.*U_*U)
        B00 += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*(-1.*om0_*om0)*(-U_*V-U*V_+2.*V_*V+r*V*dU_\
                +r*V_*dU-2.*r*V*dV_-2*r*V_*dV+r*U*dV_+r*U_*dV+2*r**2 *dV_*dV)

        B00 = (0.5*((-1)**np.abs(self.mm))*prefac)[:,:,np.newaxis] \
                * B00[np.newaxis,:]

        #B+- EXPRESSION
        Bpm = self.wig_red(0,0,0)*2.*(-2.*r*dU_*U-2.*r*dU*U_+om0**2*r*dU_*V+om0_**2*r*dU*V_-2.*r**2*dU_*dU \
                -om0_**2*U*V_-om0**2*U_*V+om0_**2*r*U*dV_+om0**2*r*U_*dV + 2.*U_*U)
        Bpm += (self.wig_red(-2,0,2)+self.wig_red(2,0,-2))*(-4.*om0*om0_*om2*om2_*V_*V)
        Bpm += (self.wig_red(-1,0,1)+self.wig_red(1,0,-1))*(om0*om0_)*(-r*V*dU_-r*V_*dU-V_*U-V*U_+r*U*dV_+r*U_*dV+2.*U_*U)

        Bpm = (0.25*((-1)**np.abs(self.mm))*prefac)[:,:,np.newaxis] \
                * Bpm[np.newaxis,:]

        #constructing the other two components of the kernel
        Bpp = parity_fac[:,:,np.newaxis]*Bmm
        Bp0 = parity_fac[:,:,np.newaxis]*B0m


        if(a_coeffkerns == True): 
            return rho,Bmm,B0m,B00,Bpm,Bp0,Bpp
        else:
            return Bmm,B0m,B00,Bpm,Bp0,Bpp


def plot_kern_diff(r, kern1, kern2, s, comp_idx):
    kern_title = np.array(['--', '0-', '00', '+-'])
    fig_title = f'$\mathcal{{B}}_{s}^{{{kern_title[comp_idx]}}}(n=-162,\, \ell=2,\, m=0)$'
    plt.figure()
    # plt.semilogy(r, np.abs((kern1 - kern2)/kern2), 'r')
    plt.semilogy(r, np.abs(kern1), 'r')
    plt.semilogy(r, np.abs(kern2), '--k')
    plt.title(fig_title)
    plt.grid()
    plt.tight_layout()

def test_computed_kernels(comp_kernels, r, isaxissymmkern=False):
    '''
    Function to compare the computed kernels for n=-162, ell=2 and
    nu_nl = 120.176 muHz. The precomputed kernels being compared against
    was computed by Shatanik in August 2023.

    Parameters:
    -----------
    comp_kernels : list of array_like, each array has shape (m, m_, s, r)
                   The list of kernel arrays with components of the list being
                   [0] Bmm, [1] B0m, [2] B00, [3] Bpm

    r : array_like of floats
        The array containing the radial grid.
    '''

    # loading the kernels from the test directory (these data files should be in Github)
    ref_kernels_s0 = np.loadtxt('../sample_eigenfunctions/Srijan_kernel0_n-162_l2_nu120176.txt')[:,1:]
    ref_kernels_s2 = np.loadtxt('../sample_eigenfunctions/Srijan_kernel2_n-162_l2_nu120176.txt')[:,1:]

    if(isaxissymmkern):
        # comparing individual kernel components
        for i in range(4):
            # testing s=2 for that kernel components
            # np.testing.assert_array_almost_equal(comp_kernels[i][2,2,1]/comp_kernels[i][2,2,1].max(),
            #                                      ref_kernels_s2[:,i]/ref_kernels_s2[:,i].max())
        
            # plotting the differences
            plot_kern_diff(r, comp_kernels[i][2,1]/comp_kernels[i][2,1].max(), ref_kernels_s2[:,i]/ref_kernels_s2[:,i].max(), 2, i)

        for i in range(2,4):
            # testing s=0 for that kernel components
            # np.testing.assert_array_almost_equal(comp_kernels[i][2,2,0]/comp_kernels[i][2,2,0].max(),
            #                                      ref_kernels_s0[:,i]/ref_kernels_s0[:,i].max())

            # plotting the differences
            plot_kern_diff(r, comp_kernels[i][2,0]/comp_kernels[i][2,0].max(), ref_kernels_s0[:,i]/ref_kernels_s0[:,i].max(), 0, i)
    else:
        # comparing individual kernel components
        for i in range(4):
            # testing s=2 for that kernel components
            # np.testing.assert_array_almost_equal(comp_kernels[i][2,2,1]/comp_kernels[i][2,2,1].max(),
            #                                      ref_kernels_s2[:,i]/ref_kernels_s2[:,i].max())
        
            # plotting the differences
            plot_kern_diff(r, comp_kernels[i][2,2,1]/comp_kernels[i][2,2,1].max(), ref_kernels_s2[:,i]/ref_kernels_s2[:,i].max(), 2, i)

        for i in range(2,4):
            # testing s=0 for that kernel components
            # np.testing.assert_array_almost_equal(comp_kernels[i][2,2,0]/comp_kernels[i][2,2,0].max(),
            #                                      ref_kernels_s0[:,i]/ref_kernels_s0[:,i].max())

            # plotting the differences
            plot_kern_diff(r, comp_kernels[i][2,2,0]/comp_kernels[i][2,2,0].max(), ref_kernels_s0[:,i]/ref_kernels_s0[:,i].max(), 0, i)



if __name__ == '__main__':
    r_norm_Rstar = np.loadtxt('../sample_eigenfunctions/rn-162.txt')
    r_raw, rho_raw = np.loadtxt('../sample_eigenfunctions/r_rho.txt').T
    rho = np.interp(r_norm_Rstar, r_raw/r_raw[-1], rho_raw)

    n = 162
    ell = 2

    s = np.array([0, 2])

    # initializing the kernel class for a specific field geometry and a radial grid
    make_kern_s = magkerns(s, r_norm_Rstar, rho)

    # calling the generic kernel computation
    # kern = make_kern_s.ret_kerns(n, ell, np.arange(-ell,ell+1))
    # calling the axisymmetric field kernel computation
    kern = make_kern_s.ret_kerns_axis_symm(n, ell, np.arange(-ell,ell+1))

    # benchmarking the kernel computation
    test_computed_kernels(kern, r_norm_Rstar, isaxissymmkern=True)