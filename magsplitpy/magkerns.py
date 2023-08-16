# code where the Lorentz-stress kernels are compouted from eigenfunctions
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import h5py

from magsplitpy import misc_funcs as fn

NAX = np.newaxis

class magkerns:
    '''
    Class to generate the Lorentz-stress kernels according to Das et al 2020.
    '''
    def __init__(self, s, r, axis_symm=True):
        '''
        The class is defined according to the field topology (denoted by s) and the radius and density grids.

        Parameters:
        -----------
        s : int
            The angular degree of the Lorentz-stress field we are interested in. 
        r : int
            The radius grid on which we want to compute the kernels. Same as the grid on which we have the eigenfunctions.
        axis_symm : bool, optional
                    By default we assume that the field is axisymmetric about the chosen coordinate axis.
        '''
        self.s = s
        # self.t = np.arange(-self.s, self.s+1)
        self.r = r

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
        # wig_vect = np.vectorize(fn.wig,otypes=[float])
        # return wig_vect(self.l_,self.ss_o,self.l,m1,m2,m3)
        return fn.w3j_vecm(self.l_,self.ss_o,self.l,m1,m2,m3)

    def wig_red(self, m1, m2, m3):
        '''3j symbol with upper row fixed (inner)'''
        # wig_vect = np.vectorize(fn.wig,otypes=[float])
        # return wig_vect(self.l_,self.ss_i,self.l,m1,m2,m3)
        # return fn.w3j_vecm(self.l_,self.ss_i,self.l,m1,m2,m3)
        return fn.w3j_vecm(self.l_,self.s,self.l,m1,m2,m3)

    def ret_kerns(self, n, l, m, Ui, Vi, n_=None, l_=None, m_=None, Ui_=None, Vi_=None, smoothen=False):    
        '''
        Function to return kernels for general configuration fields in a star computed from 
        eigenfunctions using expressions (C31) - (C44) of Das et al 2020.

        Parameters:
        -----------
        n : int
            Radial order of the first multiplet.
        l : int
            Angular degree of the first multiplet.
        m : array_like of int
            Array of azimuthal orders of the first multiplet. Usually (-2l+1, ..., 0, ... 2l+1)
        n_ : int, optional
            Radial order of the second multiplet. By default, we consider self coupling. So, n_ = n if n_ is not specified.
        l_ : int, optional
            Angular degree of the second multiplet. By default, we consider self coupling. So, l_ = l if l_ is not specified.
        m_ : array_like of int, optional
            Array of azimuthal orders of the second multiplet. By default, we consider self coupling. So, m_ = m if l_ is not specified.
            Note that although we would set m = m_ for self-coupling case (meaning we are interested in the same set of modes of the 
            coupled multiplet), the result would have a meshgrid in m x m_ and not just m = m_ case (which is for axisymmetric case). For
            a purely axisymmetric field, we have a different function called ret_kerns_axis_symm().
        smoothen : If we need to smoothen the eigenfunctions to avoid unreasonably large values from spurious jumps in the eigenfunctions 
                   obtained from the GYRE or other oscillation codes.

        Returns:
        --------
        Bmm : array_like, shape (m x m_ x s x r) 
              The -- component of the Lorentz stress sensitivity kernels.
        B0m : array_like, shape (m x m_ x s x r) 
              The 0- component of the Lorentz stress sensitivity kernels.
        B00 : array_like, shape (m x m_ x s x r) 
              The 00 component of the Lorentz stress sensitivity kernels.
        Bpm : array_like, shape (m x m_ x s x r) 
              The +- component of the Lorentz stress sensitivity kernels.
        Bp0 : array_like, shape (m x m_ x s x r) 
              The +0 component of the Lorentz stress sensitivity kernels.
        Bpp : array_like, shape (m x m_ x s x r) 
              The ++ component of the Lorentz stress sensitivity kernels.

        In later versions of the code, returning Bp0 and Bpp should be deprecated since these can be generated from 
        B0m and Bmm using parity factors, respectively.
        '''

        self.Ui = Ui
        self.Vi = Vi

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

            #re-assigning with smoothened variables
            r = r_new

        
        ###############################################################
        #no smoothing
        else: 
            r = self.r
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
        Bmm = self.wig_red(2,-2,0)[:,NAX]*om0_*om2_*(V_*(3.*U-2.*om0**2 *V + 3.*r*dU) - r*U*dV_)
        Bmm += self.wig_red(0,-2,2)[:,NAX]*om0*om2*(V*(3.*U_-2.*om0_**2 *V_ + 3.*r*dU_) - r*U_*dV)
        Bmm += self.wig_red(1,-2,1)[:,NAX]*om0_*om0*(3.*U*V_+3.*U_*V-2.*om0_**2 *V_*V - 2.*om0**2 *V_*V \
                + om2_**2*V_*V + om2**2 *V_*V + r*V*dU_ + r*V_*dU - r*U*dV_ - r*U_*dV - 2.*U_*U)
        Bmm += self.wig_red(3,-2,-1)[:,NAX]*om0*om0_*om2_*om3_*V_*V
        Bmm += self.wig_red(-1,-2,3)[:,NAX]*om0_*om0*om2*om3*V_*V                

        Bmm = 0.5*(((-1)**np.abs(1+self.mm_))*prefac)[:,:,:,NAX] \
                 * Bmm[NAX,:,:]
    

        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)[:,NAX]*om0_*(4.*om0**2 *V_*V + U_*(8.*U-5.*om0**2* V) - 3.*r*om0**2*V*dV_ \
                + 2*r**2*dU*dV_ - r*om0**2 *V_*dV + r**2 *V_*d2U + U*((-6.-2.*om0_**2+om0**2)*V_ + r*(4.*dV_-r*d2V_)))
        B0m += self.wig_red(0,-1,1)[:,NAX]*om0*(4.*om0_**2 *V*V_ + U*(8.*U_-5.*om0_**2 *V_) - 3.*r*om0_**2 *V_*dV \
                + 2*r**2*dU_*dV - r*om0_**2 *V*dV_ + r**2 *V*d2U_ + U_*((-6.-2.*om0**2+om0_**2)*V + r*(4.*dV-r*d2V)))
        B0m += self.wig_red(-1,-1,2)[:,NAX]*om0*om0_*om2*(U*V_ + V*(U_-4.*V_+3.*r*dV_) + r*V_*dV)
        B0m += self.wig_red(2,-1,-1)[:,NAX]*om0_*om0*om2_*(U_*V + V_*(U-4.*V+3.*r*dV) + r*V*dV_)

        B0m = (0.25*((-1)**np.abs(self.mm_))*prefac)[:,:,:,NAX] \
                * B0m[NAX,:,:]


        #B00 EXPRESSION
        B00 = self.wig_red(0,0,0)[:,NAX]*2.*(-2.*r*U*dU_ - 2.*r*U_*dU + om0**2*r*V*dU_ + om0_**2*r*V_*dU - 5.*om0_**2*V_*U \
                - 5.*om0**2*V*U_ + 4.*om0**2*om0_**2*V_*V + om0_**2*r*U*dV_ + om0**2*r*U_*dV + 6.*U_*U)
        B00 += (self.wig_red(-1,0,1)+self.wig_red(1,0,-1))[:,NAX]*(-1.*om0_*om0)*(-U_*V-U*V_+2.*V_*V+r*V*dU_\
                +r*V_*dU-2.*r*V*dV_-2*r*V_*dV+r*U*dV_+r*U_*dV+2*r**2 *dV_*dV)

        B00 = (0.5*((-1)**np.abs(self.mm_))*prefac)[:,:,:,NAX] \
                * B00[NAX,:,:]


        #B+- EXPRESSION
        Bpm = self.wig_red(0,0,0)[:,NAX]*2.*(-2.*r*dU_*U-2.*r*dU*U_+om0**2*r*dU_*V+om0_**2*r*dU*V_-2.*r**2*dU_*dU \
                -om0_**2*U*V_-om0**2*U_*V+om0_**2*r*U*dV_+om0**2*r*U_*dV + 2.*U_*U)
        Bpm += (self.wig_red(-2,0,2)+self.wig_red(2,0,-2))[:,NAX]*(-4.*om0*om0_*om2*om2_*V_*V)
        Bpm += (self.wig_red(-1,0,1)+self.wig_red(1,0,-1))[:,NAX]*(om0*om0_)*(-r*V*dU_-r*V_*dU-V_*U-V*U_+r*U*dV_+r*U_*dV+2.*U_*U)


        Bpm = (0.25*((-1)**np.abs(self.mm_))*prefac)[:,:,:,NAX] \
                * Bpm[NAX,:,:]


        #constructing the other two components of the kernel
        Bpp = parity_fac[:,:,:,NAX]*Bmm
        Bp0 = parity_fac[:,:,:,NAX]*B0m

        return Bmm,B0m,B00,Bpm,Bp0,Bpp
        
    def ret_kerns_axis_symm(self, n, l, m, Ui, Vi, n_=None, l_=None, m_=None, Ui_=None, Vi_=None, smoothen = False, a_coeffkerns = False):
        '''
        Function to return kernels for axisymmetric fields in a star computed from 
        eigenfunctions using expressions (C31) - (C44) of Das et al 2020.

        Parameters:
        -----------
        n : int
            Radial order of the first multiplet.
        l : int
            Angular degree of the first multiplet.
        m : array_like of int
            Array of azimuthal orders of the first multiplet. Usually (-2l+1, ..., 0, ... 2l+1)
        n_ : int, optional
            Radial order of the second multiplet. By default, we consider self coupling. So, n_ = n if n_ is not specified.
        l_ : int, optional
            Angular degree of the second multiplet. By default, we consider self coupling. So, l_ = l if l_ is not specified.
        smoothen : If we need to smoothen the eigenfunctions to avoid unreasonably large values from spurious jumps in the eigenfunctions 
                   obtained from the GYRE or other oscillation codes.
        a_coeffkerns : Modified kernels when using a-coefficients instead of frequency splittings. See Eqn (33) in Das et al 2020.

        Returns:
        --------
        Bmm : array_like, shape (m x s x r) 
              The -- component of the Lorentz stress sensitivity kernels.
        B0m : array_like, shape (m x s x r) 
              The 0- component of the Lorentz stress sensitivity kernels.
        B00 : array_like, shape (m x s x r) 
              The 00 component of the Lorentz stress sensitivity kernels.
        Bpm : array_like, shape (m x s x r) 
              The +- component of the Lorentz stress sensitivity kernels.
        Bp0 : array_like, shape (m x s x r) 
              The +0 component of the Lorentz stress sensitivity kernels.
        Bpp : array_like, shape (m x s x r) 
              The ++ component of the Lorentz stress sensitivity kernels.

        In later versions of the code, returning Bp0 and Bpp should be deprecated since these can be generated from 
        B0m and Bmm using parity factors, respectively.
        '''

        self.Ui = Ui
        self.Vi = Vi

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

            #re-assigning with smoothened variables
            r = r_new
        
        #----------------------no smoothing------------------------------#

        else:
            r = self.r
            Ui = self.Ui 
            Vi = self.Vi 
            Ui_ = self.Ui_ 
            Vi_ = self.Vi_ 

            dUi, dVi = np.gradient(Ui,r), np.gradient(Vi,r)
            d2Ui,d2Vi = np.gradient(dUi,r), np.gradient(dVi,r)
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
        Bmm = self.wig_red(2,-2,0)[:,NAX]*om0_*om2_*(V_*(3.*U-2.*om0**2 *V + 3.*r*dU) - r*U*dV_)
        Bmm += self.wig_red(0,-2,2)[:,NAX]*om0*om2*(V*(3.*U_-2.*om0_**2 *V_ + 3.*r*dU_) - r*U_*dV)
        Bmm += self.wig_red(1,-2,1)[:,NAX]*om0_*om0*(3.*U*V_+3.*U_*V-2.*om0_**2 *V_*V - 2.*om0**2 *V_*V \
                + om2_**2*V_*V + om2**2 *V_*V + r*V*dU_ + r*V_*dU - r*U*dV_ - r*U_*dV - 2.*U_*U)
        Bmm += self.wig_red(3,-2,-1)[:,NAX]*om0*om0_*om2_*om3_*V_*V
        Bmm += self.wig_red(-1,-2,3)[:,NAX]*om0_*om0*om2*om3*V_*V   

        Bmm = 0.5*(((-1)**np.abs(1+self.mm))*prefac)[:,:,NAX] \
                 * Bmm[NAX,:,:]

        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)[:,NAX]*om0_*(4.*om0**2 *V_*V + U_*(8.*U-5.*om0**2* V) - 3.*r*om0**2*V*dV_ \
                + 2*r**2*dU*dV_ - r*om0**2 *V_*dV + r**2 *V_*d2U + U*((-6.-2.*om0_**2+om0**2)*V_ + r*(4.*dV_-r*d2V_)))
        B0m += self.wig_red(0,-1,1)[:,NAX]*om0*(4.*om0_**2 *V*V_ + U*(8.*U_-5.*om0_**2 *V_) - 3.*r*om0_**2 *V_*dV \
                + 2*r**2*dU_*dV - r*om0_**2 *V*dV_ + r**2 *V*d2U_ + U_*((-6.-2.*om0**2+om0_**2)*V + r*(4.*dV-r*d2V)))
        B0m += self.wig_red(-1,-1,2)[:,NAX]*om0*om0_*om2*(U*V_ + V*(U_-4.*V_+3.*r*dV_) + r*V_*dV)
        B0m += self.wig_red(2,-1,-1)[:,NAX]*om0_*om0*om2_*(U_*V + V_*(U-4.*V+3.*r*dV) + r*V*dV_)

        B0m = (0.25*((-1)**np.abs(self.mm))*prefac)[:,:,NAX] \
                * B0m[NAX,:]
        

        #B00 EXPRESSION
        B00 = self.wig_red(0,0,0)[:,NAX]*2.*(-2.*r*U*dU_ - 2.*r*U_*dU + om0**2*r*V*dU_ + om0_**2*r*V_*dU - 5.*om0_**2*V_*U \
                - 5.*om0**2*V*U_ + 4.*om0**2*om0_**2*V_*V + om0_**2*r*U*dV_ + om0**2*r*U_*dV + 6.*U_*U)
        B00 += (self.wig_red(-1,0,1)+self.wig_red(1,0,-1))[:,NAX]*(-1.*om0_*om0)*(-U_*V-U*V_+2.*V_*V+r*V*dU_\
                +r*V_*dU-2.*r*V*dV_-2*r*V_*dV+r*U*dV_+r*U_*dV+2*r**2 *dV_*dV)

        B00 = (0.5*((-1)**np.abs(self.mm))*prefac)[:,:,NAX] \
                * B00[NAX,:]

        #B+- EXPRESSION
        Bpm = self.wig_red(0,0,0)[:,NAX]*2.*(-2.*r*dU_*U-2.*r*dU*U_+om0**2*r*dU_*V+om0_**2*r*dU*V_-2.*r**2*dU_*dU \
                -om0_**2*U*V_-om0**2*U_*V+om0_**2*r*U*dV_+om0**2*r*U_*dV + 2.*U_*U)
        Bpm += (self.wig_red(-2,0,2)+self.wig_red(2,0,-2))[:,NAX]*(-4.*om0*om0_*om2*om2_*V_*V)
        Bpm += (self.wig_red(-1,0,1)+self.wig_red(1,0,-1))[:,NAX]*(om0*om0_)*(-r*V*dU_-r*V_*dU-V_*U-V*U_+r*U*dV_+r*U_*dV+2.*U_*U)

        Bpm = (0.25*((-1)**np.abs(self.mm))*prefac)[:,:,NAX] \
                * Bpm[NAX,:]

        #constructing the other two components of the kernel
        Bpp = parity_fac[:,:,NAX]*Bmm
        Bp0 = parity_fac[:,:,NAX]*B0m


        return Bmm,B0m,B00,Bpm,Bp0,Bpp

def plot_kern_components(r, kern, n, ell):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16,10))

    kernel_components = np.array(['--','0-','00','+-'])

    mag_comp = np.array(['$B_{\\theta}\,B_{\phi}$ or $B_{\\theta}^2 - B_{\phi}^2$',
                         '$B_r \, B_{\\theta}$ or $B_r \, B_{\phi}$',
                         '$B_r^2$',
                         '$B_{\\theta}^2 + B_{\phi}^2$'])

    for i in range(4):
        row, col = i//2, i%2
        # plotting the m = 0 component
        ax[row,col].set_yscale('symlog')
        ax[row,col].plot(r, kern[i][4, 0], 'k', label='$s=0$')
        ax[row,col].plot(r, kern[i][4, 1], 'r', label='$s=2$')
        ax[row,col].grid(True)
        ax[row,col].set_xlim([0,1])
        ax[row,col].text(0.1, 0.9, f'$K^{{{kernel_components[i]}}}$ for {mag_comp[i]}',
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform = ax[row,col].transAxes, fontsize=16)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(r"$r/R_{\star}$", fontsize=16, labelpad=15)
    plt.ylabel("Kernel in arbitrary units", fontsize=16, labelpad=20)

    # making a single legend for all subplots
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=16)

    plt.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.93, wspace=0.1, hspace=0.1)
    plt.title(f'Mode: $n=${n}, $\ell$={ell}', fontsize=20)

    # plt.savefig(f'VincentKern_n={n}_ell={ell}.pdf')

def plot_kern_diff(r, kern1, kern2, s, comp_idx):
    '''
    Function to compare the different components of two kernels (possibly computed from
    two different sources). Mostly used for visual benchmarking of kernel computation. No 
    m index mentioned since as of now we pass kernels for m=0 only.

    Parameters:
    -----------
    r : array_like
        Radial grid to plot the kernel (grid on which the kernel has been processed).
    kern1 : array_like, shape (r,)
            First kernel.
    kern2 : array_like, shape (r,)
            Second kernel.
    s : int
        The angular degree of perturbation for which the kernels are passed.
    comp_idx : int
               The component demarcating the index of the geometry of magnetic field 
               for which the kernels are passed. 0 = --, 1 = 0-, 2 = 00, 3 = +- 
    '''
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
    benchmarking = False

     #----------------------for the benchmarking case------------------------------#
    if(benchmarking):
        # reading the radius grid
        r_norm_Rstar = np.loadtxt('../sample_eigenfunctions/rn-162.txt')
        # desired mode
        n = 162
        ell = 2
        # loading eigenfunctions here
        Ui_raw = np.loadtxt('../sample_eigenfunctions/Un-162.txt')
        Vi_raw = np.loadtxt('../sample_eigenfunctions/Vn-162.txt')

    #----------------------when running the code in NOT benghmark mode------------------------------#
    else:
        # desired mode
        n_str = '-2'
        ell_str = '2'
        # loading the corresponding file
        eigfile = h5py.File(f'../Vincent_Eig/mode_h.{ell_str}_{n_str}_hz.h5')
        # the radius grid
        r_norm_Rstar = eigfile['x'][()]   # reading it off a random file since its the same for all
        rho = eigfile['rho'][()]
        # loading eigenfunctions here
        Ui_raw = eigfile['xi_r']['re'][()]
        Vi_raw = eigfile['xi_h']['re'][()]
        # # normalizing eigenfunctions at the outset
        # eignorm = fn.eignorm(Ui_raw, Vi_raw, int(ell_str), r_norm_Rstar, rho)
        # Ui_raw = Ui_raw / eignorm
        # Vi_raw = Vi_raw / eignorm

        # converting n and ell back to integers
        n, ell = int(n_str), int(ell_str)

    # the desired s values
    s = np.array([0, 2])

    # initializing the kernel class for a specific field geometry and a radial grid
    make_kern_s = magkerns(s, r_norm_Rstar)

    # calling the generic kernel computation
    # kern = make_kern_s.ret_kerns(n, ell, np.arange(-ell,ell+1))
    # calling the axisymmetric field kernel computation
    kern = make_kern_s.ret_kerns_axis_symm(n, ell, np.arange(-ell,ell+1), Ui_raw, Vi_raw)

    # benchmarking the kernel computation
    if(benchmarking):
        test_computed_kernels(kern, r_norm_Rstar, isaxissymmkern=True)

    else:
        plot_kern_components(r_norm_Rstar, kern, n_str, ell_str)