# code where the Lorentz-stress kernels are compouted from eigenfunctions

import numpy as np

class magkerns:
    def __init__(self, s, r):
        self.s = s
        self.t = np.arange(-self.s, self.s+1)
        self.r = r

    def ret_kerns(self, n, l, m, smoothen=False):
        # since it will always be self-coupling for asteroseismology
        n_, l_, m_ = n, l, m

        # loading the eigenfunctions
        # Ui, Vi = 
        nl = fn.find_nl(n,l)
        nl_ = fn.find_nl(n_,l_)

        len_m, len_m_, len_s = np.shape(self.ss_o)

        #Savitsky golay filter for smoothening
        window = 45  #must be odd
        order = 3

        if(nl == None or nl_ == None):
            print("Mode not found. Exiting."); exit()


        tstamp()
        om = np.vectorize(fn.omega,otypes=[float])
        parity_fac = (-1)**(l+l_+self.ss_o) #parity of selected modes
        prefac = 1./(4.* np.pi) * np.sqrt((2*l_+1.) * (2*self.ss_o+1.) * (2*l+1.) \
                    / (4.* np.pi)) * self.wig_red_o(-m_,m_-m,m)


        #EIGENFUCNTION DERIVATIVES

        ##########################################################3
        #smoothingR2 = 0.78
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
        
        # np.save('prefac.npy',prefac)

        Bmm = 0.5*(((-1)**np.abs(1+m_))*prefac)[:,:,:,np.newaxis] \
                 * (Bmm/r**2)[np.newaxis,:,:]
    


        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)*om0_*(4.*om0**2 *V_*V + U_*(8.*U-5.*om0**2* V) - 3.*r*om0**2*V*dV_ \
                + 2*r**2*dU*dV_ - r*om0**2 *V_*dV + r**2 *V_*d2U + U*((-6.-2.*om0_**2+om0**2)*V_ + r*(4.*dV_-r*d2V_)))
        B0m += self.wig_red(0,-1,1)*om0*(4.*om0_**2 *V*V_ + U*(8.*U_-5.*om0_**2 *V_) - 3.*r*om0_**2 *V_*dV \
                + 2*r**2*dU_*dV - r*om0_**2 *V*dV_ + r**2 *V*d2U_ + U_*((-6.-2.*om0**2+om0_**2)*V + r*(4.*dV-r*d2V)))
        B0m += self.wig_red(-1,-1,2)*om0*om0_*om2*(U*V_ + V*(U_-4.*V_+3.*r*dV_) + r*V_*dV)
        B0m += self.wig_red(2,-1,-1)*om0_*om0*om2_*(U_*V + V_*(U-4.*V+3.*r*dV) + r*V*dV_)

        B0m = (0.25*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B0m/r**2)[np.newaxis,:,:]

        
#        print(np.shape(self.wig_red(-1,-0,1)))
#        exit()

        #B00 EXPRESSION
        B00 = self.wig_red(0,0,0)*2.*(-2.*r*U*dU_ - 2.*r*U_*dU + om0**2*r*V*dU_ + om0_**2*r*V_*dU - 5.*om0_**2*V_*U \
                - 5.*om0**2*V*U_ + 4.*om0**2*om0_**2*V_*V + om0_**2*r*U*dV_ + om0**2*r*U_*dV + 6.*U_*U)
        B00 += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*(-1.*om0_*om0)*(-U_*V-U*V_+2.*V_*V+r*V*dU_\
                +r*V_*dU-2.*r*V*dV_-2*r*V_*dV+r*U*dV_+r*U_*dV+2*r**2 *dV_*dV)

        B00 = (0.5*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (B00/r**2)[np.newaxis,:,:]

        #B+- EXPRESSION
        Bpm = self.wig_red(0,0,0)*2.*(-2.*r*dU_*U-2.*r*dU*U_+om0**2*r*dU_*V+om0_**2*r*dU*V_-2.*r**2*dU_*dU \
                -om0_**2*U*V_-om0**2*U_*V+om0_**2*r*U*dV_+om0**2*r*U_*dV + 2.*U_*U)
        Bpm += (self.wig_red(-2,0,2)+self.wig_red(2,0,-2))*(-4.*om0*om0_*om2*om2_*V_*V)
        Bpm += (self.wig_red(-1,0,1)+self.wig_red(1,0,-1))*(om0*om0_)*(-r*V*dU_-r*V_*dU-V_*U-V*U_+r*U*dV_+r*U_*dV+2.*U_*U)


        Bpm = (0.25*((-1)**np.abs(m_))*prefac)[:,:,:,np.newaxis] \
                * (Bpm/r**2)[np.newaxis,:,:]


        # print(prefac.dtype,parity_fac.dtype,Bmm.dtype,B0m.dtype,B00.dtype,Bpm.dtype)

        #constructing the other two components of the kernel
        Bpp = parity_fac[:,:,:,np.newaxis]*Bmm
        Bp0 = parity_fac[:,:,:,np.newaxis]*B0m

        return Bmm,B0m,B00,Bpm,Bp0,Bpp
        
    def ret_kerns_axis_symm(self,smoothen = False, a_coeffkerns = False):
        n,l,m,n_,l_ = self.n, self.l, self.mm, self.n_, self.l_
        m_ = m
        r_start , r_end = self.r_range
        nl = fn.find_nl(n,l)
        nl_ = fn.find_nl(n_,l_)

        len_m, len_s = np.shape(self.ss_o)

        #Savitsky golay filter for smoothening
        window = 45  #must be odd
        order = 3

        if(nl == None or nl_ == None):
            print("Mode not found. Exiting."); exit()

        om = np.vectorize(fn.omega,otypes=[float])
        parity_fac = (-1)**(l+l_+self.ss_o) #parity of selected modes
        if(a_coeffkerns == True):
            prefac = ((-1.)**l)/(4.* np.pi) * np.sqrt((2*l_+1.) * (2*self.ss_o+1.) * (2*l+1.) \
                    / (4.* np.pi)) * self.wig_red_o(-l,0,l) / l 
        else:
            prefac = 1./(4.* np.pi) * np.sqrt((2*l_+1.) * (2*self.ss_o+1.) * (2*l+1.) \
                    / (4.* np.pi)) * self.wig_red_o(-m,0,m)

        #EIGENFUCNTION DERIVATIVES

        ###################################################################
        #smoothing

        #interpolation params

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
        

        ######################################################################
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

        Bmm = 0.5*(((-1)**np.abs(1+m_))*prefac)[:,:,np.newaxis] \
                 * (Bmm/r**2)[np.newaxis,:,:]

        #B0- EXPRESSION
        B0m = self.wig_red(1,-1,0)*om0_*(4.*om0**2 *V_*V + U_*(8.*U-5.*om0**2* V) - 3.*r*om0**2*V*dV_ \
                + 2*r**2*dU*dV_ - r*om0**2 *V_*dV + r**2 *V_*d2U + U*((-6.-2.*om0_**2+om0**2)*V_ + r*(4.*dV_-r*d2V_)))
        B0m += self.wig_red(0,-1,1)*om0*(4.*om0_**2 *V*V_ + U*(8.*U_-5.*om0_**2 *V_) - 3.*r*om0_**2 *V_*dV \
                + 2*r**2*dU_*dV - r*om0_**2 *V*dV_ + r**2 *V*d2U_ + U_*((-6.-2.*om0**2+om0_**2)*V + r*(4.*dV-r*d2V)))
        B0m += self.wig_red(-1,-1,2)*om0*om0_*om2*(U*V_ + V*(U_-4.*V_+3.*r*dV_) + r*V_*dV)
        B0m += self.wig_red(2,-1,-1)*om0_*om0*om2_*(U_*V + V_*(U-4.*V+3.*r*dV) + r*V*dV_)

        B0m = (0.25*((-1)**np.abs(m_))*prefac)[:,:,np.newaxis] \
                * (B0m/r**2)[np.newaxis,:]
        
#        print(np.shape(self.wig_red(-1,-0,1)))
#        exit()

        #B00 EXPRESSION
        B00 = self.wig_red(0,0,0)*2.*(-2.*r*U*dU_ - 2.*r*U_*dU + om0**2*r*V*dU_ + om0_**2*r*V_*dU - 5.*om0_**2*V_*U \
                - 5.*om0**2*V*U_ + 4.*om0**2*om0_**2*V_*V + om0_**2*r*U*dV_ + om0**2*r*U_*dV + 6.*U_*U)
        B00 += (self.wig_red(-1,0,1) + self.wig_red(1,0,-1))*(-1.*om0_*om0)*(-U_*V-U*V_+2.*V_*V+r*V*dU_\
                +r*V_*dU-2.*r*V*dV_-2*r*V_*dV+r*U*dV_+r*U_*dV+2*r**2 *dV_*dV)

        B00 = (0.5*((-1)**np.abs(m_))*prefac)[:,:,np.newaxis] \
                * (B00/r**2)[np.newaxis,:]

        #B+- EXPRESSION
        Bpm = self.wig_red(0,0,0)*2.*(-2.*r*dU_*U-2.*r*dU*U_+om0**2*r*dU_*V+om0_**2*r*dU*V_-2.*r**2*dU_*dU \
                -om0_**2*U*V_-om0**2*U_*V+om0_**2*r*U*dV_+om0**2*r*U_*dV + 2.*U_*U)
        Bpm += (self.wig_red(-2,0,2)+self.wig_red(2,0,-2))*(-4.*om0*om0_*om2*om2_*V_*V)
        Bpm += (self.wig_red(-1,0,1)+self.wig_red(1,0,-1))*(om0*om0_)*(-r*V*dU_-r*V_*dU-V_*U-V*U_+r*U*dV_+r*U_*dV+2.*U_*U)

        Bpm = (0.25*((-1)**np.abs(m_))*prefac)[:,:,np.newaxis] \
                * (Bpm/r**2)[np.newaxis,:]

        #constructing the other two components of the kernel
        Bpp = parity_fac[:,:,np.newaxis]*Bmm
        Bp0 = parity_fac[:,:,np.newaxis]*B0m


        if(a_coeffkerns == True): 
            return rho,Bmm,B0m,B00,Bpm,Bp0,Bpp
        else:
            return Bmm,B0m,B00,Bpm,Bp0,Bpp