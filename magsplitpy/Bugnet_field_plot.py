import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.pyplot as plt

lam   = 2.80
R_rad = 0.136391


d_temp = np.loadtxt('./profile'+str(50)+'.data.GYRE',skiprows=1,usecols=(1,6)).T

theta_vals_interp   = d_temp[0]/max(d_temp[0])
rho_vals_interp = d_temp[1]

rho_interp = interp1d( theta_vals_interp , rho_vals_interp )
plt.plot( theta_vals_interp , rho_vals_interp )
#plt.xscale('log')
#plt.show()

####################################################################

def j1(r):
    arg = lam * r / R_rad
    return spherical_jn(1,arg)


def y1(r):
    arg = lam * r / R_rad
    return spherical_yn(1,arg)

####################################################################

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
 
###################################################################
    
def A(r):
    term1a = - r * j1(r)
    term1b = Y_integral(r, R_rad)
    term1  = term1a * term1b
    
    term2a = - r * y1(r)
    term2b = J_integral(0, r)
    term2  = term2a * term2b
        
    return term1 + term2
    
#my_r_vals = np.linspace(1e-5,1.,1000)
my_r_vals = np.logspace(np.log10(1e-4),np.log10(1),1000)

#plt.plot( A(my_r_vals) )

plt.plot(my_r_vals,rho_interp(my_r_vals))
plt.yscale('log')
plt.xscale('log')
plt.show()

A_array = []
Br_array = []

for r_val in my_r_vals:
    A_array.append( A(r_val) )
    

mask2 = np.where((my_r_vals[:] >= R_rad))
A_array = np.array( A_array )
A_array[mask2] = 0.

Br_array = 2 * A_array / my_r_vals**2
Bt_array = - np.gradient( A_array , my_r_vals ) / my_r_vals
Bp_array = lam/R_rad * A_array / my_r_vals

Bt_array[0]=Bt_array[1]

'''
# Normalized the profile with max[ br(r) ]
Br_array = Br_array/max(Br_array)
Bt_array = Bt_array/max(Br_array)
Bp_array = Bp_array/max(Br_array)
'''
my_r_vals= np.append(0,my_r_vals)
Br_array = np.append(Br_array[0],Br_array)
Bt_array = np.append(Bt_array[0],Bt_array)
Bp_array = np.append(Bp_array[0],Bp_array)


Bt_array = Bt_array/np.max(Br_array)
Bp_array = Bp_array/np.max(Br_array)
Br_array = Br_array/np.max(Br_array)

B0 = 1#e6

'''
plt.axvspan(0.0055,0.0070,alpha=0.3,label='H-burning shell', color='grey')
plt.plot( my_r_vals, B0*Br_array, label=r'$B_{r}$' )
plt.plot( my_r_vals, B0*Bt_array, label=r'$B_{\theta}$' )
plt.plot( my_r_vals, B0*Bp_array, label=r'$B_{\phi}$' )
plt.axhline(0,color='red',linestyle=':')
plt.xscale('log')
#plt.xticks( my_r_vals , my_r_vals/R_rad )
plt.legend()
plt.savefig('./Bugnet_unnormalized_B0_1e0_log.pdf',bbox_inches='tight')
plt.show()
'''
plt.axvspan(0.0055,0.0070, color='grey')
plt.plot( my_r_vals, Br_array/max(Br_array), label=r'$B_{r}$' )
plt.plot( my_r_vals, Bt_array/max(Br_array), label=r'$B_{\theta}$' )
plt.plot( my_r_vals, Bp_array/max(Br_array), label=r'$B_{\phi}$' )
plt.axhline(0,color='red',linestyle=':')
plt.xscale('log')
#plt.xticks( my_r_vals , my_r_vals/R_rad )
plt.legend()
plt.show()
'''

plt.axvspan(0.0055,0.0070,alpha=0.3,label='H-burning shell', color='grey')
plt.plot( my_r_vals, B0*Br_array, label=r'$b_{r}$' )
plt.plot( my_r_vals, B0*Bt_array, label=r'$b_{\theta}$' )
plt.plot( my_r_vals, B0*Bp_array, label=r'$b_{\phi}$' )
plt.axhline(0,color='red',linestyle=':')

plt.xlabel(r'$r/R_{\rm{rad}}$',fontsize=13)
plt.ylabel('Amplitude',fontsize=13)
plt.xscale('linear')
#plt.xticks( np.linspace(0,R_rad,11) , np.linspace(0,R_rad,11)/R_rad )
plt.legend()
plt.savefig('./Bugnet_unnormalized_B0_1e0.pdf',bbox_inches='tight')
plt.show()

# For normalized amplitude
plt.axvspan(0.0055,0.0070,alpha=0.3,label='H-burning shell', color='grey')
plt.plot( my_r_vals, Br_array/max(Br_array), label=r'$B_{r}$' )
plt.plot( my_r_vals, Bt_array/max(Br_array), label=r'$B_{\theta}$' )
plt.plot( my_r_vals, Bp_array/max(Br_array), label=r'$B_{\phi}$' )
plt.axhline(0,color='red',linestyle=':')

plt.xlabel(r'$r/R_{\rm{rad}}$',fontsize=13)
plt.ylabel('Normalized Amplitude',fontsize=13)
plt.xscale('linear')
#plt.xticks( np.linspace(0,R_rad,11) , np.linspace(0,R_rad,11)/R_rad )
plt.legend()
plt.savefig('./Bugnet_normalized.pdf',bbox_inches='tight')
plt.show()

arr_to_save = np.array([ my_r_vals , Br_array , Bt_array , Bp_array  ]).T

np.savetxt('/home/shatanik/Desktop/Paper1/Lisa_fields/Lisa_field',arr_to_save,header='r (in units of R_star)\t br \t bt \t bp ')



r_min , r_max = 1e-5 , 1e-3
mask = np.where((my_r_vals[:] >= r_min) & (my_r_vals[:] <= r_max)) 
#print(mask)


br_H = Br_array[mask]
bp_H = Bp_array[mask]
plt.figure()
print("Mean deep br = ",simpson(br_H,my_r_vals[mask]) * 1e6)
print("Mean deep bp = ",simpson(bp_H,my_r_vals[mask]) * 1e6)

r_min , r_max = 1e-3 , 1e-2
mask = np.where((my_r_vals[:] >= r_min) & (my_r_vals[:] <= r_max)) 
#print(mask)


br_H = Br_array[mask]
bp_H = Bp_array[mask]
plt.figure()
print("Mean shell br = ",simpson(br_H,my_r_vals[mask]) * 1e6)
print("Mean shell bp = ",simpson(bp_H,my_r_vals[mask]) * 1e6)


r_min , r_max = 0.985 , 1e0
mask = np.where((my_r_vals[:] >= r_min) & (my_r_vals[:] <= r_max)) 
#print(mask)


br_H = Br_array[mask]
bp_H = Bp_array[mask]
plt.figure()
print("Mean surf br = ",np.mean(simpson(br_H,my_r_vals[mask])) * 1e6)
print("Mean surf bp = ",np.mean(simpson(bp_H,my_r_vals[mask])) * 1e6)


'''
