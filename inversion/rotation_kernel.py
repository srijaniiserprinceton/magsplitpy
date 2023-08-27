import numpy as np
import os
import h5py
from magsplitpy import misc_funcs as fn
from scipy.integrate import simps
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rc
font = {'size'   : 16}
rc('font', **font)

NAX = np.newaxis

os.environ["F90"] = "gfortran"
from avni.tools.bases import eval_splrem, eval_polynomial, eval_vbspl

class rot_kern:
    def __init__(self, dir_eigfiles, mode_nl1_arr, mode_nl2_arr=None, s_arr = np.array([1]), to_save_kernels=False,
                 custom_knot_num = 10, return_splined_kernel=False):
        # the directory containing all the eigenfunctions
        self.dir_eigfiles = dir_eigfiles
        
        # computing the relevant kernels
        self.nl1_arr = mode_nl1_arr.astype('int')

        # meaning if we only want self coupling
        if(mode_nl2_arr == None):
            self.nl2_arr = mode_nl1_arr
            self.self_coupling = True
        else:
            self.nl2_arr = mode_nl2_arr.astype('int')
            self.self_coupling = False

        # n and ell of mode files in the sequence in which they exist in memory
        self.files_n, self.files_ell = [], []
        self.find_file_idx()
                
        # the s array of rotation for which we want kernels
        self.s_arr = s_arr

        # flag whether to save kernels (useful for MCMC inversions)
        self.tosavekerns = to_save_kernels

        # finding r and rho correspoding to the most common radius grid
        self.r_common_grid, self.rho_common_grid = fn.find_mode_r_grid(self.dir_eigfiles)
        self.r_common_grid = self.r_common_grid / self.r_common_grid[-1]

        # creating a uniform grid of the same length and range as the most common grid
        self.r_bsp_basis_rot = np.linspace(self.r_common_grid.min(), self.r_common_grid.max(),
                                           len(self.r_common_grid))

        # making B-splines
        self.bsp_basis_rot = None 
        self.knot_locs_rot = None
        self.create_bsplines(custom_knot_num) 
        self.num_knots_rot = self.bsp_basis_rot.shape[0]
        self.return_splined_kernel = return_splined_kernel

        self.kern_dict = {}
        self.compute_all_mode_kerns()

    def find_file_idx(self):
        filenames = np.asarray(os.listdir(f'{self.dir_eigfiles}/'))

        for i, f in enumerate(filenames):
            eigfile = h5py.File(f'{self.dir_eigfiles}/' + f)
            # mode contained in the file
            n, ell = eigfile.attrs['n_pg'], eigfile.attrs['l']
            # storing the n and ell in the mode files according to their sequence
            self.files_n.append(n)
            self.files_ell.append(ell)

        self.files_n = np.array(self.files_n).astype('int')
        self.files_ell = np.array(self.files_ell).astype('int')


    def compute_all_mode_kerns(self):
        """
        Finding the mode files, reading the Ui, Vi and normalizing them
        before computing the rotation kernels.
        """

        filenames = np.asarray(os.listdir(f'{self.dir_eigfiles}/'))

        for kern_idx in range(len(self.nl1_arr)):
            file1_idx = np.where((self.files_n == self.nl1_arr[kern_idx,0]) *\
                                 (self.files_ell == self.nl1_arr[kern_idx,1]))[0][0]
            file2_idx = np.where((self.files_n == self.nl2_arr[kern_idx,0]) *\
                                 (self.files_ell == self.nl2_arr[kern_idx,1]))[0][0]

            f1 = filenames[file1_idx]
            f2 = filenames[file2_idx]

            eigfile1 = h5py.File(f'{self.dir_eigfiles}/' + f1)
            eigfile2 = h5py.File(f'{self.dir_eigfiles}/' + f2)

            print(eigfile1.attrs['n_pg'], eigfile1.attrs['l'])
            print(eigfile2.attrs['n_pg'], eigfile2.attrs['l'])

            # desired mode
            n1, ell1 = self.nl1_arr[kern_idx]
            n2, ell2 = self.nl2_arr[kern_idx]

            r1, rho1 = eigfile1['x'][()], eigfile1['rho'][()]
            r1 = r1/r1[-1]
            U1, V1 = eigfile1['xi_r']['re'][()], eigfile1['xi_h']['re'][()]
            r2, rho2 = eigfile2['x'][()], eigfile2['rho'][()]
            r2 = r2/r2[-1]
            U2, V2 = eigfile2['xi_r']['re'][()], eigfile2['xi_h']['re'][()]

            # interpolating the two eigenfunctions on the same grid
            # using the eigenfunction with larger number of grids as final mesh in r
            if(self.self_coupling or (len(r1) == len(r2))):
                r = r1
                rho = rho1
                pass

            elif(len(r1) > len(r2)):
                r = r1
                rho = rho1
                # interpolate the second set of eigenfunctions
                U2 = np.interp(r1, r2, U2)
                V2 = np.interp(r1, r2, V2)

            else:
                r = r2
                rho = rho2
                # interpolate the first set of eigenfunctions
                U1 = np.interp(r2, r1, U1)
                V1 = np.interp(r2, r1, V1)

            # checking if we need interpolation of B-spline basis
            if(len(r) == len(self.r_bsp_basis_rot)):
                bsp_basis_thismode = self.bsp_basis_rot
            else:
                bsp_basis_thismode = self.interp_splines(r)

            # normalizing Ui and Vi
            eignorm1 = fn.eignorm(U1, V1, ell1, r, rho)
            U1, V1 = U1 / eignorm1, V1 / eignorm1
            eignorm2 = fn.eignorm(U2, V2, ell2, r, rho)
            U2, V2 = U2 / eignorm2, V2 / eignorm2

            Tsr_nl = self.compute_Tsr(r, ell1, ell2, U1, V1, U2, V2)
            rho_Tsr_nl = Tsr_nl * (rho * r**2)[NAX,:]

            # shape (m x s)
            ellmin = np.min([ell1, ell2])
            m = np.arange(-ellmin, ellmin+1)
            wigvals = np.zeros((2*ellmin+1, len(self.s_arr)))
            for i, s in enumerate(self.s_arr):
                wigvals[:, i] = fn.w3j_vecm(ell1, s, ell2, -m, 0*m, m)

            prod_gammas = fn.gam(ell1) * fn.gam(ell2) # * fn.gam(self.s_arr), not using for Omega
            prefactors = wigvals * ((-1)**(np.abs(m)) * 4*np.pi * prod_gammas)[:,NAX]

            # multiplying the various other factors.
            # condensing the r dimension to knots 
            if(self.return_splined_kernel):
                # K shape is now (s x Nparams)
                K = simps(rho_Tsr_nl[:,NAX,:] * bsp_basis_thismode[NAX,:,:], axis=2, x=r)
            else:
                # K shape is now (s x r)
                K = rho_Tsr_nl

            # K shape is now (m x s x Nparams) 
            # also converting to kernel for freq (1/time) from Omega (rad / time)
            K = K[NAX,:,:] *  prefactors[:,:,NAX] * 2 * np.pi

            # saving the kernel in dictionary
            n1_str, ell1_str, n2_str, ell2_str = str(n1), str(ell1), str(n2), str(ell2)
            # this 1D in m form will be stored to save memory as compared to 2D submatrix form
            self.kern_dict[f'n1={n1_str}_ell1={ell1_str}_n2={n2_str}_ell2={ell2_str}'] = {}
            self.kern_dict[f'n1={n1_str}_ell1={ell1_str}_n2={n2_str}_ell2={ell2_str}']['kernel'] = K.squeeze()
            self.kern_dict[f'n1={n1_str}_ell1={ell1_str}_n2={n2_str}_ell2={ell2_str}']['r'] = r

    # {{{ compute_Tsr():                                                                                                                                                                     
    def compute_Tsr(self, r, ell1, ell2, U1, V1, U2, V2):
        """Function to compute the T-kern as in LR92 for coupling                                                                                                                        
        of two multiplets.                                                                                                                                                               
        """
        Tsr = np.zeros((len(self.s_arr), len(r)))

        L1sq = ell1*(ell1+1)
        L2sq = ell2*(ell2+1)

        Om1 = fn.omega(ell1, 0)
        Om2 = fn.omega(ell2, 0)

        for i, s in enumerate(self.s_arr):
            ls2fac = L1sq + L2sq - s*(s+1)
            eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac
            # eigfac = U2*V1 + V2*U1 + V1*V2
            wigval = fn.w3j_vecm(ell1, s, ell2, -1, np.array([0]), 1)
            Tsr[i, :] = -(1 - (-1)**(np.abs(ell1 + ell2 + s))) * \
                        Om1 * Om2 * wigval * eigfac          # / GVAR.r, not using for Omega                                                                                                   

        return Tsr
    # }}} compute_Tsr()   


    def create_bsplines(self, custom_knot_num):
        rmin, rmax = self.r_bsp_basis_rot.min(), self.r_bsp_basis_rot.max()

        # adjusting the total number of knots for a B-spline
        total_knot_num = custom_knot_num
        total_knot_num += 4 - total_knot_num%4 - 1

        # sampling the knots according to the spacing in the radius grid
        # TO CUSTOMIZE THE KNOT LOCATIONS, USE AN ADEQUATELY SPACED RADIUS GRID
        num_skip = len(self.r_bsp_basis_rot)//total_knot_num
        knot_locs_uniq = self.r_bsp_basis_rot[::num_skip][:total_knot_num-1]
        knot_locs_uniq = np.append(knot_locs_uniq, rmax)

        #----------ORIGINAL SPLINE CONFIGURATION FROM HELIOSEISMOLOGY-------------------#
        # vercof1, dvercof1 = eval_polynomial(self.r,
        #                                     [rmin, self.knot_locs[self.knot_ind_shell]],
        #                                     1, types= ['TOP','BOTTOM'])
        # vercof2, dvercof2 = eval_vbspl(self.r, knot_locs_uniq[:self.knot_ind_shell+1])
        # vercof3, dvercof3 = eval_polynomial(self.r,
        #                                     [self.knot_locs[self.knot_ind_shell], rmax],
        #                                     1, types= ['TOP','BOTTOM'])
        # vercof4, dvercof4 = eval_vbspl(self.r, knot_locs_uniq[self.knot_ind_shell:])
        
        # idx = np.where(vercof3[:, -1] > 0)[0][0]
        # vercof3[idx, -1] = 0.0

        # idx = np.where(vercof4[:, 0] > 0)[0][0]
        # vercof4[idx, 0] = 0.0

        # arranging the basis from left to right with st lines                                
        # bsp_basis = np.column_stack((vercof1[:, -1],
        #                              vercof2[:, 1:-1],
        #                              vercof1[:, 0],
        #                              vercof3[:, -1],
        #                              vercof4[:, 1:-1],
        #                              vercof3[:, 0]))

        #-----------SPLINE CONFIGURATION WITH NO BOUNDARY CONSIDERATION IN BETWEEN-------------#
        vercof1, dvercof1 = eval_polynomial(self.r_bsp_basis_rot, [rmin, rmax],
                                            1, types= ['TOP','BOTTOM'])

        vercof2, dvercof2 = eval_vbspl(self.r_bsp_basis_rot, knot_locs_uniq)                                      

        # bsp_basis = np.column_stack((vercof1[:, 1],
        #                              vercof2[:, 1:-1],
        #                              vercof1[:, 0]))

        bsp_basis_rot = vercof2

        self.knot_locs = knot_locs_uniq

        # storing the analytically derived B-splines
        # making them of shape (n_basis, r)
        self.bsp_basis_rot = bsp_basis_rot.T

    def interp_splines(self, r_new):
        bsp_basis_new = np.zeros((len(self.bsp_basis_rot), len(r_new)))
        for i in range(len(self.bsp_basis_rot)):
            bsp_basis_new[i] = np.interp(r_new, self.r_common_grid, self.bsp_basis_rot[i])

        return bsp_basis_new

if __name__ == '__main__':
    # location of the eigenfunction files
    dir_eigfiles = '../Vincent_Eig/mode_files'

    # modes we want to use for rotation inversion
    mode_nl1_arr = np.array([[2, 2],
                             [-1, 2],
                             [-2, 2],
                             [4, 1]])

    # choice to indicate if we want the kernel splined in radius
    return_splined_kernel = True

    # makes the rotation kernels (self-coupling by default)
    rotation_kerns = rot_kern(dir_eigfiles, mode_nl1_arr, custom_knot_num = 6,
                              return_splined_kernel=return_splined_kernel)

    # # plotting the kernels
    # color = np.array(['red', 'orange', 'blue', 'cyan'])
    # m_idx_plot = np.array([3, 3, 3, 2])
    # for i, kern_key in enumerate(rotation_kerns.kern_dict.keys()):
    #     # plotting only m=1 component
    #     plt.plot(rotation_kerns.kern_dict[kern_key]['r'],
    #              rotation_kerns.kern_dict[kern_key]['kernel'][m_idx_plot[i]],
    #              color=color[i])
