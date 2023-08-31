import numpy as np
from inversion import rotation_kernel
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rc

font = {'size'   : 16}
rc('font', **font)

# location of the eigenfunction files
dir_eigfiles = '../Vincent_Eig/mode_files'

# modes we want to use for rotation inversion
mode_nl1_arr = np.array([[2, 2],
                            [-1, 2],
                            [-2, 2],
                            [4, 1]])

# mode_nl1_arr = np.array([[2, 2],
#                          [4, 1]])

# mode_nl1_arr = np.array([[-1, 2],
#                          [-2, 2]])

# choice to indicate if we want the kernel splined in radius
return_splined_kernel = True

# running loop over the number of knots
vline_locs = np.array([0.126, 0.66, 0.84, 0.96])
for fidx, num_knots in enumerate(np.arange(0,1,4)):

    # makes the rotation kernels (self-coupling by default)
    rotation_kerns = rotation_kernel.rot_kern(dir_eigfiles, mode_nl1_arr, custom_knot_num = 6+num_knots,
                                return_splined_kernel=return_splined_kernel)

    key_str = []
    for key in rotation_kerns.kern_dict.keys():
        key_str.append(key)

    key_str = np.asarray(key_str)

    K_total = rotation_kerns.kern_dict[key_str[0]]['kernel'].T

    for i in range(1, len(key_str)):
        K_total = np.column_stack((K_total, rotation_kerns.kern_dict[key_str[i]]['kernel'].T))


    # plotting the SVD for all the knots used
    # plt.figure(figsize=(10,5))
    __, S, __ = np.linalg.svd(K_total)
    # plt.plot(S, '.k')

    # number of dominant singular values (tune according to the above result)
    Ndom_svd = 4

    # plotting the percent difference in the dominant singular values on removal of one  knot at a time
    percent_diff = []
    for i in range(K_total.shape[0]-1):
        # K_new = np.delete(K_total, i, 0)
        K_new = K_total[i:i+1]
        __, S_new, __ = np.linalg.svd(K_new)

        # percent_diff.append(np.sum(np.abs((S[:Ndom_svd] - S_new[:Ndom_svd]) / S[:Ndom_svd])))
        percent_diff.append(np.sum(S_new[:Ndom_svd])/np.sum(S[:Ndom_svd]))

    # plotting to identify the least sensitive knots (where the percent diff is least)
    plt.figure(figsize=(10,5))
    plt.plot(rotation_kerns.knot_locs[:-1], percent_diff)
    for lines in vline_locs: plt.axvline(lines, linestyle='--', color='red')
    plt.title(f'Number of knots = {rotation_kerns.num_knots_rot}')
    plt.xlabel('Radius')
    plt.ylabel('Sensitivity of knot location to inversion')
    plt.xlim([0, 1])
    plt.ylim([0, 0.45])
    plt.tight_layout()
    plt.savefig(f'./knot_sensitivity_plots/{45-fidx}.png')
    plt.close()


