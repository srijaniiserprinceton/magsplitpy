# magsplitpy
Repository to calculate frequency splitting in stars due to general 3D magnetism

## Installation
This repository is intended to be used as a package in your current working environment. If you do not already have a patent environment, I would recommend installing the `environment.yml` file to get the environment with default name `astero` (you may choose to name it something else by changing the field "name" in the `environment.yml` file.

Assuming you are going to setup `astero` as your working environment, you can run the following bash commands to (a) create the environment, (b) activate the environment, and (c) install the package `magsplitpy` in the `astero` environment. This ensures you have the same package dependencies and almost a clone of what the developer has, thereby ensuring a smooth launching of the scripts.
```
# Create the conda environment and install dependencies
conda env create -f environment.yml

# Activate the conda environment astero
conda activate astero

# Install magsplitpy
pip install -e .
```
If you choose to not use `astero` and/or use your own patent environment, checkout the packages in the `environment.yml` file and install them manually in your environment (preferably using `conda` sso that it automatically resolves the environment and installs the neessary versions).

The above steps essentially ensure that you can load `magsplitpy` as a package as follows
```
# import the package
from magplitpy import mag_GSH_funcs
```

## Using `magsplitpy` to get frequency splittings
The script `magsplitpy/compute_splittings.py` is supposed to be the front-end driving code that the user should run to get the frequency splittings. It is recommended that the user makes a copy of this code in their local directory for their project instead of changing this in the `magsplitpy` directory. If the user needs to define a new magnetic field topology than what is already present in `magsplitpy/synthetic_B_profiles.py`, then it is recommended that the user also makes a copy of this file in their local project directory. In that case, change the line
```
from magsplitpy import synthetic_B_profiles as B_profiles
```
to
```
import synthetic_B_profiles as B_profiles
```
This is because the local version of this file needs to be accessed.

### Which parts of `compute_splittings.py` needs to be checked or edited by the user
The user needs to primarily check and edit lines under the line
```
if __name__ == "__main__":
```
There are the following crucial components that the user needs to adapt according to their purpose 

### Providing the stellar model
Providing the eigenfunctions and eigenfrequencies of the background stellar model (usually obtained from GYRE) is the first step. This can be done by providing the directory name where the eigenfunction files are saved as follows
```
# path to the directory containing the eigfiles
dir_eigfiles = <name of directory cotaining only the eigenfunction files>
```
### Specifying the units of frequency splittings
For most purposes yet, returning this in `muHz` or `day_inv` has suffised. This can be specified in the following lines
```
# units in which we want the splittings
# currently setup for 'muHz' or 'day_inv'
freq_units = 'day_inv'
```
### Setting up the Lorentz-stress field object
This is possibly the part that takes the most interference from the user. 
```
# the string name for the category of field we want (see magsplitpy/synthetic_B_profiles.py)
B_field_type = 'Bugnet_2021'

# the scaling factor, by default all fields are in some sense normalized
B0 = 50000

# the number of spline knots to be used to discretize the field in radius
user_knot_num = 50

# the maximum angular degree to which magnetic field B is to be decomposed
sB_max = 1

# tilt of the magnetic axis wrt the rotation axis in degrees
mag_obliquity = 75.0
```
The above lines completely define the magnetic field (its topology, how strong it is, how many splines in radius to use for it, highest angular degree and its inclination with respect to the rotation axis. 
### A sample `magsplitpy` run 
This is to test if the installation was successful and the user has `magsplitpy` optimally setup. 
* Download the directory `mode_files` and the file `Field_BC.json` from the Google Drive link: https://drive.google.com/drive/folders/1fviDK9tbWnjr0IjlIW-Pqz821bAfmNKP?usp=share_link
* Create a directory `Vincent_Eig` (just because this is how it is on my system now... should be changed later) in the repository.
* Place `mode_files` inside `Vincent_Eig` and `Field_BC.json` inside `tests` directory.
* `cd magsplitpy`
* `ipython`
* `run compute_splittings.py`

You should see the following plot show up after the code has successfully finished running.

![Alt text](Figures/expected_splitting_plot.png)


