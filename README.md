# magsplitpy
Repository to calculate frequency splitting in stars due to general 3D magnetism

## Using `magsplitpy` to get Generalized Spherical Harmonic (GSH) cofficients from a 3D magnetic field

```
# import the package
from magplitpy import mag_GSH_funcs

# pass the 3D magnetic field with all the three vector components as B
B_GSH_coefs = mag_GSH_funcs.get_B_GSHcoeffs_from_B(B)
```
