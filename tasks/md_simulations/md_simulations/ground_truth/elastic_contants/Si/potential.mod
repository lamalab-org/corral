# NOTE: This script can be modified for different pair styles
# See in.elastic for more info.

# Choose potential
pair_style      reaxff NULL safezone 100 mincap 10000
pair_coeff      * * ../../potentials/ffield.reax Si

fix             qeq all qeq/reaxff 1 0.0 10.0 1e-6 reaxff


# Setup neighbor style
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# Setup minimization style
#min_style	     cg
#min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		1
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no
