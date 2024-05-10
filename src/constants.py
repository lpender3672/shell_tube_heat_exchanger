import numpy as np

cp = 4179 # J/kgK
rho_w = 990.1 # kg/m3
k_w = 0.632 # W / mK
mu = 6.51e-4 # kg/ms
k_tube = 386 # W / mK

Moody_Transition_RE = 10e5

### Fixed geometric requirements that are not optimised for.

max_HE_length = 0.35 # m
D_shell = 0.064 # m
D_inlet_nozzle = 0.02 # m
D_inner_tube = 0.006 # m

c_triangle = 0.2
c_square = 0.15

## Areas of tubes

A_tube = np.pi * D_inner_tube ** 2 / 4
A_nozzle = np.pi * D_inlet_nozzle ** 2 / 4
# wrong!: A_shell = np.pi * D_shell ** 2 / 4

