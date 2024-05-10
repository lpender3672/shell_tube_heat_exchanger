import numpy as np
from enum import Enum

class Pattern(Enum):
    TRIANGLE = 1
    SQUARE = 2

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
D_outer_tube = 0.008 # m

# confusingly also referred to as c in the handout
a_triangle = 0.02
a_square = 0.014

## Areas of tubes
A_tube = np.pi * D_inner_tube ** 2 / 4
A_nozzle = np.pi * D_inlet_nozzle ** 2 / 4
A_shell = np.pi * D_shell ** 2 / 4
# A shell is variable
