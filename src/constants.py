import numpy as np
from enum import Enum

class Pattern(Enum):
    TRIANGLE = 1
    SQUARE = 2

class Direction(Enum):
    COUNTERFLOW = 1
    COFLOW = 2

class Side(Enum):
    SAME = 1
    OPPOSITE = 2

cp = 4179 # J/kgK
rho_w = 990.1 # kg/m3
k_w = 0.632 # W / mK
mu = 6.51e-4 # kg/ms
k_tube = 386 # W / mK
Pr = 4.31

Moody_Transition_RE = 10e5
max_hydraulic_iterations = 900
max_thermal_iterations = 900
hydraulic_error_tolerance = 1e-8

cold_side_compressor_characteristic = np.array([
    [
    0.7083,   
    0.6417,   
    0.5750,   
    0.5083,   
    0.4250,   
    0.3583,   
    0.3083,   
    0.2417,   
    0.1917,   
    0.1583],
    [
    0.1310,
    0.2017,
    0.2750,
    0.3417,
    0.4038,
    0.4503,
    0.4856,
    0.5352,
    0.5717,
    0.5876
    ]
])

hot_side_compressor_characteristic = np.array([
    [
    0.4722,   
    0.4340,   
    0.3924,   
    0.3507,   
    0.3021,   
    0.2535,   
    0.1979,   
    0.1493,   
    0.1111,   
    0.0694],
    [
    0.0538,
    0.1192,
    0.1727,
    0.2270,
    0.2814,
    0.3366,
    0.3907,
    0.4456,
    0.4791,
    0.5115 
    ] 
])

### Fixed geometric requirements that are not optimised for.

max_HE_length = 0.35 # m
D_shell = 0.064 # m
D_inlet_nozzle = 0.02 # m
D_inner_tube = 0.006 # m
D_outer_tube = 0.008 # m

a_triangle = 0.2
a_square = 0.34

c_triangle = 0.2
c_square = 0.15

## Areas of tubes
A_tube = np.pi * D_inner_tube ** 2 / 4
A_nozzle = np.pi * D_inlet_nozzle ** 2 / 4
A_shell = np.pi * D_shell ** 2 / 4
# A shell is variable

## Included mass of parts
rho_copper_tube = 0.2 # kg / m
rho_acrylic_tube = 0.65 # kg / m
m_nozzle = 0.65
rho_abs = 2.39 # kg / m2
rho_resin = 1150 # kg / m3
m_small_O = 0.0008
m_large_O = 0.0053
