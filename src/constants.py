import numpy as np
from enum import Enum

from matplotlib import pyplot as plt

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
max_hydraulic_iterations = 100
max_thermal_iterations = 100
hydraulic_error_tolerance = 1e-8
thermal_error_tolerance = 1e-8


cold_side_compressor_characteristic_2022 = np.array([[
    0.5833,
    0.5083,
    0.4750,
    0.4250,
    0.3792,
    0.3417,
    0.2958,
    0.2583,
    0.2125,
    0.1708
    ],[
    0.1113,
    0.2157,
    0.2538,
    0.3168,
    0.3613,
    0.4031,
    0.4511,
    0.4846,
    0.5181,
    0.5573
]])

hot_side_compressor_characteristic_2022 = np.array([[
    0.4583,
    0.4236,
    0.4010,
    0.3611,
    0.3125,
    0.2639,
    0.2222,
    0.1597,
    0.1181,
    0.0694
],[
    0.1333,
    0.1756,
    0.2024,
    0.2577,
    0.3171,
    0.3633,
    0.4233,
    0.4784,
    0.5330,
    0.5715
]])

cold_side_compressor_characteristic_2023 = np.array([
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

hot_side_compressor_characteristic_2023 = np.array([
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


cold_side_compressor_characteristic_2024 = np.array(
    [
        [
0.6333,
0.6083,
0.5750,
0.5083,
0.4250,
0.3583,
0.3083,
0.2417,
0.1917,
0.1583
        ],
        [
0.1024,
0.1444,
0.1870,
0.2717,
0.3568,
0.4203,
0.4626,
0.5152,
0.5597,
0.5776
        ]
    ]
)

hot_side_compressor_characteristic_2024 = np.array([[
0.4826,
0.4340,   
0.3924,   
0.3507,   
0.3021,   
0.2535,   
0.1979,   
0.1493,   
0.1111,   
0.0694
], [
0.0944,
0.1662,
0.2297,
0.2820,
0.3294,
0.3856,
0.4447,
0.5006,
0.5311,
0.5615
]]
)

### Fixed geometric requirements that are not optimised for.

max_HE_length = 0.35 # m
max_total_tube_length = 3.5
max_HE_mass = 1.2
D_shell = 0.064 # m
D_inlet_nozzle = 0.02 # m
D_inner_tube = 0.006 # m
D_outer_tube = 0.008 # m

end_cap_width_nozzle = 0.040
end_cap_width = 0.020
baffle_width = 0.0015

a_triangle = 0.2
a_square = 0.34

c_triangle = 0.2
c_square = 0.15

pitch_offset = 0.001 # 1 mm is the minimum allowable distance between tubes

## Areas of tubes
A_tube = np.pi * D_inner_tube ** 2 / 4
A_nozzle = np.pi * D_inlet_nozzle ** 2 / 4
A_shell = np.pi * D_shell ** 2 / 4
# A shell is variable

## Included mass of parts
rho_copper_tube = 0.2 # kg / m
rho_acrylic_tube = 0.65 # kg / m
m_nozzle = 0.025 # kg
rho_abs = 2.39 # kg / m2
rho_resin = 1150 # kg / m3
m_small_O = 0.0008
m_large_O = 0.0053


num_threads = 1

if __name__ == "__main__":

    plt.plot(cold_side_compressor_characteristic_2024[0], cold_side_compressor_characteristic_2024[1], label = "2024 Cold side")
    plt.plot(hot_side_compressor_characteristic_2024[0], hot_side_compressor_characteristic_2024[1], label = "2024 Hot side")

    plt.plot(cold_side_compressor_characteristic_2023[0], cold_side_compressor_characteristic_2023[1], label = "2023 Cold side")
    plt.plot(hot_side_compressor_characteristic_2023[0], hot_side_compressor_characteristic_2023[1], label = "2023 Hot side")

    plt.plot(cold_side_compressor_characteristic_2022[0], cold_side_compressor_characteristic_2022[1], label = "2022 Cold side")
    plt.plot(hot_side_compressor_characteristic_2022[0], hot_side_compressor_characteristic_2022[1], label = "2022 Hot side")

    plt.xlabel("Mass flow rate (kg/s)")
    plt.ylabel("Pressure drop (Bar)")
    
    plt.legend()
    plt.grid()
    plt.show()

