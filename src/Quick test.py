
import scipy.stats
from constants import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy


df = pd.read_csv('src\GA3_previous_designs.csv')

# remove weird group C designs
df = df[~((df["year"] == 2022) & (df["group"] == "Group-C"))] # remove group C 2022
df = df[~((df["year"] == 2023) & (df["group"] == "Group-C"))] # remove group C 2023

hot_passes = np.ones(len(df))
cold_passes = np.ones(len(df))

tubes = np.ones(len(df))
baffles = np.ones(len(df))
length = np.ones(len(df))
cold_flow = np.ones(len(df))
hot_flow = np.ones(len(df))
dp_cold = np.ones(len(df))
dp_hot = np.ones(len(df))



for i, row in enumerate(df.iterrows()):
    
    tube_vals = str(row[1]["Tubes"]).split(",")
    tubes[i] = sum([int(t) for t in tube_vals])
    baffle_vals = str(row[1]["Baffles"]).split(",")
    baffles[i] = sum([int(b) for b in baffle_vals])
    length[i] = float(row[1]["Tube Length"])
    cold_flow[i] = (float(row[1]["Flowrate1"]))
    hot_flow[i] = (float(row[1]["Flowrate2"]))
    dp_cold[i] = (float(row[1]["DP_HX1"]))*10**5
    dp_hot[i] = (float(row[1]["DP_HX2"]))*10**5

    hot_passes[i] = len(tube_vals)
    cold_passes[i] = len(baffle_vals)


A_shell_effective = length/(baffles+1) * D_shell/(5*cold_passes) * 1.5
d_shell_effective = D_shell*(D_shell/(tubes*D_outer_tube)) - D_outer_tube

v_shell = cold_flow/(rho_w*A_shell_effective)

Re_shell = rho_w*v_shell*d_shell_effective/mu

v_nozzle = cold_flow/(rho_w*A_nozzle)

dp_cold_rel = 2*dp_cold/(rho_w*(v_shell)**2)

def K_s(N):
    K_s = np.zeros_like(N)
    for i in range(len(N)):
        if N[i] > 1:
            K_s[i] = 1*(N[i] - 1)
        else:
            K_s[i] = 0
    return K_s



unknown_function = (dp_cold_rel - 2*(v_nozzle/(v_shell))**2 - K_s(cold_passes) ) * d_shell_effective/(D_shell*(baffles/cold_passes +1))

#print(unknown_function)
x_vals = -np.logspace(0.1,1,10000)
regression = []
r_sq = np.ones_like(x_vals)

for i in range(len(x_vals)):
    regression.append(scipy.stats.linregress((Re_shell)*x_vals[i], np.log10(unknown_function)))
    r_sq[i] = regression[i][2]

x_max = np.argmax(r_sq)
r_max = np.max(r_sq)

best_regression = regression[x_max]


fit = np.poly1d(best_regression[0:2])

print(x_vals[x_max], r_max)
print(best_regression)

plt.plot(Re_shell*x_vals[x_max], fit(Re_shell*x_vals[x_max]))

plt.scatter(Re_shell*x_vals[x_max], np.log10(unknown_function))

plt.show()
