from constants import *
import numpy as np

import matplotlib.pyplot as plt

def entry_exit_loss_coefficients(Re, sigma):
    ## Need to ask demonstrator about this ???
    #Kc = 0.45
    #Ke = 0.8

    # only valid for sigma less than 0.35
    if sigma > 0.35:
        print("Warning: sigma > 0.35 which invalidates simple relation determined from figure 8")

    Ke = 1 - 1.8 * sigma
    Kc = 0.5 - 0.5 * sigma

    return Kc, Ke
    
def moody_friction_coefficient(Re, roughness):
    # if Re then apply different rules
    return (1.82 * np.log10(Re) - 1.64)**-2

    # THIS COULD ALSO POSSIBLY SOLVE THE ColeBrook-White equation for potentially better results.


def cold_mass_flow_from_dp(cold_dp):

    return np.interp(cold_dp * 1e-5,
                     cold_side_compressor_characteristic[1],
                     cold_side_compressor_characteristic[0]) * rho_w / 1000

def hot_mass_flow_from_dp(hot_dp):
    
    return np.interp(hot_dp * 1e-5, # bar
                     hot_side_compressor_characteristic[1],
                     hot_side_compressor_characteristic[0]) * rho_w / 1000

def logmeanT(T1in, T1out, T2in, T2out):
    dt1 = (T2in - T1out)
    dt2 = (T2out - T1in)

    return (dt1 - dt2) / np.log(dt1 / dt2)

def heat_solve_iteration(T1in, T1out, T2in, T2out):
    pass # ??



class Heat_Exchanger():
    def __init__(self, fluid_path, pattern, N_tubes, B_baffles):

        self.N_tubes = N_tubes
        self.B_baffles = B_baffles

        # initial values
        self.mdot_hot = 0.3
        self.mdot_cold = 0.3

        self.L_hot_tube = 0.35

        self.pattern = pattern
        self.pitch = 0.014 # Y in handout

        self.hydraulic_iteration_count = 0

        self.hot_pressure_factor = 2
        self.cold_pressure_factor = 2


    def hydraulic_iteration(self):

        ## HOT STREAM
        mdot_hot_tube = self.mdot_hot / self.N_tubes

        self.v_hot_tube = mdot_hot_tube / (rho_w * A_tube)
        v_hot_nozzle = mdot_hot_tube / (rho_w * A_nozzle)

        self.Re_hot = self.v_hot_tube * rho_w * D_inner_tube / mu

        self.f_hot = moody_friction_coefficient(self.Re_hot, 0.00015)

        DP_hot_tube_friction = 0.5 * rho_w * self.v_hot_tube**2 * (self.f_hot * self.L_hot_tube / D_inner_tube)
        sigma = self.N_tubes * A_tube / A_shell
        Kc, Ke = entry_exit_loss_coefficients(self.Re_hot, sigma)
        
        self.DP_hot = DP_hot_tube_friction + 0.5 * rho_w * self.v_hot_tube ** 2 * (Kc + Ke) + rho_w * v_hot_nozzle **2
        
        ## COLD STREAM
        
        self.v_shell = self.mdot_cold / (rho_w * self.A_shell_effective)

        effective_d_shell = D_shell * self.A_shell_effective / A_shell
        self.Re_shell = self.v_shell * rho_w * effective_d_shell / mu

        v_cold_nozzle = self.mdot_cold / (rho_w * A_nozzle)

        if self.pattern == Pattern.SQUARE:
            DP_cold_shell = 4 * a_square * self.Re_shell ** (-0.15) * self.N_tubes * rho_w * self.v_shell ** 2
        elif self.pattern == Pattern.TRIANGLE:
            DP_cold_shell = 4 * a_triangle * self.Re_shell ** (-0.15) * self.N_tubes * rho_w * self.v_shell ** 2
        else:
            print("Error: Unknown pattern")

        self.DP_cold = DP_cold_shell + rho_w * v_cold_nozzle**2

        new_mdot_hot = hot_mass_flow_from_dp(self.DP_hot * self.hot_pressure_factor)
        new_mdot_cold = cold_mass_flow_from_dp(self.DP_cold * self.cold_pressure_factor)

        dmhot = new_mdot_hot - self.mdot_hot
        dmcold = new_mdot_cold - self.mdot_cold
        
        if np.abs(dmhot) > 1e-8 or np.abs(dmcold) > 1e-8:

            if self.hydraulic_iteration_count > max_hydraulic_iterations:
                return False
            
            self.mdot_hot = new_mdot_hot
            self.mdot_cold = new_mdot_cold

            self.hydraulic_iteration_count += 1
            return self.hydraulic_iteration()
        
        else:
            return True


    def compute_effectiveness(self, T1in, T2in):

        # Variable geometry parameters
        self.B_spacing = self.L_hot_tube / (self.B_baffles + 1)
        self.A_shell_effective = (self.pitch - D_outer_tube) * self.B_spacing * D_shell / self.pitch

        ## HYDRAULIC ANALYSIS

        plt.plot(cold_side_compressor_characteristic[0], cold_side_compressor_characteristic[1])
        plt.plot(hot_side_compressor_characteristic[0], hot_side_compressor_characteristic[1])
        plt.grid()
        #plt.show()

        
        hydraulic_solution = self.hydraulic_iteration()

        if not hydraulic_solution:
            print("Warning did not converge")
            return
        else:
            print("successfully converged!!!!")

        ## THERMAL ANALYSIS
        
        F = 1 # varies for different flow paths

        ## obtrain heat transfer coefficients

        Nu_i = 0.023 * self.Re_hot ** 0.8 * Pr **0.3

        if self.pattern == Pattern.SQUARE:
            Nu_o = c_square * self.Re_shell **0.6 * Pr **0.3
        elif self.pattern == Pattern.TRIANGLE:
            Nu_o = c_triangle * self.Re_shell **0.6 * Pr **0.3

    
        h_i = Nu_i * k_w / D_inner_tube
        h_o = Nu_o * k_w / D_outer_tube

        A_i = np.pi * D_inner_tube * self.L_hot_tube
        A_o = np.pi * D_outer_tube * self.L_hot_tube
        H = ( 1/h_i + A_i * np.log(D_outer_tube / D_inner_tube) / (2*np.pi * k_tube * self.L_hot_tube) + 1 / h_o * (A_i / A_o)  )**-1


        Qdot = F * H * self.N_tubes * np.pi * self.L_hot_tube

        print(Qdot)

    def calculate_mass(self):
        # calculates the mass of the heat exchanger

        pass

    def is_geometrically_feasible(self):
        # performs collision detection to see if the heat exchanger is geometrically feasible

        # check square or triangle design packing of the N_tubes in a shell for the given pitch
        # also check length of tubes are within individual and total limits

        pass

