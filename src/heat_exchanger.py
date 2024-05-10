from constants import *

def entry_exit_loss_coefficients(Re, sigma):
    ## Need to ask demonstrator about this ???

    Kc = 0.45
    Ke = 0.8

    return Kc, Ke
    
def moody_friction_coefficient(Re, roughness):
    # if Re then apply different rules
    return (1.82 * np.log10(Re) - 1.64)**-2

class Heat_Exchanger():
    def __init__(self, pattern, N_tubes, B_baffles):

        self.N_tubes = N_tubes
        self.B_baffles = B_baffles

        # initial values
        self.mdot_hot = 0.45
        self.mdot_cold = 0.5

        self.L_hot_tube = 0.35

        self.pattern = pattern
        self.pitch = 0.014 # Y in handout


    def compute_effectiveness(self):

        # Variable geometry parameters
        B_spacing = self.L_hot_tube / (self.B_baffles + 1)
        A_shell_effective = (self.pitch - D_outer_tube) * B_spacing * D_shell / self.pitch

        print(B_spacing)
        print(A_shell)

        ## HYDRAULIC ANALYSIS

        ## HOT STREAM
        self.mdot_hot_tube = self.mdot_hot / self.N_tubes

        self.v_hot_tube = self.mdot_hot_tube / (rho_w * A_tube)
        self.v_hot_nozzle = self.mdot_hot_tube / (rho_w * A_nozzle)

        self.Re_hot = self.v_hot_tube * rho_w * D_inner_tube / mu

        self.f_hot = moody_friction_coefficient(self.Re_hot, 0.00015)

        DP_hot_tube_friction = 0.5 * rho_w * self.v_hot_tube**2 * (self.f_hot * self.L_hot_tube / D_inner_tube)
        sigma = self.N_tubes * A_tube / A_shell
        Kc, Ke = entry_exit_loss_coefficients(self.Re_hot, sigma)
        
        self.DP_hot_tube = DP_hot_tube_friction + 0.5 * rho_w * self.v_hot_tube ** 2 * (Kc + Ke) + rho_w * self.v_hot_nozzle **2
        
        ## COLD STREAM
        
        self.V_shell = self.mdot_cold / (rho_w * A_shell_effective)

        effective_d_shell = D_shell * A_shell_effective / A_shell
        self.Re_shell = self.V_shell * rho_w * effective_d_shell / mu

        print(self.Re_shell)



        ## THERMAL ANALYSIS


