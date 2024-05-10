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
    def __init__(self, N_tubes, B_baffles):

        self.N = N_tubes
        self.B = B_baffles

        self.mdot_hot = 0.45
        self.mdot_cold = 0.5

        
        print(self.DP_hot_tube)

        



    def compute_effectiveness():

        pass
        ## HYDRAULIC ANALYSIS

        ## HOT STREAM
        self.mdot_hot_tube = self.mdot_hot / self.N
        self.L_hot_tube = 0.35

        self.v_hot_tube = self.mdot_hot_tube / (rho_w * A_tube)
        self.v_hot_nozzle = self.mdot_hot_tube / (rho_w * A_nozzle)

        self.Re_hot = self.v_hot_tube * rho_w * D_inner_tube / mu

        self.f_hot = moody_friction_coefficient(self.Re_hot, )

        DP_hot_tube_friction = 0.5 * rho_w * self.v_hot_tube**2 * (self.f_hot * self.L_hot_tube / D_inner_tube)
        sigma = self.N * A_tube / A_shell
        Kc, Ke = entry_exit_loss_coefficients(self.Re_hot, self.sigma)
        
        self.DP_hot_tube = DP_hot_tube_friction + 0.5 * rho_w * self.v_hot_tube ** 2 * (Kc + Ke) + rho_w * self.v_hot_nozzle **2
        
        ## COLD STREAM
        
        self.V_sh = self.mdot_cold / (rho_w * A_shell)


        ## THERMAL ANALYSIS


