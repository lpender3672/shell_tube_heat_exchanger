
import numpy as np
import matplotlib.pyplot as plt

import copy

from scipy.optimize import fsolve
import scipy
import scipy.optimize
import scipy.interpolate

import logging

from constants import *

from fluid_path import Fluid_Path, Entry_Constriction, Exit_Expansion, U_Bend, Heat_Transfer_Element
from heat_exchanger import Heat_Exchanger, e_NTU, cold_mass_flow_from_dp, hot_mass_flow_from_dp, logmeanT


class Heat_Exchanger_Perturbed(Heat_Exchanger):
    def __init__(self, cold_fluid_path, hot_fluid_path, flow_path_entries_side):
        super().__init__(cold_fluid_path, hot_fluid_path, flow_path_entries_side)

    
    def calc_dp(self, mdot, avg_pitch, baffle_spacing):

        mdot_cold, mdot_hot = mdot

        # HOT STREAM

        DP_hot = 0

        for i, element in enumerate(self.hot_path.elements):

            if isinstance(element, Heat_Transfer_Element):

                mdot_hot_tube = mdot_hot / element.tubes
                v_hot_tube = mdot_hot_tube / (rho_w * A_tube)

                Re_hot = v_hot_tube * rho_w * D_inner_tube / mu
                f_hot = element.friction_coefficient(Re_hot, 0.00015)

                DP_hot += 0.5 * rho_w * v_hot_tube**2 * \
                    (f_hot * self.L_hot_tube / D_inner_tube)

            if isinstance(element, Entry_Constriction):
                try:
                    next_element = self.hot_path.elements[i+1]
                    assert isinstance(next_element, Heat_Transfer_Element)
                except (IndexError, AssertionError):
                    raise ValueError("Entry constriction must be followed by a heat transfer element")

                mdot_hot_tube = mdot_hot / next_element.tubes
                v_hot_tube = mdot_hot_tube / (rho_w * A_tube)
                Re_hot = v_hot_tube * rho_w * D_inner_tube / mu

                # if constriction is from inlet then A_constriction = A_section
                # otherwise constriction is from a bend so A_constriction = 2 * A_section
                if i == 0:
                    A_constriction = A_shell / self.hot_flow_sections
                else:
                    A_constriction = 2 * A_shell / self.hot_flow_sections
                 
                sigma = next_element.tubes * A_tube / A_constriction


                DP_hot += 0.5 * rho_w * v_hot_tube ** 2 * \
                    element.loss_coefficient(Re_hot, sigma)

            if isinstance(element, Exit_Expansion):
                try:
                    prev_element = self.hot_path.elements[i-1]
                    assert isinstance(prev_element, Heat_Transfer_Element)
                except (IndexError, AssertionError):
                    raise ValueError("Exit expansion must be preceded by a heat transfer element")

                mdot_hot_tube = mdot_hot / prev_element.tubes
                v_hot_tube = mdot_hot_tube / (rho_w * A_tube)

                Re_hot = v_hot_tube * rho_w * D_inner_tube / mu

                # if expanding to last section, then A_expansion = A_section
                # if not, then expanding to a bend so A_expansion = 2 * A_section
                if i == len(self.hot_path.elements) - 1:
                    A_expansion = A_shell / self.hot_flow_sections
                else:
                    A_expansion = 2 * A_shell / self.hot_flow_sections

                sigma = prev_element.tubes * A_tube / A_expansion

                DP_hot += 0.5 * rho_w * v_hot_tube ** 2 * \
                    element.loss_coefficient(Re_hot, sigma)
                
            if isinstance(element, U_Bend):
                try:
                    prev_element = self.hot_path.elements[i-2]
                    assert isinstance(prev_element, Heat_Transfer_Element)
                except (IndexError, AssertionError):
                    raise ValueError("U Bend must be preceded by a Heat_Transfer_Element and Exit_Expansion")
                
                mdot_hot_tube = mdot_hot / prev_element.tubes
                v_hot_tube = mdot_hot_tube / (rho_w * A_tube)
                
                DP_hot += element.loss_coefficient() * 0.5 * rho_w * v_hot_tube ** 2
            
        v_hot_nozzle = mdot_hot / (rho_w * A_nozzle)
        DP_hot += rho_w * v_hot_nozzle ** 2  # nozzle loss

        # COLD STREAM
        DP_cold = 0

        for i, element in enumerate(self.cold_path.elements):

            if isinstance(element, Heat_Transfer_Element):
                hot_sections = self.hot_flow_sections//self.cold_flow_sections
                for j in range(hot_sections):
                    
                    if element.pattern[j] == Pattern.SQUARE:
                        effective_d_shell = 1.27/D_outer_tube * (avg_pitch**2 - 0.785 * D_outer_tube**2)
                    elif element.pattern[j] == Pattern.TRIANGLE:
                        effective_d_shell = 1.10/D_outer_tube * (avg_pitch**2 - 0.917 * D_outer_tube**2)
                    else:
                        logging.error("Error: Unknown pattern")
                    
                    effective_d_shell  = (D_shell**2 - self.total_tubes*D_outer_tube**2)/(self.total_tubes*D_outer_tube)
                    
                    A_shell_effective = (avg_pitch - D_outer_tube) * \
                        baffle_spacing * D_shell / avg_pitch
                    
                    v_shell = mdot_cold / (rho_w * A_shell_effective)

                    Re_shell = v_shell * rho_w * effective_d_shell / mu

                    j_f = 0.202 * Re_shell**(-0.153)

                    DP_cold += 4 * j_f * (D_shell / effective_d_shell) * (element.baffles + 1)/self.hot_flow_sections \
                        * rho_w * v_shell ** 2
                
            if isinstance(element, U_Bend):
                for i in range(hot_sections):

                    try:
                        prev_element = self.cold_path.elements[i-1]
                        assert isinstance(prev_element, Heat_Transfer_Element)
                    except (IndexError, AssertionError):
                        raise ValueError("U Bend must be preceded by a heat transfer element")
                    
                    
                    B_spacing = self.L_hot_tube / (prev_element.baffles + 1)
                    A_shell_effective = (avg_pitch - D_outer_tube) * B_spacing * D_shell / avg_pitch

                    A_section = A_shell_effective / self.cold_flow_sections

                    v_shell = mdot_cold / (rho_w * A_section)
                    
                    DP_cold += element.loss_coefficient() * 0.5 * rho_w * v_shell ** 2/self.hot_flow_sections
            

        v_cold_nozzle = mdot_cold / (rho_w * A_nozzle)

        DP_cold += rho_w * v_cold_nozzle**2
        
        # Fudge factors based on experimental data in analysis.ipynb
        DP_cold = 1.35525014e+00 * DP_cold + 1.08200831e+04
        DP_hot = 5.20759596e-01 * DP_hot + 5.49446226e+03

        #DP_cold = 1.984012962563786 * DP_cold +7217.109422930143
        #DP_cold = 1.6953864289383282 * DP_cold-10686.652389444924

        return DP_cold, DP_hot
    
    def hydraulic_iteration(self, mdot, avg_pitch, baffle_spacing):
        mdot_cold, mdot_hot = mdot

        dp_cold, dp_hot = self.calc_dp(mdot, avg_pitch, baffle_spacing)

        new_mdot_cold = cold_mass_flow_from_dp(dp_cold, self.characteristic_year)
        new_mdot_hot = hot_mass_flow_from_dp(dp_hot, self.characteristic_year)
        error = [new_mdot_cold - mdot_cold, new_mdot_hot - mdot_hot]

        return error
    

    def calc_area_times_H(self, mdot, avg_pitch):
        mdot_cold, mdot_hot = mdot
        areatimesH = 0

        for i, element in enumerate(self.hot_path.elements):
            if not isinstance(element, Heat_Transfer_Element):
                continue

            mdot_hot_tube = mdot_hot / element.tubes
            v_hot_tube = mdot_hot_tube / (rho_w * A_tube)
            Re_hot = v_hot_tube * rho_w * D_inner_tube / mu

            # obtain the heat transfer coefficient for the inner and outer tubes            
            if element.pattern == Pattern.SQUARE:
                effective_d_shell = 1.27/D_outer_tube * (avg_pitch**2 - 0.785 * D_outer_tube**2) 
                c = c_square
            elif element.pattern == Pattern.TRIANGLE:
                effective_d_shell = 1.10/D_outer_tube * (avg_pitch**2 - 0.917 * D_outer_tube**2) 
                c = c_triangle
            else:
                logging.error("Error: Unknown pattern")
            

            B_spacing = self.L_hot_tube / (element.baffles + 1)
            A_shell_effective = (avg_pitch - D_outer_tube) * \
                B_spacing * D_shell / (avg_pitch)
            effective_d_shell  =  (D_shell**2 - self.total_tubes*D_outer_tube**2)/(self.total_tubes*D_outer_tube)
            
            v_shell = mdot_cold / (rho_w * A_shell_effective)

            Re_shell = v_shell * rho_w * effective_d_shell / mu

            j_h = 0.4275*Re_shell**(-0.466)

            Nu_i = 0.023 * Re_hot ** 0.8 * Pr ** 0.33
            Nu_o = j_h * Re_shell * Pr ** 0.33
            #Nu_o = c * Re_shell ** 0.6 * Pr ** 0.33

            h_i = Nu_i * k_w / D_inner_tube
            h_o = Nu_o * k_w / D_outer_tube
            #h_o = Nu_o * k_w / effective_d_shell

            A_i = np.pi * D_inner_tube * self.L_hot_tube
            A_o = np.pi * D_outer_tube * self.L_hot_tube
            one_over_H = 1/h_i + A_i * np.log(D_outer_tube / D_inner_tube) / (
                2 * np.pi * k_tube * self.L_hot_tube) + (A_i / A_o) / h_o
            
            # additional fouling resistance
            one_over_H += self.Rfouling[0] + D_outer_tube / D_inner_tube * self.Rfouling[1]
            
            areatimesH += element.tubes * np.pi * D_inner_tube * self.L_hot_tube / one_over_H

        return areatimesH
    

    def compute_effectiveness(self, avg_pitch, baffle_spacing, method="LMTD", optimiser='fsolve'):
        
        try:
            self.mdot, _, res, *_ = fsolve(self.hydraulic_iteration,
                                self.mdot,
                                args = (avg_pitch, baffle_spacing),
                                xtol = hydraulic_error_tolerance,
                                full_output = True)

            assert res == 1
        except AssertionError:
            logging.error("Hydraulic analysis failed to converge")
            return False
        except Exception as e:
            logging.critical(f"Hydraulic analysis failed: {e}")
            return False
        
        mdot_cold, mdot_hot = self.mdot
        T1in, T2in = self.Tin

        # Thermal analysis

        try:
            self.area_times_H = self.calc_area_times_H(self.mdot, avg_pitch)
        except Exception as e:
            logging.error(f"Failed to calculate area times H: {e}")
            return False
        
        N_shell = self.cold_flow_sections
        N_tube = self.hot_flow_sections
        C_1 = cp*mdot_cold
        C_2 = cp*mdot_hot
        
        NTU = e_NTU(self.area_times_H, C_1, C_2, N_shell, N_tube, self.flow_path_entries_side)

        effectiveness = e_NTU.effectiveness(NTU)

        Qdot_max = (NTU.C_min * (T2in - T1in))
        Qdot = effectiveness*Qdot_max

        self.Qdot = Qdot
        self.ntu = NTU.ntu

        T1out = T1in + Qdot / C_1
        T2out = T2in - Qdot / C_2
        self.Tout = [T1out, T2out]
        self.LMTD = logmeanT(T1in, T1out, T2in, T2out)

        self.effectiveness = effectiveness
        Qdot_corrected = 0.5083278756212372 * Qdot + 6425.748610071553

        self.Qdot = Qdot_corrected
        
        return True