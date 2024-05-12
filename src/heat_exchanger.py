
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

from constants import *

from fluid_path import Entry_Constriction, Exit_Expansion, U_Bend, Heat_Transfer_Element


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




class Heat_Exchanger():
    def __init__(self, cold_fluid_path, hot_fluid_path, flow_path_entries_side):

        self.cold_path = cold_fluid_path
        self.hot_path = hot_fluid_path

        self.flow_path_entries_side = flow_path_entries_side

        cold_side_bends = 0
        hot_side_bends = 0
        self.total_tubes = 0

        for element in self.hot_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                self.total_tubes += element.tubes
            if isinstance(element, U_Bend):
                hot_side_bends += 1
        for element in self.cold_path.elements:
            if isinstance(element, U_Bend):
                cold_side_bends += 1
        
        if cold_side_bends % 2 == hot_side_bends % 2:
            self.flow_path_exits_side = flow_path_entries_side
        elif flow_path_entries_side == Side.SAME:
            self.flow_path_exits_side = Side.OPPOSITE
        else:
            self.flow_path_exits_side = Side.SAME
        
        # TODO: ask if this is something that changes calculations
        self.cold_flow_sections = cold_side_bends + 1
        self.hot_flow_sections = hot_side_bends + 1

        # initial values
        self.mdot_hot = 0.3
        self.mdot_cold = 0.3

        self.L_hot_tube = 0.35

        # TODO: vary this with the heat transfer element pattern
        self.pitch = 0.014 # Y in handout


        self.hydraulic_iteration_count = 0

        # TODO: change these to be actually correct
        self.hot_pressure_factor = 1
        self.cold_pressure_factor = 1


    def hydraulic_iteration(self):

        ## HOT STREAM

        self.DP_hot = 0

        for i, element in enumerate(self.hot_path.elements):
            
            if isinstance(element, Heat_Transfer_Element):

                mdot_hot_tube = self.mdot_hot / element.tubes
                v_hot_tube = mdot_hot_tube / (rho_w * A_tube)

                Re_hot = v_hot_tube * rho_w * D_inner_tube / mu
                f_hot = element.friction_coefficient(Re_hot, 0.00015)

                self.DP_hot += 0.5 * rho_w * v_hot_tube**2 * (f_hot * self.L_hot_tube / D_inner_tube)

            if isinstance(element, Entry_Constriction):
                try:
                    next_element = self.hot_path.elements[i+1]
                    assert isinstance(next_element, Heat_Transfer_Element)
                except (IndexError, AssertionError):
                    return False
                
                mdot_hot_tube = self.mdot_hot / next_element.tubes
                v_hot_tube = mdot_hot_tube / (rho_w * A_tube)
                Re_hot = v_hot_tube * rho_w * D_inner_tube / mu

                sigma = next_element.tubes * A_tube / A_shell
                self.DP_hot += 0.5 * rho_w * v_hot_tube ** 2 * element.loss_coefficient(Re_hot, sigma)
            
            if isinstance(element, Exit_Expansion):
                try:
                    prev_element = self.hot_path.elements[i-1]
                    assert isinstance(prev_element, Heat_Transfer_Element)
                except (IndexError, AssertionError):
                    return False
                
                mdot_hot_tube = self.mdot_hot / prev_element.tubes
                v_hot_tube = mdot_hot_tube / (rho_w * A_tube)
                Re_hot = v_hot_tube * rho_w * D_inner_tube / mu

                sigma = prev_element.tubes * A_tube / A_shell
                self.DP_hot += 0.5 * rho_w * v_hot_tube ** 2 * element.loss_coefficient(Re_hot, sigma)

            #TODO: bend elements probably another loss coefficient

        v_hot_nozzle = mdot_hot_tube / (rho_w * A_nozzle)
        self.DP_hot += rho_w * v_hot_nozzle **2 # nozzle loss
        
        ## COLD STREAM

        self.DP_cold = 0
        
        for i, element in enumerate(self.hot_path.elements):

            if isinstance(element, Heat_Transfer_Element):

                if element.pattern == Pattern.SQUARE:
                    a_factor = a_square
                elif element.pattern == Pattern.TRIANGLE:
                    a_factor = a_triangle
                else:
                    print("Error: Unknown pattern")

                B_spacing = self.L_hot_tube / (element.baffles + 1)
                A_shell_effective = (self.pitch - D_outer_tube) * B_spacing * D_shell / self.pitch

                v_shell = self.mdot_cold / (rho_w * A_shell_effective)

                effective_d_shell = D_shell * A_shell_effective / A_shell
                Re_shell = v_shell * rho_w * effective_d_shell / mu

                self.DP_cold += 4 * a_factor * Re_shell ** (-0.15) * element.tubes * rho_w * v_shell ** 2
            
            # TODO: add bend elements

                
        v_cold_nozzle = self.mdot_cold / (rho_w * A_nozzle)
        
        self.DP_cold += rho_w * v_cold_nozzle**2

        new_mdot_hot = hot_mass_flow_from_dp(self.DP_hot * self.hot_pressure_factor) # TODO: revisit this as this makes no sense
        new_mdot_cold = cold_mass_flow_from_dp(self.DP_cold * self.cold_pressure_factor)

        dmhot = new_mdot_hot - self.mdot_hot
        dmcold = new_mdot_cold - self.mdot_cold
        
        if (np.abs(dmhot) > hydraulic_error_tolerance or 
            np.abs(dmcold) > hydraulic_error_tolerance):

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

        ## HYDRAULIC ANALYSIS
        
        hydraulic_solution = self.hydraulic_iteration()

        if not hydraulic_solution:
            print("Warning did not converge")
            return
        else:
            print("successfully converged!!!!")
            print("Cold mass flow rate: ", self.mdot_cold)
            print("Hot mass flow rate: ", self.mdot_hot)

        ## THERMAL ANALYSIS
        
        Fscale = 1 # TODO: find out what this is


        # TODO: check that 1/H can be added up in a loop like this
        one_over_H = 0

        for i, element in enumerate(self.hot_path.elements):
            if not isinstance(element, Heat_Transfer_Element):
                continue

            # TODO: do something with element.direction

            mdot_hot_tube = self.mdot_hot / element.tubes
            v_hot_tube = mdot_hot_tube / (rho_w * A_tube)
            Re_hot = v_hot_tube * rho_w * D_inner_tube / mu
             
            # obtain the heat transfer coefficient for the inner and outer tubes            

            if element.pattern == Pattern.SQUARE:
                c_factor = c_square
            elif element.pattern == Pattern.TRIANGLE:
                c_factor = c_triangle
            else:
                print("Error: Unknown pattern")

            B_spacing = self.L_hot_tube / (element.baffles + 1)
            A_shell_effective = (self.pitch - D_outer_tube) * B_spacing * D_shell / self.pitch

            v_shell = self.mdot_cold / (rho_w * A_shell_effective)
            effective_d_shell = D_shell * A_shell_effective / A_shell
            Re_shell = v_shell * rho_w * effective_d_shell / mu

            Nu_i = 0.023 * Re_hot ** 0.8 * Pr **0.3
            Nu_o = c_factor * Re_shell **0.6 * Pr **0.3

            h_i = Nu_i * k_w / D_inner_tube
            h_o = Nu_o * k_w / D_outer_tube

            A_i = np.pi * D_inner_tube * self.L_hot_tube
            A_o = np.pi * D_outer_tube * self.L_hot_tube
            one_over_H += 1/h_i + A_i * np.log(D_outer_tube / D_inner_tube) / (2 * np.pi * k_tube * self.L_hot_tube) + (A_i / A_o) / h_o


        # TODO: check if this is correct
        # The handout seems to do an incorrect calculation so the formula needs to be checked

        Hcoeff = 1 / one_over_H
        Area = self.total_tubes * np.pi * D_inner_tube * self.L_hot_tube


        # TODO: solve thermal equations

        def heat_solve_iteration(Tout):
            T1out, T2out = Tout # cold and hot outlet temperatures

            cold_eq = self.mdot_cold * cp * (T1out - T1in) - Hcoeff * Area * Fscale * logmeanT(T1in, T1out, T2in, T2out)
            hot_eq = self.mdot_hot * cp * (T2in - T2out) - Hcoeff * Area * Fscale * logmeanT(T1in, T1out, T2in, T2out)

            return [hot_eq, cold_eq]
        
        try:
            solution = fsolve(heat_solve_iteration, [2*T1in, 0.5*T2in],
                              maxfev = max_thermal_iterations)
        except Exception as e:
            print(e)
            return
        
        T1out, T2out = solution
        LMTD = logmeanT(T1in, T1out, T2in, T2out)
        Qdot = Hcoeff * Area * Fscale * LMTD
        effectiveness = Qdot / (self.mdot_cold * cp * (T2in - T1in))

        self.LMTD = LMTD
        self.Qdot = Qdot
        self.effectiveness = effectiveness

        return effectiveness


    def is_geometrically_feasible(self):
        # performs collision detection to see if the heat exchanger is geometrically feasible

        # check square or triangle design packing of the N_tubes in a shell for the given pitch
        # also check length of tubes are within individual and total limits

        pass

