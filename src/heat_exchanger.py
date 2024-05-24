
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


def cold_mass_flow_from_dp(cold_dp, year = 2024): # cold_dp in Pa

    if year == 2024:
        cold_side_compressor_characteristic = cold_side_compressor_characteristic_2024
    elif year == 2023:
        cold_side_compressor_characteristic = cold_side_compressor_characteristic_2023
    elif year == 2022:
        cold_side_compressor_characteristic = cold_side_compressor_characteristic_2022
    elif year == 2019:
        cold_side_compressor_characteristic = cold_side_compressor_characteristic_2019

    
    return np.interp(cold_dp * 1e-5,
                     cold_side_compressor_characteristic[1],
                     cold_side_compressor_characteristic[0]) * rho_w / 1000
    
    f =  scipy.interpolate.interp1d(cold_side_compressor_characteristic[1], 
                                      cold_side_compressor_characteristic[0], 
                                      kind = 'cubic')
    return f(cold_dp * 1e-5) * rho_w / 1000


def hot_mass_flow_from_dp(hot_dp, year = 2024): # hot_dp in Pa

    if year == 2024:
        hot_side_compressor_characteristic = hot_side_compressor_characteristic_2024
    elif year == 2023:
        hot_side_compressor_characteristic = hot_side_compressor_characteristic_2023
    elif year == 2022:
        hot_side_compressor_characteristic = hot_side_compressor_characteristic_2022
    elif year == 2019:
        hot_side_compressor_characteristic = hot_side_compressor_characteristic_2019


    return np.interp(hot_dp * 1e-5,  # bar
                     hot_side_compressor_characteristic[1],
                     hot_side_compressor_characteristic[0]) * rho_w / 1000

    f = scipy.interpolate.interp1d(hot_side_compressor_characteristic[1],
                                    hot_side_compressor_characteristic[0],
                                    kind = 'cubic')
    return f(hot_dp * 1e-5) * rho_w / 1000
    

def dp_from_cold_mass_flow(mdot_cold, year = 2024):
        if year == 2024:
            cold_side_compressor_characteristic = cold_side_compressor_characteristic_2024
        elif year == 2023:
            cold_side_compressor_characteristic = cold_side_compressor_characteristic_2023
        elif year == 2022:
            cold_side_compressor_characteristic = cold_side_compressor_characteristic_2022
        elif year == 2019:
            cold_side_compressor_characteristic = cold_side_compressor_characteristic_2019
        # check if in the range for the compressor characteristics
        
        return np.interp(mdot_cold * 1000 / rho_w,
                        np.fliplr(cold_side_compressor_characteristic)[0],
                        np.fliplr(cold_side_compressor_characteristic)[1]) * 1e5  # Pa

def dp_from_hot_mass_flow(mdot_hot, year = 2024):
        if year == 2024:
            hot_side_compressor_characteristic = hot_side_compressor_characteristic_2024
        elif year == 2023:
            hot_side_compressor_characteristic = hot_side_compressor_characteristic_2023
        elif year == 2022:
            hot_side_compressor_characteristic = hot_side_compressor_characteristic_2022
        elif year == 2019:
            hot_side_compressor_characteristic = hot_side_compressor_characteristic_2019
        
        return np.interp(mdot_hot * 1000 / rho_w,
                            np.fliplr(hot_side_compressor_characteristic)[0],
                            np.fliplr(hot_side_compressor_characteristic)[1]) * 1e5  # Pa

def logmeanT(T1in, T1out, T2in, T2out):
    dt1 = (T2in - T1out)
    dt2 = (T2out - T1in)

    return (dt1 - dt2) / np.log(dt1 / dt2)

class e_NTU():

    ##Handbook of Heat Transfer 3rd Edition MCGRAW-HILL, Chapter 17
    def __init__(self, areatimesH, C_1, C_2, N_shell, N_tube, flow_path_entries_side):
        self.N_shell = N_shell
        self.N_tube = N_tube
        self.flow_path_entries_side = flow_path_entries_side

        self.C_1 = C_1
        self.C_2 = C_2
        self.C_min = np.min([C_1, C_2])
        self.C_max = np.max([C_1, C_2])
        self.C_ntu = self.C_min/self.C_max
        self.ntu_overall = areatimesH / self.C_min 
        self.ntu = self.ntu_overall / self.N_shell                                          ## essentially a cascade of N 1-2m shell-and-tubes, heat transfer split evenly between shell passes


    
    def effectiveness(self):

        if ((self.N_tube / self.N_shell) % 2 == 0):                                                 ## N-2mN shell-and-tube (N shell passes, 2mN tube passes)
            e_1 = 2 / (1 + self.C_ntu + (1 + self.C_ntu**2)**(1/2) * (1 + np.exp(-self.ntu *
                    (1 + self.C_ntu**2)**(1/2)))/(1-np.exp(-self.ntu * (1 + self.C_ntu**2)**(1/2))))
            e = (((1 - e_1 * self.C_ntu)/(1 - e_1))**self.N_shell - 1) / \
                (((1 - e_1 * self.C_ntu)/(1 - e_1))**self.N_shell - self.C_ntu)
        
        elif (self.N_shell == self.N_tube): # N-N passes can be considered the same as a 1-1 pass?
            if (self.flow_path_entries_side == Side.OPPOSITE):              ## N pass counterflow
                if (self.C_ntu >=0 and self.C_ntu < 1):
                    e = (1-np.exp(-self.ntu_overall*(1-self.C_ntu)))/(1-self.C_ntu*np.exp(-self.ntu_overall*(1-self.C_ntu)))
                elif (self.C_ntu ==1):
                    e = self.ntu_overall/(1+self.ntu_overall)

            elif (self.flow_path_entries_side == Side.SAME):                ## N pass parallelflow
                    e = (1-np.exp(-self.ntu*(1+self.C_ntu)))/(1+self.C_ntu)
        
        elif (self.N_shell == 1 and self.N_tube == 3 and self.flow_path_entries_side == Side.OPPOSITE): ## 1-3 shell-and-tube one parallelflow, 2 counterflow
            L = np.zeros(3)
            X = np.zeros_like(L)
            L[0] = -3/2 + (9/4 + self.C_ntu*(self.C_ntu - 1))**(1/2)
            L[1] = -3/2 - (9/4 + self.C_ntu*(self.C_ntu - 1))**(1/2)
            L[2] = -self.C_ntu
            delta = L[0] - L[1]
            for i in range(len(L)):
                X[i] = np.exp(L[i]*self.ntu/3)/(2*delta)
            if (self.C_ntu >=0 and self.C_ntu < 1):
                A = X[0]*(self.C_ntu + L[0])*(self.C_ntu - L[1])/(2*L[0]) - X[2]*delta - X[1] * (self.C_ntu + L[1]) * (self.C_ntu - L[0]) / (2*L[1]) + 1/(1-self.C_ntu)
            elif (self.C_ntu == 1):
                A = -np.exp(-self.ntu)/18 - np.exp(self.ntu/3)/2 + (self.ntu + 5)/9
            B = X[0] * (self.C_ntu - L[1]) - X[1] * (self.C_ntu - L[0]) + X[2] * delta
            C = X[1] * (3*self.C_ntu + L[0]) - X[0] * (3*self.C_ntu + L[1]) + X[2] * delta
            e = 1/self.C_ntu * (1 - C/(A*C + B**2))

        else:
            e = None
            logging.critical("Error: e_NTU undefined for this configuration")
            raise NotImplementedError("e_NTU for this configuration not implemented yet")
            
        return e
    
    def F_factor(self):      
                                                                           ##for the overall HX, use ntu_overall
        e = e_NTU.effectiveness(self)
        if (self.C_ntu >=0 and self.C_ntu < 1):
            F = 1/(self.ntu_overall * (1 - self.C_ntu)) * np.log((1-self.C_ntu * e)/(1-e))
        elif (self.C_ntu == 1):
            F = e/(self.ntu_overall * (1 - e))
        
        return F



'''
def E_NTU(NTU, C_rel, N_shell, N_tube):
    # N cold passes; 2N, 4N,... Hot passes
    if ((N_tube / N_shell) % 2 == 0):
        e_1 = 2 / (1 + C_rel + (1 + C_rel**2)**(1/2) * (1 + np.exp(-NTU *
                   (1 + C_rel**2)**(1/2)))/(1-np.exp(-NTU * (1 + C_rel**2)**(1/2))))
        e = (((1 - e_1 * C_rel)/(1 - e_1))**N_shell - 1) / \
            (((1 - e_1 * C_rel)/(1 - e_1))**N_shell - C_rel)

    # 1 Cold Pass; 1 Hot Pass Counterflow ONLY
    elif (N_shell == 1 and N_tube == 1):
        if (C_rel < 1 and C_rel >= 0):
            e = (1-np.exp(-NTU*(1-C_rel)))/(1-C_rel*np.exp(-NTU*(1-C_rel)))
        elif (C_rel == 1):
            e = NTU/(1+NTU)
    else:
        e = None
    return e

def GET_F(T1in, T2in, T1out, T2out, N_shell, N_tube, flow_path_entries_side):
    # A General Expression for the Determination of the Log Mean Temperature Correction Factor for Shell and Tube Heat Exchangers (Ahmad Fakheri) 2003
    ## only for even tube passes
    if ((N_tube / N_shell) % 2 == 0):
        p = (T1out - T1in)/(T2in - T1in)
        r = np.min([(T2in - T2out)/(T1out - T1in), (T1out - T1in)/(T2in - T2out)])
        if (r != 1):
            s = (r**2 + 1)**0.5 / (r-1)    
            w = ((1-p*r)/(1 - p))**(1/N_shell)
            F = s*np.log(w)/np.log((1 + w - s + s*w)/(1 + w + s - s*w))

        elif (r == 1):
            u = (N_shell - N_shell*p)/(N_shell - N_shell*p + p)
            F = 2**(1/2)*((1-u)/u)*(np.log((u/(1-u) + 2**-(1/2))/((u/(1-u) - 2**-(1/2)))))**(-1)

    elif (N_shell == 1 and N_tube == 1):
            if flow_path_entries_side == Side.OPPOSITE:
                F = 1

            elif flow_path_entries_side == Side.SAME:

                # page 1286 of Holman, J. P. “Heat Transfer”. 8th Edition, McGraw Hill.
                p = (T1out - T1in)/(T2in - T1in)
                r = np.min([(T2in - T2out)/(T1out - T1in), (T1out - T1in)/(T2in - T2out)])
                if r == 1:
                    F = 2 * p / ((p - 1) * np.log(1 - 2*p))
                else:
                    F = (r + 1) * np.log((1 - r * p) / (1 - p)) / ((r - 1) * np.log(1 - p * (1 + r)))

    else:
        logging.critical("Error: F undefined for this configuration")
        raise NotImplementedError("F for this configuration not implemented yet")
        F = None

    return F
'''

def pitch_from_tubes(tubes_per_section, N, pattern):
    # approximate pitch based on number of tubes
    '''if pattern == Pattern.SQUARE:
        k = 1 / np.sqrt(2)
    elif pattern == Pattern.TRIANGLE:
        k = 1 / np.sqrt(3)
    else:
        logging.error("Unknown pattern")

    pitch = k * D_shell / np.sqrt(N_tubes)
    '''
    if pattern == Pattern.SQUARE:
        a =  D_shell*(np.pi / (4*N))**(1/2)                       ## N is the number of shell or tube passes
        pitch = a * (tubes_per_section)**(-1/2)
    
    elif pattern == Pattern.TRIANGLE:
        a = D_shell*(np.pi/(N*3**(1/2)))**(1/2)
        n = -1/2 + 1/2 * (1+8*tubes_per_section)**(1/2)
        pitch = a / (n + 3**(1/2) - 1)
    else:
        logging.error("Unknown pattern")

    if pitch < D_outer_tube:
        logging.error("Pitch is less than the tube diameter")
        raise ValueError("Pitch is less than the tube diameter")
    
    elif pitch < D_outer_tube + pitch_offset:
        logging.warning("Tubes are closer than minimum set distance")

    #pitch = 1.25*D_outer_tube
    #print(pitch)

    return pitch

class Heat_Exchanger():
    def __init__(self, cold_fluid_path, hot_fluid_path, flow_path_entries_side):
        super().__init__()

        self.characteristic_year = 2024

        self.cold_path = cold_fluid_path
        self.hot_path = hot_fluid_path

        self.flow_path_entries_side = flow_path_entries_side

        cold_side_bends = 0
        hot_side_bends = 0
        self.total_tubes = 0
        self.total_baffles = 0

        for element in self.hot_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                self.total_tubes += element.tubes
                pattern = element.pattern
            if isinstance(element, U_Bend):
                hot_side_bends += 1
        for element in self.cold_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                self.total_baffles += element.baffles

            if isinstance(element, U_Bend):
                cold_side_bends += 1

        if cold_side_bends % 2 == hot_side_bends % 2:
            self.flow_path_exits_side = flow_path_entries_side
        elif flow_path_entries_side == Side.SAME:
            self.flow_path_exits_side = Side.OPPOSITE
        else:
            self.flow_path_exits_side = Side.SAME

        self.cold_flow_sections = cold_side_bends + 1
        self.hot_flow_sections = hot_side_bends + 1

        # initial values
        self.mdot = [0.3, 0.25]

        self.L_hot_tube = 0.35 - 2 * end_cap_width

        self.hydraulic_iteration_count = 0

        # TODO: change these to be actually correct
        self.hot_pressure_factor = 1
        self.cold_pressure_factor = 1

    def set_conditions(self, Tin):
        self.Tin = Tin

    def calc_dp(self, mdot):

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
                    # TODO: fix this temporary solution
                    pitch = pitch_from_tubes(element.tubes[j], self.hot_flow_sections, element.pattern[j])
                    
                    if element.pattern[j] == Pattern.SQUARE:
                        effective_d_shell = 1.27/D_outer_tube * (pitch**2 - 0.785 * D_outer_tube**2)
                    elif element.pattern[j] == Pattern.TRIANGLE:
                        effective_d_shell = 1.10/D_outer_tube * (pitch**2 - 0.917 * D_outer_tube**2)
                    else:
                        logging.error("Error: Unknown pattern")
                    
                    effective_d_shell  = (D_shell**2 - self.total_tubes*D_outer_tube**2)/(self.total_tubes*D_outer_tube)
                    

                    B_spacing = self.L_hot_tube / (element.baffles + 1)
                    A_shell_effective = (pitch - D_outer_tube) * \
                        B_spacing * D_shell / pitch
                    

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
                    
                    # TODO: fix this temporary solution
                    pitch = pitch_from_tubes(prev_element.tubes[i], self.cold_flow_sections, prev_element.pattern[i])

                    
                    B_spacing = self.L_hot_tube / (prev_element.baffles + 1)
                    A_shell_effective = (pitch - D_outer_tube) * B_spacing * D_shell / pitch

                    A_section = A_shell_effective / self.cold_flow_sections

                    v_shell = mdot_cold / (rho_w * A_section)
                    
                    DP_cold += element.loss_coefficient() * 0.5 * rho_w * v_shell ** 2/self.hot_flow_sections
            

        v_cold_nozzle = mdot_cold / (rho_w * A_nozzle)

        DP_cold += rho_w * v_cold_nozzle**2
        
        # Fudge factors based on experimental data in analysis.ipynb
        #DP_cold = 1.45443318e+00 * DP_cold + 1.22838965e+04
        #DP_hot = 4.37738030e-01 * DP_hot + 7.86899623e+03

        DP_cold = 2.0782816914547007 * DP_cold + 6316.406315165725
        DP_hot = 0.5207595964160064 * DP_hot + 5494.462263156731

        #DP_cold = 1.984012962563786 * DP_cold +7217.109422930143
        #DP_cold = 1.6953864289383282 * DP_cold-10686.652389444924

        return DP_cold, DP_hot

    def hydraulic_iteration(self, mdot):
        mdot_cold, mdot_hot = mdot

        dp_cold, dp_hot = self.calc_dp(mdot)

        # check if mdot is in the range for the compressor characteristics
        
        new_mdot_cold = cold_mass_flow_from_dp(dp_cold, self.characteristic_year)
        new_mdot_hot = hot_mass_flow_from_dp(dp_hot, self.characteristic_year)
        error = [new_mdot_cold - mdot_cold, new_mdot_hot - mdot_hot]

        #dp_cold_calc = dp_from_cold_mass_flow(mdot_cold, self.characteristic_year)
        #dp_hot_calc = dp_from_hot_mass_flow(mdot_hot, self.characteristic_year)
        #error = [dp_cold_calc - dp_cold, dp_hot_calc - dp_hot]

        return error
    
    def hydraulic_iteration_squared(self, mdot):
        mdot_cold, mdot_hot = mdot

        dp_cold, dp_hot = self.calc_dp(mdot)
        
        cold_rel_dp = dp_from_cold_mass_flow(mdot_cold) - dp_cold
        hot_rel_dp = dp_from_hot_mass_flow(mdot_hot) - dp_hot

        return cold_rel_dp**2 + hot_rel_dp**2

    def calc_area_times_H(self, mdot):
        mdot_cold, mdot_hot = mdot
        areatimesH = 0

        for i, element in enumerate(self.hot_path.elements):
            if not isinstance(element, Heat_Transfer_Element):
                continue

            mdot_hot_tube = mdot_hot / element.tubes
            v_hot_tube = mdot_hot_tube / (rho_w * A_tube)
            Re_hot = v_hot_tube * rho_w * D_inner_tube / mu

            # obtain the heat transfer coefficient for the inner and outer tubes
            # TODO: fix this temporary solution
            pitch = pitch_from_tubes(element.tubes, self.hot_flow_sections, element.pattern)


            
            if element.pattern == Pattern.SQUARE:
                effective_d_shell = 1.27/D_outer_tube * (pitch**2 - 0.785 * D_outer_tube**2) 
                c = c_square
            elif element.pattern == Pattern.TRIANGLE:
                effective_d_shell = 1.10/D_outer_tube * (pitch**2 - 0.917 * D_outer_tube**2) 
                c = c_triangle
            else:
                logging.error("Error: Unknown pattern")
            

            B_spacing = self.L_hot_tube / (element.baffles + 1)
            A_shell_effective = (pitch - D_outer_tube) * \
                B_spacing * D_shell / (pitch)
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
            
            areatimesH += element.tubes * np.pi * D_inner_tube * self.L_hot_tube / one_over_H 

        # TODO: investigate why this is correct
        return areatimesH

    def LMTD_heat_solve_iteration(self, Tout):
        T1in, T2in = self.Tin
        T1out, T2out = Tout  # cold and hot outlet temperatures
        
        areatimesH = self.calc_area_times_H(self.mdot)
        C_1 = self.mdot[0] * cp
        C_2 = self.mdot[1] * cp
        N_shell = self.cold_flow_sections
        N_tube = self.hot_flow_sections
        
        # TODO: calculate this for various cold flow sections
        NTU = e_NTU(areatimesH, C_1, C_2, N_shell, N_tube, self.flow_path_entries_side)

        Fscale = e_NTU.F_factor(NTU)
        self.Fscale = Fscale
        
        T1in, T2in = self.Tin

        cold_eq = C_1 * \
            (T1out - T1in) - self.area_times_H * Fscale * \
            logmeanT(T1in, T1out, T2in, T2out)
        hot_eq = C_2 * \
            (T2in - T2out) - self.area_times_H * Fscale * \
            logmeanT(T1in, T1out, T2in, T2out)

        return [hot_eq, cold_eq]


    def calc_mdot(self, x = None):

        try:
            mdot, _, res, *_ = fsolve(self.hydraulic_iteration, 
                                     self.mdot,
                                     xtol = hydraulic_error_tolerance,
                                     maxfev = max_hydraulic_iterations,
                                     full_output = True
                                     )
            assert res == 1
        except AssertionError:
            logging.error("Hydraulic analysis failed to converge")
            return None
        except Exception as e:
            logging.critical(f"Hydraulic analysis failed: {e}")
            return None
        
        return mdot

    def compute_effectiveness(self, method = "LMTD", optimiser = 'fsolve'):

        # HYDRAULIC ANALYSIS

        if (optimiser =='brute'):
            try:
                grid = (slice(np.min(cold_side_compressor_characteristic_2024[0]), np.max(cold_side_compressor_characteristic_2024[0]), 0.01), slice(np.min(hot_side_compressor_characteristic_2024[0]), np.max(hot_side_compressor_characteristic_2024[0]), 0.01))
                self.mdot = scipy.optimize.brute(
                    self.hydraulic_iteration_squared, 
                    ranges=grid, 
                    finish=None)
            
                assert np.isfinite(self.mdot).all()
            except Exception as e:
                logging.error(f"Hydraulic analysis failed to converge: {e}")
                return False
        if (optimiser == 'fsolve'):
            try:
                self.mdot, _, res, *_ = fsolve(self.hydraulic_iteration,
                                   self.mdot,
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

        # THERMAL ANALYSIS

        try:
            self.area_times_H = self.calc_area_times_H(self.mdot)
        except Exception as e:
            logging.error(f"Failed to calculate area times H: {e}")
            return False

        if (method == 'LMTD'):
            
            try:
                solution, _, res, *_ = fsolve(self.LMTD_heat_solve_iteration,
                                  [2*T1in, 0.5*T2in],
                                  xtol = thermal_error_tolerance,
                                  maxfev = max_thermal_iterations,
                                  full_output = True)
                assert res == 1
            except AssertionError:
                logging.error("LMTD analysis failed to converge")
                return False
            except Exception as e:
                logging.error(f"LMTD analysis failed to converge: {e}")
                return False
            
            T1out, T2out = solution
            C_min = np.min([cp*mdot_hot, cp*mdot_cold])
            LMTD = logmeanT(T1in, T1out, T2in, T2out)    
            Qdot = mdot_cold * cp * (T1out - T1in)
            Qdot_max = (C_min * (T2in - T1in))
            effectiveness = Qdot / Qdot_max
            DT_min = np.max([(T1out - T1in), -(T2out - T2in)])

            self.Tout = [T1out, T2out]
            self.LMTD = LMTD
            self.Qdot = Qdot
            #Qdot_corrected = 5.70618187e-01 * Qdot + 6.32419119e+03
            #self.Qdot = Qdot_corrected
            self.DT_min = DT_min

        elif (method == 'E_NTU'):

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

            # could possibly iterate to find Fscale with the new T1out and T2out

        self.effectiveness = effectiveness
        Qdot_corrected = 0.5083278756212372 * Qdot + 6425.748610071553
        #Qdot_corrected = 0.5173616955929328 * Qdot + 6347.84023109489
        self.Qdot = Qdot_corrected

        return True

    def calc_mass(self, *args):
        
        baffle_area_occlusion_ratio = 0.75

        mpipes = self.total_tubes * self.L_hot_tube * rho_copper_tube
        mshell = (self.L_hot_tube + end_cap_width_nozzle)* rho_acrylic_tube

        m_seals = 4 * (m_nozzle + m_large_O + self.total_tubes * m_small_O)

        #baffle_area = (D_shell**2/4)*(np.pi - np.arccos(1-2*baffle_cut/D_shell)) + (D_shell-baffle_cut)*(D_shell**2-(D_shell-baffle_cut)**2)**0.5
        # say two tubes are unbaffled in the unoccluded area
        unbaffled_tubes = 2
        baffle_area = baffle_area_occlusion_ratio * A_shell - (self.total_tubes - unbaffled_tubes) * np.pi * D_outer_tube**2 / 4

        m_baffles = baffle_area * rho_abs * self.total_baffles / self.cold_flow_sections
        
        m_caps = 2 * end_cap_width * A_shell * rho_abs # TODO: check end cap mass and inlcude mass of hot and cold section dividers

        m_seperator = (self.cold_flow_sections - 1) * (self.L_hot_tube - end_cap_width_nozzle) * (D_shell) * rho_abs
        if self.cold_flow_sections > 2:
            logging.error("Warning: Cold flow sections is greater than 2, this is not supported")


        return (
            mpipes +
            mshell +
            m_seals +
            m_baffles +
            m_caps + 
            m_seperator
        )

    def get_pitch(self):
        pitches = []

        for element in self.hot_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                # TODO: fix this temporary solution
                pitch = pitch_from_tubes(element.tubes, self.hot_flow_sections, element.pattern)
                pitches.append(pitch)
        
        return pitches


    def set_geometry(self, length, tubes, baffles):
        # tubes and baffle is per element

        self.L_hot_tube = length

        hot_stages =  self.hot_flow_sections
        cold_stages =  self.cold_flow_sections

        tubes = np.rint(tubes)
        baffles = np.rint(baffles)

        '''if isinstance(tubes, int):
            tubes = [tubes] * hot_stages
        if isinstance(baffles, int):
            baffles = [baffles] * cold_stages
        '''
        assert len(tubes) == hot_stages
        assert len(baffles) == cold_stages

        self.total_tubes = sum(tubes)
        self.total_baffles = sum(baffles)

        baffles_per_hot_stage = np.zeros(hot_stages)
        tubes_per_cold_stage = []
        tubes_per_cold_stage_temp = []
        for i in range(cold_stages):
            for j in range(hot_stages//cold_stages):
                baffles_per_hot_stage[i*hot_stages//cold_stages + j] = baffles[i]
                tubes_per_cold_stage_temp.append(tubes[i*hot_stages//cold_stages + j])
            tubes_per_cold_stage.append(tubes_per_cold_stage_temp)
            tubes_per_cold_stage_temp = []
                
        i = 0
        for element in self.hot_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                element.tubes = tubes[i]
                element.baffles = baffles_per_hot_stage[i]
                i += 1
        
        i = 0
        for element in self.cold_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                element.tubes = tubes_per_cold_stage[i]
                element.baffles = baffles[i]
                i += 1

    
    def get_random_geometry_copy(self, constraints = None):

        new_HE = copy.deepcopy(self)

        # set random values for the geometry parameters
        new_HE.set_geometry(0.35,
                            np.random.randint(1, 20),
                            np.random.randint(1, 20))

        return new_HE


def build_heat_exchanger(tubes_per_stage, baffles_per_stage, length, flow_path_entries_side, hot_tube_pattern, cold_tube_pattern = None):
    # build the heat exchanger

    hot_stages = len(tubes_per_stage)
    cold_stages = len(baffles_per_stage)

    tubes_per_stage = np.rint(tubes_per_stage)
    baffles_per_stage = np.rint(baffles_per_stage)

    #cold_tube_patterns = np.empty((cold_stages, hot_stages//cold_stages), dtype=  'Pattern'>)
    #hot_tube_patterns = np.empty(hot_stages, dtype=enum 'Pattern')
    cold_tube_patterns_temp = []
    cold_tube_patterns = []

    if isinstance(hot_tube_pattern, Pattern):
        hot_tube_patterns = [hot_tube_pattern] * hot_stages
    else:
        hot_tube_patterns = hot_tube_pattern
    for i in range(cold_stages):
        for j in range(hot_stages//cold_stages):
            cold_tube_patterns_temp.append(hot_tube_patterns[i*hot_stages//cold_stages + j])
        cold_tube_patterns.append(cold_tube_patterns_temp)
        cold_tube_patterns_temp = []

    '''if cold_tube_pattern is None:
        cold_tube_patterns = [hot_tube_pattern] * cold_stages
    elif isinstance(cold_tube_pattern, Pattern):
        cold_tube_patterns = [cold_tube_pattern] * cold_stages
    else:
        cold_tube_patterns = cold_tube_pattern'''
    
    Hot_path = Fluid_Path(rho_w, mu, cp, k_w)

    baffles_per_hot_stage = np.zeros(hot_stages)
    tubes_per_cold_stage = []
    tubes_per_cold_stage_temp = []
    for i in range(cold_stages):
        for j in range(hot_stages//cold_stages):
            baffles_per_hot_stage[i*hot_stages//cold_stages + j] = baffles_per_stage[i]
            tubes_per_cold_stage_temp.append(tubes_per_stage[i*hot_stages//cold_stages + j])
        tubes_per_cold_stage.append(tubes_per_cold_stage_temp)
        tubes_per_cold_stage_temp = []

    Hot_path.add_element(Entry_Constriction())
    Hot_path.add_element(
        Heat_Transfer_Element(tubes_per_stage[0], 
                              baffles_per_hot_stage[0], 
                            flow_direction=Direction.COUNTERFLOW,
                            tube_pattern = hot_tube_patterns[0])
    )
    Hot_path.add_element(Exit_Expansion())
    for i in range(1, hot_stages):
        Hot_path.add_element(U_Bend())
        Hot_path.add_element(Entry_Constriction())
        Hot_path.add_element(
            Heat_Transfer_Element(tubes_per_stage[i], 
                                  baffles_per_hot_stage[i], 
                                flow_direction=Direction.COUNTERFLOW,
                                tube_pattern = hot_tube_patterns[i])
        )
        Hot_path.add_element(Exit_Expansion())

    Cold_path = Fluid_Path(rho_w, mu, cp, k_w)


    Cold_path.add_element(
        Heat_Transfer_Element(
                            tubes_per_cold_stage[0], 
                            baffles_per_stage[0], 
                            flow_direction=Direction.COUNTERFLOW,
                            tube_pattern = cold_tube_patterns[0])
    )
    for i in range(1,cold_stages):

        Cold_path.add_element(U_Bend())
        Cold_path.add_element(
            Heat_Transfer_Element(
                                tubes_per_cold_stage[i], 
                                baffles_per_stage[i], 
                                flow_direction=Direction.COFLOW,
                                tube_pattern = cold_tube_patterns[i])
        )

    HeatExchanger = Heat_Exchanger(Cold_path, Hot_path, 
                            flow_path_entries_side)
    
    HeatExchanger.L_hot_tube = length

    return HeatExchanger