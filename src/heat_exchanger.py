
import numpy as np
import matplotlib.pyplot as plt

import copy

from scipy.optimize import fsolve

from constants import *

from fluid_path import Entry_Constriction, Exit_Expansion, U_Bend, Heat_Transfer_Element


def cold_mass_flow_from_dp(cold_dp):

    return np.interp(cold_dp * 1e-5,
                     cold_side_compressor_characteristic[1],
                     cold_side_compressor_characteristic[0]) * rho_w / 1000


def hot_mass_flow_from_dp(hot_dp):

    return np.interp(hot_dp * 1e-5,  # bar
                     hot_side_compressor_characteristic[1],
                     hot_side_compressor_characteristic[0]) * rho_w / 1000

def dp_from_cold_mass_flow(mdot_cold):
    
        return np.interp(mdot_cold * 1000 / rho_w,
                        cold_side_compressor_characteristic[0],
                        cold_side_compressor_characteristic[1]) * 1e5  # Pa

def dp_from_hot_mass_flow(mdot_hot):
        
            return np.interp(mdot_hot * 1000 / rho_w,
                            hot_side_compressor_characteristic[0],
                            hot_side_compressor_characteristic[1]) * 1e5  # Pa

def logmeanT(T1in, T1out, T2in, T2out):
    dt1 = (T2in - T1out)
    dt2 = (T2out - T1in)

    return (dt1 - dt2) / np.log(dt1 / dt2)


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


class Heat_Exchanger():
    def __init__(self, cold_fluid_path, hot_fluid_path, flow_path_entries_side):
        super().__init__()

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
        self.mdot_hot = 0.3
        self.mdot_cold = 0.3

        self.L_hot_tube = 0.35

        # TODO: vary this with the heat transfer element pattern
        self.pitch = 0.014  # Y in handout

        self.hydraulic_iteration_count = 0

        # TODO: change these to be actually correct
        self.hot_pressure_factor = 1
        self.cold_pressure_factor = 1

    def calc_dp(self):

        # HOT STREAM

        self.DP_hot = 0

        for i, element in enumerate(self.hot_path.elements):

            if isinstance(element, Heat_Transfer_Element):

                mdot_hot_tube = self.mdot_hot / element.tubes
                v_hot_tube = mdot_hot_tube / (rho_w * A_tube)

                Re_hot = v_hot_tube * rho_w * D_inner_tube / mu
                f_hot = element.friction_coefficient(Re_hot, 0.00015)

                self.DP_hot += 0.5 * rho_w * v_hot_tube**2 * \
                    (f_hot * self.L_hot_tube / D_inner_tube)

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
                self.DP_hot += 0.5 * rho_w * v_hot_tube ** 2 * \
                    element.loss_coefficient(Re_hot, sigma)

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
                self.DP_hot += 0.5 * rho_w * v_hot_tube ** 2 * \
                    element.loss_coefficient(Re_hot, sigma)

            if isinstance(element, U_Bend):
                #TODO: Add bend loss
                self.DP_hot += 0

        v_hot_nozzle = mdot_hot_tube / (rho_w * A_nozzle)
        self.DP_hot += rho_w * v_hot_nozzle ** 2  # nozzle loss

        # COLD STREAM

        self.DP_cold = 0

        for i, element in enumerate(self.cold_path.elements):

            if isinstance(element, Heat_Transfer_Element):

                if element.pattern == Pattern.SQUARE:
                    a_factor = a_square
                elif element.pattern == Pattern.TRIANGLE:
                    a_factor = a_triangle
                else:
                    print("Error: Unknown pattern")

                B_spacing = self.L_hot_tube / (element.baffles + 1)
                A_shell_effective = (self.pitch - D_outer_tube) * \
                    B_spacing * D_shell / self.pitch

                v_shell = self.mdot_cold / (rho_w * A_shell_effective)

                effective_d_shell = D_shell * A_shell_effective / A_shell
                Re_shell = v_shell * rho_w * effective_d_shell / mu

                self.DP_cold += 4 * a_factor * \
                    Re_shell ** (-0.15) * element.tubes * rho_w * v_shell ** 2

            if isinstance(element, U_Bend):
                self.DP_cold += 0

        v_cold_nozzle = self.mdot_cold / (rho_w * A_nozzle)

        self.DP_cold += rho_w * v_cold_nozzle**2

        return self.DP_cold, self.DP_hot


    def calc_rel_rise(self, x):
        self.calc_dp()
        
        cold_rel_dp = dp_from_cold_mass_flow(self.mdot_cold) - self.DP_cold
        hot_rel_dp = dp_from_hot_mass_flow(self.mdot_hot) - self.DP_hot

        return cold_rel_dp, hot_rel_dp

    def set_mass_flow(self, mdot):
        self.mdot_cold, self.mdot_hot = mdot

    def compute_effectiveness(self, Tin, method = "LMTD"):

        T1in, T2in = Tin

        # THERMAL ANALYSIS

        Fscale = 1

        areatimesH = 0

        for i, element in enumerate(self.hot_path.elements):
            if not isinstance(element, Heat_Transfer_Element):
                continue

            # TODO: do something with element.direction
            # to calculate Fscale

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
            A_shell_effective = (self.pitch - D_outer_tube) * \
                B_spacing * D_shell / self.pitch

            v_shell = self.mdot_cold / (rho_w * A_shell_effective)
            effective_d_shell = D_shell * A_shell_effective / A_shell
            Re_shell = v_shell * rho_w * effective_d_shell / mu

            Nu_i = 0.023 * Re_hot ** 0.8 * Pr ** 0.3
            Nu_o = c_factor * Re_shell ** 0.6 * Pr ** 0.3

            h_i = Nu_i * k_w / D_inner_tube
            h_o = Nu_o * k_w / D_outer_tube

            A_i = np.pi * D_inner_tube * self.L_hot_tube
            A_o = np.pi * D_outer_tube * self.L_hot_tube
            one_over_H = 1/h_i + A_i * np.log(D_outer_tube / D_inner_tube) / (
                2 * np.pi * k_tube * self.L_hot_tube) + (A_i / A_o) / h_o

            areatimesH += element.tubes * np.pi * D_inner_tube * self.L_hot_tube / one_over_H
            

        # TODO: check if this is correct
        # The handout seems to do an incorrect calculation so the formula needs to be checked

        if (method == 'LMTD'):
            def LMTD_heat_solve_iteration(Tout):
                T1out, T2out = Tout  # cold and hot outlet temperatures

                cold_eq = self.mdot_cold * cp * \
                    (T1out - T1in) - areatimesH * Fscale * \
                    logmeanT(T1in, T1out, T2in, T2out)
                hot_eq = self.mdot_hot * cp * \
                    (T2in - T2out) - areatimesH * Fscale * \
                    logmeanT(T1in, T1out, T2in, T2out)

                return [hot_eq, cold_eq]

            try:
                solution = fsolve(LMTD_heat_solve_iteration,
                                  [2*T1in, 0.5*T2in])
            except Exception as e:
                print(e)
                return

            T1out, T2out = solution
            LMTD = logmeanT(T1in, T1out, T2in, T2out)
            Qdot = areatimesH * Fscale * LMTD
            effectiveness = Qdot / (self.mdot_cold * cp * (T2in - T1in))

            self.LMTD = LMTD
            self.Qdot = Qdot

        elif (method == 'E_NTU'):
            Area = self.total_tubes * np.pi * D_inner_tube * self.L_hot_tube

            N_shell = self.cold_flow_sections
            N_tube = self.hot_flow_sections
            C_min = np.min([cp*self.mdot_hot, cp*self.mdot_cold])
            C_max = np.max([cp*self.mdot_hot, cp*self.mdot_cold])
            C_rel = C_min/C_max
            NTU = Hcoeff * Area / C_min
            effectiveness = E_NTU(NTU, C_rel, N_shell, N_tube)

        self.effectiveness = effectiveness

        return effectiveness

    def calc_mass(self, x = None):
        
        baffle_area_occlusion_ratio = 0.8

        mpipes = self.total_tubes * self.L_hot_tube * rho_copper_tube
        mshell = self.L_hot_tube * rho_acrylic_tube

        m_seals = 4 * (m_nozzle + m_large_O + self.total_tubes * m_small_O)

        m_baffles = baffle_area_occlusion_ratio * A_shell * rho_abs * self.total_baffles / self.cold_flow_sections
        
        m_caps = 2 * 0.01 * A_shell * rho_abs # TODO: check end cap mass

        return (
            mpipes +
            mshell +
            m_seals +
            m_baffles +
            m_caps
        )

    def is_geometrically_feasible(self):
        # TODO: check if the heat exchanger is geometrically feasible

        # performs collision detection to see if the heat exchanger is geometrically feasible

        # check square or triangle design packing of the N_tubes in a shell for the given pitch
        # also check length of tubes are within individual and total limits

        pass

    def set_geometry(self, L, pitch, tubes, baffles):
        
        for element in self.hot_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                element.tubes = tubes
                element.baffles = baffles

        for element in self.cold_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                element.tubes = tubes
                element.baffles = baffles

        self.L_hot_tube = L
        self.pitch = pitch
    
    def get_random_geometry_copy(self, constraints = None):

        new_HE = copy.deepcopy(self)

        # set random values for the geometry parameters
        new_HE.set_geometry(np.random.uniform(0.1, max_HE_length),
                            np.random.uniform(0.01, 0.02),
                            np.random.randint(10, 20),
                            np.random.randint(10, 20))

        return new_HE

