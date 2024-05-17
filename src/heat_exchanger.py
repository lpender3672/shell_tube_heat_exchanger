
import numpy as np
import matplotlib.pyplot as plt

import copy

from scipy.optimize import fsolve
import scipy
import scipy.optimize

from constants import *

from fluid_path import Entry_Constriction, Exit_Expansion, U_Bend, Heat_Transfer_Element


def cold_mass_flow_from_dp(cold_dp):

    return np.interp(cold_dp * 1e-5,
                     cold_side_compressor_characteristic_2024[1],
                     cold_side_compressor_characteristic_2024[0]) * rho_w / 1000


def hot_mass_flow_from_dp(hot_dp):

    return np.interp(hot_dp * 1e-5,  # bar
                     hot_side_compressor_characteristic_2024[1],
                     hot_side_compressor_characteristic_2024[0]) * rho_w / 1000


def dp_from_cold_mass_flow(mdot_cold):
        # check if in the range for the compressor characteristics
        
        return np.interp(mdot_cold * 1000 / rho_w,
                        np.fliplr(cold_side_compressor_characteristic_2024)[0],
                        np.fliplr(cold_side_compressor_characteristic_2024)[1]) * 1e5  # Pa

def dp_from_hot_mass_flow(mdot_hot):
        
        return np.interp(mdot_hot * 1000 / rho_w,
                            np.fliplr(hot_side_compressor_characteristic_2024)[0],
                            np.fliplr(hot_side_compressor_characteristic_2024)[1]) * 1e5  # Pa

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

def GET_F(T1in, T2in, T1out, T2out, N, flow_path_entries_side):
    # reference? 
    ## only for even tube passes
    if N > 1:
        p = (T1out - T1in)/(T2in - T1in)
        r = np.min([(T2in - T2out)/(T1out - T1in), (T2in - T2out)/(T1out - T1in)])
        #r = (T2in - T2out)/(T1out - T1in)
        if (r != 1):
            s = (r**2 + 1)**0.5 / (r-1)    
            w = ((1-p*r)/(1 - p))**(1/N)
            F = s*np.log(w)/np.log((1 + w - s + s*w)/(1 + w + s - s*w))

        elif (r == 1):
            u = (N - N*p)/(N - N*p + p)
            F = 2**(1/2)*((1-u)/u)*(np.log((u/(1-u) + 2**-(1/2))/((u/(1-u) - 2**-(1/2)))))**(-1)
    elif N == 1:

        if flow_path_entries_side == Side.OPPOSITE:
            F = 1
        else:
            raise NotImplementedError("F for coflow not implemented yet")
            # page 1286 of Holman, J. P. “Heat Transfer”. 8th Edition, McGraw Hill.
            # doesnt converge
            p = (T1out - T1in)/(T2in - T1in)
            r = (T2in - T2out)/(T1out - T1in)
            if r == 1:
                F = 2 * p / ((p - 1) * np.log(1 - 2*p))
            else:
                F = (r + 1) * np.log((1 - r * p) / (1 - p)) / ((r - 1) * np.log(1 - p * (1 + r)))
            print(F)
    else:
        print("Error: F undefined for this configuration")
        F = None

    return F

def pitch_from_tubes(tubes, pattern):
    # approximate pitch based on number of tubes
    if pattern == Pattern.SQUARE:
        k = 1 / np.sqrt(2)
    elif pattern == Pattern.TRIANGLE:
        k = 1 / np.sqrt(3)
    else:
        print("Error: Unknown pattern")

    pitch = k * D_shell / np.sqrt(tubes)

    return pitch

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

        self.L_hot_tube = 0.35

        self.pitch = pitch_from_tubes(self.total_tubes, pattern)
        if self.pitch < D_outer_tube:
            print("Warning: Pitch is less than the tube diameter")
        #self.pitch = 0.014

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
                
                if element.pattern == Pattern.SQUARE:
                    effective_d_shell = 1.27/D_outer_tube * (self.pitch**2 - 0.785 * D_outer_tube**2) * self.cold_flow_sections**(-1/2)
                elif element.pattern == Pattern.TRIANGLE:
                    effective_d_shell = 1.10/D_outer_tube * (self.pitch**2 - 0.917 * D_outer_tube**2) * self.cold_flow_sections**(-1/2)
                else:
                    print("Error: Unknown pattern")

                B_spacing = self.L_hot_tube / (element.baffles + 1)
                A_shell_effective = (self.pitch - D_outer_tube) * \
                    B_spacing * D_shell / (self.pitch * self.cold_flow_sections)

                v_shell = mdot_cold / (rho_w * A_shell_effective)

                Re_shell = v_shell * rho_w * effective_d_shell / mu

                j_f = 0.202 * Re_shell**(-0.153)

                DP_cold += 4 * j_f * (D_shell / effective_d_shell) * (element.baffles + 1) \
                    * rho_w * v_shell ** 2
            
            if isinstance(element, U_Bend):

                try:
                    prev_element = self.cold_path.elements[i-1]
                    assert isinstance(prev_element, Heat_Transfer_Element)
                except (IndexError, AssertionError):
                    raise ValueError("U Bend must be preceded by a heat transfer element")
                
                B_spacing = self.L_hot_tube / (prev_element.baffles + 1)
                A_shell_effective = (self.pitch - D_outer_tube) * B_spacing * D_shell / self.pitch

                A_section = A_shell_effective / self.cold_flow_sections

                v_shell = mdot_cold / (rho_w * A_section)
                
                DP_cold += element.loss_coefficient() * 0.5 * rho_w * v_shell ** 2
            

        v_cold_nozzle = mdot_cold / (rho_w * A_nozzle)

        DP_cold += rho_w * v_cold_nozzle**2

        return DP_cold, DP_hot

    def hydraulic_iteration(self, mdot):
        mdot_cold, mdot_hot = mdot

        dp_cold, dp_hot = self.calc_dp(mdot)

        # check if mdot is in the range for the compressor characteristics
        
        new_mdot_cold = cold_mass_flow_from_dp(dp_cold)
        new_mdot_hot = hot_mass_flow_from_dp(dp_hot)

        error = [new_mdot_cold - mdot_cold, new_mdot_hot - mdot_hot]

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

            # TODO: do something with element.direction
            # to calculate Fscale

            mdot_hot_tube = mdot_hot / element.tubes
            v_hot_tube = mdot_hot_tube / (rho_w * A_tube)
            Re_hot = v_hot_tube * rho_w * D_inner_tube / mu

            # obtain the heat transfer coefficient for the inner and outer tubes

            if element.pattern == Pattern.SQUARE:
                effective_d_shell = 1.27/D_outer_tube * (self.pitch**2 - 0.785 * D_outer_tube**2) * self.cold_flow_sections**(-1/2)
            elif element.pattern == Pattern.TRIANGLE:
                effective_d_shell = 1.10/D_outer_tube * (self.pitch**2 - 0.917 * D_outer_tube**2) * self.cold_flow_sections**(-1/2)
            else:
                print("Error: Unknown pattern")

            B_spacing = self.L_hot_tube / (element.baffles + 1)
            A_shell_effective = (self.pitch - D_outer_tube) * \
                B_spacing * D_shell / (self.pitch* self.cold_flow_sections)
            
            v_shell = mdot_cold / (rho_w * A_shell_effective)

            Re_shell = v_shell * rho_w * effective_d_shell / mu

            j_h = 0.4275*Re_shell**(-0.466)

            Nu_i = 0.023 * Re_hot ** 0.8 * Pr ** 0.33
            Nu_o = j_h * Re_shell * Pr ** 0.33

            h_i = Nu_i * k_w / D_inner_tube
            h_o = Nu_o * k_w / D_outer_tube

            A_i = np.pi * D_inner_tube * self.L_hot_tube
            A_o = np.pi * D_outer_tube * self.L_hot_tube
            one_over_H = 1/h_i + A_i * np.log(D_outer_tube / D_inner_tube) / (
                2 * np.pi * k_tube * self.L_hot_tube) + (A_i / A_o) / h_o
            
            areatimesH += element.tubes * np.pi * D_inner_tube * self.L_hot_tube / one_over_H

        # TODO: investigate why this is correct
        return areatimesH * self.cold_flow_sections

    def LMTD_heat_solve_iteration(self, Tout):
        mdot_cold, mdot_hot = self.mdot
        T1in, T2in = self.Tin
        T1out, T2out = Tout  # cold and hot outlet temperatures
        
        # TODO: calculate this for various cold flow sections

        Fscale = GET_F(T1in, T2in, T1out, T2out, self.hot_flow_sections, self.flow_path_entries_side)
        
        T1in, T2in = self.Tin

        cold_eq = mdot_cold * cp * \
            (T1out - T1in) - self.area_times_H * Fscale * \
            logmeanT(T1in, T1out, T2in, T2out)
        hot_eq = mdot_hot * cp * \
            (T2in - T2out) - self.area_times_H * Fscale * \
            logmeanT(T1in, T1out, T2in, T2out)

        return [hot_eq, cold_eq]


    def calc_mdot(self, x = None):

        try:
            mdot = fsolve(self.hydraulic_iteration, 
                                     self.mdot
                                     )
            
            assert np.isfinite(self.mdot).all()
        except Exception as e:
            print("Failed to solve hydraulic analsis")
            print(e)
            return None

        return mdot

    def compute_effectiveness(self, method = "LMTD", optimiser = 'brute'):

        # HYDRAULIC ANALYSIS

        if (optimiser =='brute'):
            try:
                grid = (slice(np.min(cold_side_compressor_characteristic_2024[0]), np.max(cold_side_compressor_characteristic_2024[0]), 0.01), slice(np.min(hot_side_compressor_characteristic_2024[0]), np.max(hot_side_compressor_characteristic_2024[0]), 0.01))
                self.mdot = scipy.optimize.brute(self.hydraulic_iteration_squared, ranges=grid, finish=None )
            
                assert np.isfinite(self.mdot).all()
            except Exception as e:
                print("Failed to solve hydraulic analsis")
                print(e)
                return False
        if (optimiser == 'fsolve'):
            try:
                self.mdot = fsolve(self.hydraulic_iteration,self.mdot)
                assert np.isfinite(self.mdot).all()
            except Exception as e:
                print("Failed to solve hydraulic analsis")
                print(e)
                return False
        
        mdot_cold, mdot_hot = self.mdot
        T1in, T2in = self.Tin

        # THERMAL ANALYSIS

        self.area_times_H = self.calc_area_times_H(self.mdot)

        if (method == 'LMTD'):
            
            try:
                solution = fsolve(self.LMTD_heat_solve_iteration,
                                  [2*T1in, 0.5*T2in])
            except Exception as e:
                print("Failed to solve thermal analsis")
                print(e)
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
            self.DT_min = DT_min

        elif (method == 'E_NTU'):

            N_shell = self.cold_flow_sections
            N_tube = self.hot_flow_sections
            C_min = np.min([cp*mdot_hot, cp*mdot_cold])
            C_max = np.max([cp*mdot_hot, cp*mdot_cold])
            C_rel = C_min/C_max

            NTU = (self.area_times_H / self.cold_flow_sections) / C_min

            effectiveness = E_NTU(NTU, C_rel, N_shell, N_tube)
            Qdot_max = (C_min * (T2in - T1in))
            Qdot = effectiveness*Qdot_max

            self.Qdot = Qdot
            self.NTU = NTU

            Qdotmax = mdot_hot * cp * (T2in - T1in)

            T1out = T1in + effectiveness * Qdotmax / (mdot_cold * cp)
            T2out = T2in - effectiveness * Qdotmax / (mdot_hot * cp)
            self.Tout = [T1out, T2out]
            self.LMTD = logmeanT(T1in, T1out, T2in, T2out)
            self.Qdot = mdot_cold * cp * (T1out - T1in)

            # could possibly iterate to find Fscale with the new T1out and T2out

        self.effectiveness = effectiveness

        return True

    def calc_mass(self, *args):
        
        baffle_area_occlusion_ratio = 0.75

        mpipes = self.total_tubes * self.L_hot_tube * rho_copper_tube
        mshell = self.L_hot_tube * rho_acrylic_tube

        m_seals = 4 * (m_nozzle + m_large_O + self.total_tubes * m_small_O)

        m_baffles = baffle_area_occlusion_ratio * A_shell * rho_abs * self.total_baffles / self.cold_flow_sections
        
        m_caps = 2 * 0.01 * A_shell * rho_abs # TODO: check end cap mass and inlcude mass of hot and cold section dividers

        return (
            mpipes +
            mshell +
            m_seals +
            m_baffles +
            m_caps
        )


    def set_geometry(self, length, tubes, baffles):
        # tubes and baffle is per element

        self.L_hot_tube = length

        self.total_tubes = 0
        self.total_baffles = 0
        
        for element in self.hot_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                element.tubes = tubes
                element.baffles = baffles

                self.total_tubes += element.tubes
                pattern = element.pattern

        for element in self.cold_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                element.tubes = tubes
                element.baffles = baffles
                self.total_baffles += element.baffles

        self.pitch = pitch_from_tubes(self.total_tubes, pattern)
        if self.pitch < D_outer_tube:
            print("Warning: Pitch is less than the tube diameter")
    
    
    def get_random_geometry_copy(self, constraints = None):

        new_HE = copy.deepcopy(self)

        # set random values for the geometry parameters
        new_HE.set_geometry(0.35,
                            np.random.randint(1, 20),
                            np.random.randint(1, 20))

        return new_HE

