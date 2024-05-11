
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtGui import QPainter, QPen, QBrush, QPolygon
from PyQt6.QtCore import Qt, QPoint

import numpy as np
import matplotlib.pyplot as plt

from constants import *
from utils import draw_zigzag_line, draw_arrow
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

def heat_solve_iteration(T1in, T1out, T2in, T2out):
    pass # ??



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
        self.hot_pressure_factor = 2
        self.cold_pressure_factor = 2


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
                self.Re_shell = v_shell * rho_w * effective_d_shell / mu

                self.DP_cold += 4 * a_factor * self.Re_shell ** (-0.15) * element.tubes * rho_w * v_shell ** 2
            
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
        
        F = 1 # TODO: find out what this is

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

            Nu_i = 0.023 * Re_hot ** 0.8 * Pr **0.3
            Nu_o = c_factor * self.Re_shell **0.6 * Pr **0.3

            h_i = Nu_i * k_w / D_inner_tube
            h_o = Nu_o * k_w / D_outer_tube

            A_i = np.pi * D_inner_tube * self.L_hot_tube
            A_o = np.pi * D_outer_tube * self.L_hot_tube
            one_over_H += 1/h_i + A_i * np.log(D_outer_tube / D_inner_tube) / (2*np.pi * k_tube * self.L_hot_tube) + 1 / h_o * (A_i / A_o)

        H = 1 / one_over_H

        print(H)

        # TODO: solve thermal equations


    def is_geometrically_feasible(self):
        # performs collision detection to see if the heat exchanger is geometrically feasible

        # check square or triangle design packing of the N_tubes in a shell for the given pitch
        # also check length of tubes are within individual and total limits

        pass


class HeatExchangerDiagram(QWidget):
    def __init__(self, width, height):
        super().__init__()
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Heat Exchanger Diagram')

        self.width = width
        self.height = height

        self.setMinimumWidth(width)
        self.setMinimumHeight(height)
    
    def set_heat_exchanger(self, heat_exchanger):
        self.heat_exchanger = heat_exchanger

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate scaling factors based on width and height
        scale_x = self.width / 800
        scale_y = self.height / 600

        # Draw boxes
        painter.setBrush(QBrush(Qt.GlobalColor.white))
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawRect(int(100 * scale_x), int(100 * scale_y), int(600 * scale_x), int(200 * scale_y))

        painter.fillRect(int(150 * scale_x), int(100 * scale_y), int(500 * scale_x), int(200 * scale_y),
                         QBrush(Qt.GlobalColor.blue, Qt.BrushStyle.Dense4Pattern))
        
        painter.fillRect(int(100 * scale_x), int(100 * scale_y), int(50 * scale_x), int(200 * scale_y),
                         QBrush(Qt.GlobalColor.red, Qt.BrushStyle.Dense4Pattern))
        painter.fillRect(int(650 * scale_x), int(100 * scale_y), int(50 * scale_x), int(200 * scale_y),
                         QBrush(Qt.GlobalColor.red, Qt.BrushStyle.Dense4Pattern))
        
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawLine(int(150 * scale_x), int(100 * scale_y), int(150 * scale_x), int(300 * scale_y))
        painter.drawLine(int(650 * scale_x), int(100 * scale_y), int(650 * scale_x), int(300 * scale_y))


        x_per_zigzag = 30

        # hot
        x_hot_ins = [125, 675]
        x_hot_in = x_hot_ins[0]
        draw_arrow(painter, QPoint(int(x_hot_in * scale_x), int(50 * scale_y)), QPoint(int(x_hot_in * scale_x), int(100 * scale_y)), 10, Qt.GlobalColor.red, 2)

        # cold
        x_cold_ins = [175, 625]
        if self.heat_exchanger.flow_path_entries_side == Side.SAME:
            x_cold_in = x_cold_ins[0]
        else:
            x_cold_in = x_cold_ins[1]
        draw_arrow(painter, QPoint(int(x_cold_in * scale_x), int(50 * scale_y)), QPoint(int(x_cold_in * scale_x), int(100 * scale_y)), 10, Qt.GlobalColor.blue, 2)
        
        if self.heat_exchanger.cold_flow_sections % 2 == 0:
            x_cold_out = x_cold_in

        else:
            x_cold_out = x_cold_ins[x_cold_in == x_cold_ins[0]]
        draw_arrow(painter, QPoint(int(x_cold_out * scale_x), int(300 * scale_y)), QPoint(int(x_cold_out * scale_x), int(350 * scale_y)), 10, Qt.GlobalColor.blue, 2)

        if self.heat_exchanger.flow_path_exits_side == Side.SAME:
            x_hot_out = x_cold_out
        else:
            x_hot_out = x_hot_ins[x_cold_out == x_cold_ins[0]]
        draw_arrow(painter, QPoint(int(x_hot_out * scale_x), int(300 * scale_y)), QPoint(int(x_hot_out * scale_x), int(350 * scale_y)), 10, Qt.GlobalColor.red, 2)

        

        # Draw zigzag lines using the draw_zigzag_line function

        cold_channel_width = 200 / self.heat_exchanger.cold_flow_sections
        for i in range(1, self.heat_exchanger.cold_flow_sections):
            sep_y_coord = (100 + i *cold_channel_width) * scale_y
            sep_x1_coord = (150 + 50 * (i % 2)) * scale_x
            sep_x2_coord = (650 - 50 * ((i+1) % 2)) * scale_x
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.drawLine(int(sep_x1_coord), int(sep_y_coord), int(sep_x2_coord), int(sep_y_coord))

        for i in range(self.heat_exchanger.cold_flow_sections):
            cold_y_coord = (100 + cold_channel_width//2 + i * cold_channel_width) * scale_y
            start_point1 = QPoint(int(175 * scale_x), int(cold_y_coord))
            end_point1 = QPoint(int(625 * scale_x), int(cold_y_coord))

            zigzag_width1 = int(30 / self.heat_exchanger.cold_flow_sections)
            num_segments1 = (end_point1.x() - start_point1.x()) // x_per_zigzag
            color1 = Qt.GlobalColor.blue
            width1 = 2

            draw_zigzag_line(painter, start_point1, end_point1, zigzag_width1, num_segments1, color1, width1)


        hot_channel_width = 200 / self.heat_exchanger.hot_flow_sections
        for i in range(self.heat_exchanger.hot_flow_sections):
            y_coord = (100 + hot_channel_width//2 + i * hot_channel_width) * scale_y
            start_point1 = QPoint(int((125) * scale_x), int(y_coord))
            end_point1 = QPoint(int(675 * scale_x), int(y_coord))

            zigzag_width1 = int(30 / self.heat_exchanger.hot_flow_sections)
            num_segments1 = (end_point1.x() - start_point1.x()) // x_per_zigzag
            color1 = Qt.GlobalColor.red
            width1 = 2

            draw_zigzag_line(painter, start_point1, end_point1, zigzag_width1, num_segments1, color1, width1)

        # connect the zigzag lines to the hot and cold inlets

        in_hot_y_connect = (100 + hot_channel_width//2) * scale_y
        out_hot_y_connect = (300 - hot_channel_width//2) * scale_y
        painter.setPen(QPen(Qt.GlobalColor.red, 2))
        painter.drawLine(int(x_hot_in * scale_x), int(100 * scale_y), int(x_hot_in * scale_x), int(in_hot_y_connect))
        painter.drawLine(int(x_hot_out * scale_x), int(300 * scale_y), int(x_hot_out * scale_x), int(out_hot_y_connect))

        for i in range(self.heat_exchanger.hot_flow_sections - 1):
            x_connect = x_hot_ins[x_hot_in == x_hot_ins[i % 2]]

            y_con_1 = (100 + hot_channel_width//2 + i * hot_channel_width) * scale_y
            y_con_2 = (100 + hot_channel_width//2 + (i+1) * hot_channel_width) * scale_y

            painter.setPen(QPen(Qt.GlobalColor.red, 2))
            painter.drawLine(int(x_connect * scale_x), int(y_con_1), int(x_connect * scale_x), int(y_con_2))

        in_cold_y_connect = (100 + cold_channel_width//2) * scale_y
        out_cold_y_connect = (300 - cold_channel_width//2) * scale_y
        painter.setPen(QPen(Qt.GlobalColor.blue, 2))
        painter.drawLine(int(x_cold_in * scale_x), int(100 * scale_y), int(x_cold_in * scale_x), int(in_cold_y_connect))
        painter.drawLine(int(x_cold_out * scale_x), int(300 * scale_y), int(x_cold_out * scale_x), int(out_cold_y_connect))

        for i in range(self.heat_exchanger.cold_flow_sections - 1):
            x_connect = x_cold_ins[x_cold_in == x_cold_ins[i % 2]]

            y_con_1 = (100 + cold_channel_width//2 + i * cold_channel_width) * scale_y
            y_con_2 = (100 + cold_channel_width//2 + (i+1) * cold_channel_width) * scale_y

            painter.setPen(QPen(Qt.GlobalColor.blue, 2))
            painter.drawLine(int(x_connect * scale_x), int(y_con_1), int(x_connect * scale_x), int(y_con_2))







        


