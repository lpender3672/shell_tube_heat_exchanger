
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtGui import QPainter, QPen, QBrush, QPolygon
from PyQt6.QtCore import Qt, QPoint

import numpy as np

from constants import *
from utils import draw_zigzag_line, draw_arrow
from fluid_path import Heat_Transfer_Element


class Heat_Exchanger_Diagram(QWidget):
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

        # draw baffles
        # just going to draw them the same for each heat element
        num_baffles = 0
        for e in self.heat_exchanger.hot_path.elements:
            if isinstance(e, Heat_Transfer_Element):
                num_baffles = e.baffles
                break

        painter.setPen(QPen(Qt.GlobalColor.darkGray, 4))
        for i in range(num_baffles):
            x_coord = (150 + 500 * (i + 1) / (num_baffles + 1)) * scale_x
            painter.drawLine(int(x_coord), int(100 * scale_y), int(x_coord), int(300 * scale_y))


        x_per_zigzag = 40

        hot_channel_width = 200 / self.heat_exchanger.hot_flow_sections
        cold_channel_width = 200 / self.heat_exchanger.cold_flow_sections

        if (self.heat_exchanger.hot_flow_sections % self.heat_exchanger.cold_flow_sections == 0 or 
            self.heat_exchanger.cold_flow_sections % self.heat_exchanger.hot_flow_sections ==0):
            v_offset = 0.1 * cold_channel_width
        else:
            v_offset = 0

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
            x_hot_out = x_hot_ins[x_cold_out != x_cold_ins[0]]
        else:
            x_hot_out = x_hot_ins[x_cold_out == x_cold_ins[0]]
        draw_arrow(painter, QPoint(int(x_hot_out * scale_x), int(300 * scale_y)), QPoint(int(x_hot_out * scale_x), int(350 * scale_y)), 10, Qt.GlobalColor.red, 2)
        

        # Draw zigzag lines using the draw_zigzag_line function

        for i in range(1, self.heat_exchanger.cold_flow_sections):
            j = x_cold_in == x_cold_ins[i % 2]
            sep_y_coord = (100 + i *cold_channel_width) * scale_y
            sep_x1_coord = (150 + 50 * j) * scale_x
            sep_x2_coord = (650 - 50 * (not j)) * scale_x
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.drawLine(int(sep_x1_coord), int(sep_y_coord), int(sep_x2_coord), int(sep_y_coord))

        for i in range(self.heat_exchanger.cold_flow_sections):
            cold_y_coord = (100 + cold_channel_width//2 + i * cold_channel_width + v_offset) * scale_y
            start_point1 = QPoint(int(200 * scale_x), int(cold_y_coord))
            end_point1 = QPoint(int(600 * scale_x), int(cold_y_coord))

            zigzag_width1 = 10
            num_segments1 = (end_point1.x() - start_point1.x()) // x_per_zigzag
            color1 = Qt.GlobalColor.blue
            width1 = 2

            draw_zigzag_line(painter, start_point1, end_point1, zigzag_width1, num_segments1, color1, width1)
            painter.drawLine(int(175 * scale_x), start_point1.y(), 200, start_point1.y())
            painter.drawLine(int(600 * scale_x), start_point1.y(), 625, start_point1.y())

        for i in range(self.heat_exchanger.hot_flow_sections - 1):
            sep_y_coord = (100 + (i+1) * hot_channel_width) * scale_y
            sep_x1_coord = (x_hot_ins[i % 2] - 25) * scale_x
            sep_x2_coord = (x_hot_ins[i % 2] + 25) * scale_x
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.drawLine(int(sep_x1_coord), int(sep_y_coord), int(sep_x2_coord), int(sep_y_coord))

        for i in range(self.heat_exchanger.hot_flow_sections):
            y_coord = (100 + hot_channel_width//2 + i * hot_channel_width - v_offset) * scale_y
            start_point1 = QPoint(int(200 * scale_x), int(y_coord))
            end_point1 = QPoint(int(600 * scale_x), int(y_coord))
 
            zigzag_width1 = 10
            num_segments1 = (end_point1.x() - start_point1.x()) // x_per_zigzag
            color1 = Qt.GlobalColor.red
            width1 = 2

            draw_zigzag_line(painter, start_point1, end_point1, zigzag_width1, num_segments1, color1, width1)
            painter.drawLine(int(125 * scale_x), start_point1.y(), 200, start_point1.y())
            painter.drawLine(int(600 * scale_x), start_point1.y(), 675, start_point1.y())

        # connect the zigzag lines to the hot and cold inlets

        in_hot_y_connect = (100 + hot_channel_width//2 - v_offset) * scale_y
        out_hot_y_connect = (300 - hot_channel_width//2 - v_offset) * scale_y
        painter.setPen(QPen(Qt.GlobalColor.red, 2))
        painter.drawLine(int(x_hot_in * scale_x), int(100 * scale_y), int(x_hot_in * scale_x), int(in_hot_y_connect))
        painter.drawLine(int(x_hot_out * scale_x), int(300 * scale_y), int(x_hot_out * scale_x), int(out_hot_y_connect))

        for i in range(self.heat_exchanger.hot_flow_sections - 1):
            x_connect = x_hot_ins[x_hot_in == x_hot_ins[i % 2]]

            y_con_1 = (100 + hot_channel_width//2 + i * hot_channel_width - v_offset) * scale_y
            y_con_2 = (100 + hot_channel_width//2 + (i+1) * hot_channel_width - v_offset) * scale_y

            painter.setPen(QPen(Qt.GlobalColor.red, 2))
            painter.drawLine(int(x_connect * scale_x), int(y_con_1), int(x_connect * scale_x), int(y_con_2))

        in_cold_y_connect = (100 + cold_channel_width//2 + v_offset) * scale_y
        out_cold_y_connect = (300 - cold_channel_width//2 + v_offset) * scale_y
        painter.setPen(QPen(Qt.GlobalColor.blue, 2))
        painter.drawLine(int(x_cold_in * scale_x), int(100 * scale_y), int(x_cold_in * scale_x), int(in_cold_y_connect))
        painter.drawLine(int(x_cold_out * scale_x), int(300 * scale_y), int(x_cold_out * scale_x), int(out_cold_y_connect))

        for i in range(self.heat_exchanger.cold_flow_sections - 1):
            x_connect = x_cold_ins[x_cold_in == x_cold_ins[i % 2]]

            y_con_1 = (100 + cold_channel_width//2 + i * cold_channel_width + v_offset) * scale_y
            y_con_2 = (100 + cold_channel_width//2 + (i+1) * cold_channel_width + v_offset) * scale_y

            painter.setPen(QPen(Qt.GlobalColor.blue, 2))
            painter.drawLine(int(x_connect * scale_x), int(y_con_1), int(x_connect * scale_x), int(y_con_2))
        
