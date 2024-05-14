
from PyQt6.QtWidgets import QWidget, QApplication, QFrame
from PyQt6.QtGui import QPainter, QPen, QBrush, QPolygon
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtWidgets import QLineEdit, QGridLayout, QVBoxLayout, QLabel, QPushButton, QSpinBox

import numpy as np

from constants import *
from utils import draw_zigzag_line, draw_arrow
from fluid_path import Heat_Transfer_Element, Fluid_Path
from heat_exchanger import Heat_Exchanger, Entry_Constriction, Exit_Expansion, U_Bend


class Cycle_Button(QWidget):
    enum_update_signal = pyqtSignal()
    def __init__(self, label_name, enum_class):
        super().__init__()
        self.setGeometry(100, 100, 300, 200)  # Set the position and size of the widget
        
        layout = QVBoxLayout()
        
        self.label_name = label_name
        self.enum_class = enum_class
        self.current_value = list(enum_class)[0]
        
        self.label = QLabel(f"{self.label_name}: {self.current_value.name}")
        layout.addWidget(self.label)
        
        self.button = QPushButton(f"Change {self.label_name}")
        self.button.clicked.connect(self.change_value)
        layout.addWidget(self.button)
        
        self.setLayout(layout)
    
    def change_value(self):
        current_index = list(self.enum_class).index(self.current_value)
        next_index = (current_index + 1) % len(self.enum_class)
        self.current_value = list(self.enum_class)[next_index]
        self.label.setText(f"{self.label_name}: {self.current_value.name}")

        self.enum_update_signal.emit()
    
    def setCurrentValue(self, value):
        self.current_value = value
        self.label.setText(f"{self.label_name}: {self.current_value.name}")


class Heat_Exchanger_Definition(QWidget):
    HE_update_signal = pyqtSignal(Heat_Exchanger)

    def __init__(self):
        super().__init__()

        layout = QGridLayout()

        self.label = QLabel("Manual Heat Exchanger Definition")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 20px;")

        self.hot_stages_label = QLabel("Hot stages:")
        self.hot_stages_input = QSpinBox()
        self.hot_stages_input.setMinimum(1)

        self.cold_stages_label = QLabel("Cold stages:")
        self.cold_stages_input = QSpinBox()
        self.cold_stages_input.setMinimum(1)

        self.input_side = Cycle_Button("Input Side", Side)

        self.baffles_label = QLabel("Number of baffles:")
        self.baffles_input = QSpinBox()
        self.baffles_input.setMinimum(0)

        self.tubes_label = QLabel("Number of tubes:")
        self.tubes_input = QSpinBox()
        self.tubes_input.setMinimum(1)

        layout.addWidget(self.label, 0, 0, 1, 2)
        layout.addWidget(self.hot_stages_label, 1, 0)
        layout.addWidget(self.hot_stages_input, 1, 1)
        layout.addWidget(self.cold_stages_label, 2, 0)
        layout.addWidget(self.cold_stages_input, 2, 1)

        layout.addWidget(self.input_side, 3, 0)

        layout.addWidget(self.baffles_label, 4, 0)
        layout.addWidget(self.baffles_input, 4, 1)
        layout.addWidget(self.tubes_label, 5, 0)
        layout.addWidget(self.tubes_input, 5, 1)


        self.setLayout(layout)

        self.hot_stages_input.valueChanged.connect(self.update_heat_exchanger)
        self.cold_stages_input.valueChanged.connect(self.update_heat_exchanger)
        self.input_side.enum_update_signal.connect(self.update_heat_exchanger)
        self.baffles_input.valueChanged.connect(self.update_heat_exchanger)
        self.tubes_input.valueChanged.connect(self.update_heat_exchanger)

    def load_heat_exchanger(self, heat_exchanger):
        self.hot_stages_input.setValue(heat_exchanger.hot_flow_sections)
        self.cold_stages_input.setValue(heat_exchanger.cold_flow_sections)

        self.input_side.setCurrentValue(heat_exchanger.flow_path_entries_side)

        self.baffles_input.setValue(heat_exchanger.hot_path.elements[1].baffles)
        self.tubes_input.setValue(heat_exchanger.hot_path.elements[1].tubes)

        self.HE_update_signal.emit(heat_exchanger)
    
    def update_heat_exchanger(self):

        hot_stages = self.hot_stages_input.value()
        cold_stages = self.cold_stages_input.value()

        flow_path_entries_side = self.input_side.current_value
        tubes = self.tubes_input.value()
        baffles = self.baffles_input.value()
            
        Hot_path = Fluid_Path(rho_w, mu, cp, k_w)
        Hot_path.add_element(Entry_Constriction())
        Hot_path.add_element(
            Heat_Transfer_Element(tubes, baffles, 
                                Direction.COUNTERFLOW,
                                Pattern.SQUARE)
        )
        Hot_path.add_element(Exit_Expansion())
        for i in range(hot_stages - 1):
            Hot_path.add_element(U_Bend())
            Hot_path.add_element(Entry_Constriction())
            Hot_path.add_element(
                Heat_Transfer_Element(tubes, baffles, 
                                    Direction.COUNTERFLOW,
                                    Pattern.SQUARE)
            )
            Hot_path.add_element(Exit_Expansion())

        Cold_path = Fluid_Path(rho_w, mu, cp, k_w)

        Cold_path.add_element(
            Heat_Transfer_Element(tubes, baffles, 
                                flow_direction=Direction.COUNTERFLOW,
                                tube_pattern = Pattern.SQUARE)
        )
        for i in range(cold_stages - 1):
            Cold_path.add_element(U_Bend())
            Cold_path.add_element(
                Heat_Transfer_Element(tubes, baffles, 
                                    flow_direction=Direction.COFLOW,
                                    tube_pattern = Pattern.SQUARE)
            )

        HEchanger = Heat_Exchanger(Cold_path, Hot_path, 
                                flow_path_entries_side)

        self.HE_update_signal.emit(HEchanger)
        

class Heat_Exchanger_Diagram(QWidget):
    def __init__(self, width, height):
        super().__init__()
        self.setWindowTitle('Heat Exchanger Diagram')

        self.width = width
        self.height = height

        self.setGeometry(0, 0, self.width, self.height)

        self.cold_inlet_box = QLineEdit()
        self.hot_inlet_box = QLineEdit()

        self.cold_outlet_box = QLineEdit()
        self.hot_outlet_box = QLineEdit()

        self.cold_outlet_box.setReadOnly(True)
        self.hot_outlet_box.setReadOnly(True)

        self.cold_inlet_box.editingFinished.connect(self.recompute)
        self.hot_inlet_box.editingFinished.connect(self.recompute)
    
        self.cold_inlet_box.setPlaceholderText("T1in")
        self.hot_inlet_box.setPlaceholderText("T2in")
        self.cold_outlet_box.setPlaceholderText("T1out")
        self.hot_outlet_box.setPlaceholderText("T2out")

        self.mdot_hot_box = QLineEdit()
        self.mdot_cold_box = QLineEdit()
        self.Qdot_box = QLineEdit()
        self.effectiveness_box = QLineEdit()

        self.mdot_hot_box.setReadOnly(True)
        self.mdot_cold_box.setReadOnly(True)
        self.Qdot_box.setReadOnly(True)
        self.effectiveness_box.setReadOnly(True)

        self.mdot_hot_box.setPlaceholderText("mdot_hot")
        self.mdot_cold_box.setPlaceholderText("mdot_cold")
        self.Qdot_box.setPlaceholderText("Qdot")
        self.effectiveness_box.setPlaceholderText("Effectiveness")

        self.mdot_hot_label = QLabel("Hot mass flow rate:")
        self.mdot_cold_label = QLabel("Cold mass flow rate:")
        self.Qdot_label = QLabel("Heat transfer rate:")
        self.effectiveness_label = QLabel("Effectiveness:")


        # Set up the layout
        layout = QGridLayout()
        layout.addWidget(self.cold_inlet_box, 0, 0)
        layout.addWidget(self.hot_inlet_box, 0, 1)
        layout.addWidget(self.cold_outlet_box, 1, 0)
        layout.addWidget(self.hot_outlet_box, 1, 1)

        layout.addWidget(self.mdot_hot_label, 2, 0)
        layout.addWidget(self.mdot_hot_box, 2, 1)
        layout.addWidget(self.mdot_cold_label, 3, 0)
        layout.addWidget(self.mdot_cold_box, 3, 1)
        layout.addWidget(self.Qdot_label, 4, 0)
        layout.addWidget(self.Qdot_box, 4, 1)
        layout.addWidget(self.effectiveness_label, 5, 0)
        layout.addWidget(self.effectiveness_box, 5, 1)

        # move labels and outputs down
        for i in range(2, 6):
            layout.setVerticalSpacing(50)

        self.setLayout(layout)
        
    def set_heat_exchanger(self, heat_exchanger):
        self.heat_exchanger = heat_exchanger
        self.recompute()
        self.update()
    
    def recompute(self):

        print("Recomputing")

        try:
            T1in = float(self.cold_inlet_box.text())
            T2in = float(self.hot_inlet_box.text())
        except ValueError:
            return

        self.heat_exchanger.set_conditions([T1in, T2in])
        res = self.heat_exchanger.compute_effectiveness(
            method='LMTD'
        )

        if not res:
            print("failed to compute effectiveness")
            return

        T1out, T2out = self.heat_exchanger.Tout
        mdot_cold, mdot_hot = self.heat_exchanger.mdot
        Qdot = self.heat_exchanger.Qdot
        effectiveness = self.heat_exchanger.effectiveness

        self.cold_outlet_box.setText(
            str(np.round(T1out,2))
            )
        self.hot_outlet_box.setText(
            str(np.round(T2out,2))
            )
        
        self.mdot_cold_box.setText(
            str(np.round(mdot_cold,2))
            )
        self.mdot_hot_box.setText(
            str(np.round(mdot_hot,2))
            )
        self.Qdot_box.setText(
            str(np.round(Qdot,2))
            )
        self.effectiveness_box.setText(
            str(np.round(effectiveness,2))
            )
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        #painter.setRenderHint(QPainter.RenderHint.Antialiasing)

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


        x_per_zigzag = int(200 * self.heat_exchanger.hot_flow_sections / self.heat_exchanger.total_tubes)

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
        
        # set positions of inlets and outlets
        self.cold_inlet_box.setGeometry(int(x_cold_in * scale_x), int(25 * scale_y), 50, 50)
        self.hot_inlet_box.setGeometry(int(x_hot_in * scale_x), int(25 * scale_y), 50, 50)
        self.cold_outlet_box.setGeometry(int(x_cold_out * scale_x), int(325 * scale_y), 50, 50)
        self.hot_outlet_box.setGeometry(int(x_hot_out * scale_x), int(325 * scale_y), 50, 50)

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
            painter.drawLine(int(175 * scale_x), start_point1.y(), int(200 * scale_x), start_point1.y())
            painter.drawLine(int(600 * scale_x), start_point1.y(), int(625 * scale_x), start_point1.y())

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
            painter.drawLine(int(125 * scale_x), start_point1.y(), int(200 * scale_x), start_point1.y())
            painter.drawLine(int(600 * scale_x), start_point1.y(), int(675 * scale_x), start_point1.y())

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

