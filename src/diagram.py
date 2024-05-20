
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QBrush
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtWidgets import QLineEdit, QGridLayout, QVBoxLayout, QLabel, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem

import numpy as np
import logging

from constants import *
from utils import draw_zigzag_line, draw_arrow
from fluid_path import Heat_Transfer_Element, Fluid_Path
from heat_exchanger import Heat_Exchanger, build_heat_exchanger


class Cycle_Button(QWidget):
    enum_update_signal = pyqtSignal()
    def __init__(self, parent, label_name, enum_class):
        super().__init__(parent)
        self.setGeometry(100, 100, 300, 200)  # Set the position and size of the widget
        
        layout = QVBoxLayout()
        
        self.label_name = label_name
        self.enum_class = enum_class
        self.current_value = list(enum_class)[0]
                
        self.button = QPushButton(f"{self.current_value.name}")
        self.button.clicked.connect(self.change_value)
        layout.addWidget(self.button)
        
        self.setLayout(layout)
    
    def change_value(self):
        current_index = list(self.enum_class).index(self.current_value)
        next_index = (current_index + 1) % len(self.enum_class)
        self.current_value = list(self.enum_class)[next_index]
        self.button.setText(f"{self.current_value.name}")

        self.enum_update_signal.emit()
    
    def setCurrentValue(self, value):
        self.current_value = value
        self.button.setText(f"{self.current_value.name}")


class Heat_Exchanger_Definition(QWidget):
    HE_update_signal = pyqtSignal(Heat_Exchanger)

    def __init__(self, parent):
        super().__init__(parent)

        layout = QGridLayout()

        self.label = QLabel("Manual Heat Exchanger Definition")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 20px;")


        self.input_side = Cycle_Button(self, "Input Side", Side)

        self.stage_table = QTableWidget()
        
        self.reset_table()

        self.stage_table.setCellWidget(0, 0, QSpinBox(self.stage_table, minimum=1))
        self.stage_table.setCellWidget(0, 1, Cycle_Button(self.stage_table, "Tube Pattern", Pattern))
        self.stage_table.setCellWidget(0, 2, QSpinBox(self.stage_table, minimum=1))
        self.stage_table.setCellWidget(0, 3, Cycle_Button(self.stage_table, "Tube Pattern", Pattern))

        self.stage_table.cellWidget(0, 0).valueChanged.connect(self.update_heat_exchanger)
        self.stage_table.cellWidget(0, 1).enum_update_signal.connect(self.update_heat_exchanger)
        self.stage_table.cellWidget(0, 2).valueChanged.connect(self.update_heat_exchanger)
        self.stage_table.cellWidget(0, 3).enum_update_signal.connect(self.update_heat_exchanger)

        add_cold_pass = QTableWidgetItem("Add\nCold Pass")
        add_cold_pass.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)

        add_cold_pass.setFlags(add_cold_pass.flags() & ~ Qt.ItemFlag.ItemIsEditable)
        add_hot_pass = QTableWidgetItem("Add\nHot Pass")
        add_hot_pass.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
        add_hot_pass.setFlags(add_hot_pass.flags() & ~ Qt.ItemFlag.ItemIsEditable)

        self.stage_table.setItem(1, 0, add_cold_pass)
        self.stage_table.setSpan(1, 0, 1, 2)
        self.stage_table.setItem(1, 2, add_hot_pass)
        self.stage_table.setSpan(1, 2, 1, 2)

        self.stage_table.cellDoubleClicked.connect(self.on_stage_cell_clicked)
        self.stage_table.verticalHeader().sectionDoubleClicked.connect(self.on_vertical_header_clicked)

        self.stage_table.resizeColumnsToContents()
        self.stage_table.resizeRowsToContents()

        self.length_label = QLabel("Length (m):")
        self.length_input = QLineEdit()

        layout.addWidget(self.label, 0, 0, 1, 2)
        layout.addWidget(self.stage_table, 1, 0, 5, 2)
        self.stage_table.setMinimumWidth(400)
        self.stage_table.setMinimumHeight(300)


        layout.addWidget(self.input_side, 6, 0, 1, 2)
        layout.addWidget(self.length_label, 7, 0)
        layout.addWidget(self.length_input, 7, 1)

        self.setLayout(layout)

        self.loading = False
        self.stage_table.itemChanged.connect(self.update_heat_exchanger)
        self.input_side.enum_update_signal.connect(self.update_heat_exchanger)
        self.length_input.editingFinished.connect(self.update_heat_exchanger)

    def on_stage_cell_clicked(self, row, column):
        
        item = self.stage_table.item(row, column)
        if item is None or item.text() == "":
            return

        if row == self.stage_table.rowCount() - 1:
            self.stage_table.setRowCount(self.stage_table.rowCount() + 1)

            self.stage_table.setVerticalHeaderItem(row, QTableWidgetItem(f"Pass {row + 1}"))
            self.stage_table.setVerticalHeaderItem(row + 1, QTableWidgetItem("Add Pass"))
    
        if column == 0:
            # add cold pass
            self.cold_passes += 1
            self.stage_table.setSpan(row, 0, 1, 1)

            self.stage_table.setCellWidget(row, 0, QSpinBox(self.stage_table, minimum=1))
            self.stage_table.setCellWidget(row, 1, Cycle_Button(self.stage_table, "Tube Pattern", Pattern))

            self.stage_table.cellWidget(row, 0).valueChanged.connect(self.update_heat_exchanger)
            self.stage_table.cellWidget(row, 1).enum_update_signal.connect(self.update_heat_exchanger)

            add_cold_pass = QTableWidgetItem("Add\nCold Pass")
            add_cold_pass.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)

            self.stage_table.setItem(row + 1, 0, add_cold_pass)
            self.stage_table.setSpan(row + 1, 0, 1, 2)
        
        elif column == 2:
            # add hot pass
            self.hot_passes += 1
            self.stage_table.setSpan(row, 2, 1, 1)

            self.stage_table.setCellWidget(row, 2, QSpinBox(self.stage_table, minimum=1))
            self.stage_table.setCellWidget(row, 3, Cycle_Button(self.stage_table, "Tube Pattern", Pattern))

            self.stage_table.cellWidget(row, 2).valueChanged.connect(self.update_heat_exchanger)
            self.stage_table.cellWidget(row, 3).enum_update_signal.connect(self.update_heat_exchanger)

            add_hot_pass = QTableWidgetItem("Add\nHot Pass")
            add_hot_pass.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)

            self.stage_table.setItem(row + 1, 2, add_hot_pass)
            self.stage_table.setSpan(row + 1, 2, 1, 2)
        
        self.stage_table.resizeColumnsToContents()
        self.stage_table.resizeRowsToContents()

        #self.update_heat_exchanger()
    
    def on_vertical_header_clicked(self, row):
        if row == 0:
            # cant remove the first row
            return
        if row == self.stage_table.rowCount() - 1:
            # cant remove the last row
            return
        
        if self.stage_table.cellWidget(row, 0) is None:
            self.hot_passes -= 1

            if (self.stage_table.item(row, 0) is not None and
                self.stage_table.item(row, 0).text() == "Add\nCold Pass"):

                add_cold_pass = QTableWidgetItem("Add\nCold Pass")
                add_cold_pass.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
                self.stage_table.setItem(row + 1, 0, QTableWidgetItem(add_cold_pass))
                self.stage_table.setSpan(row + 1, 0, 1, 2)
        
        elif self.stage_table.cellWidget(row, 2) is None:
            self.cold_passes -= 1

            if (self.stage_table.item(row, 2) is not None and
                self.stage_table.item(row, 2).text() == "Add\nHot Pass"):

                add_hot_pass = QTableWidgetItem("Add\nHot Pass")
                add_hot_pass.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
                self.stage_table.setItem(row + 1, 2, add_hot_pass)
                self.stage_table.setSpan(row + 1, 2, 1, 2)
        
        else:
            self.cold_passes -= 1
            self.hot_passes -= 1

        self.stage_table.removeRow(row)

        # update rows below to reflect new row numbers
        for i in range(row, self.stage_table.rowCount()):
            self.stage_table.setVerticalHeaderItem(i - 1, QTableWidgetItem(f"Pass {i}"))

        self.stage_table.resizeColumnsToContents()
        self.stage_table.resizeRowsToContents()

        self.update_heat_exchanger()

    def reset_table(self):

        self.stage_table.setRowCount(2)
        self.stage_table.setColumnCount(4) # tubes, pattern,  baffles, pattern
        self.stage_table.setHorizontalHeaderLabels(["Baffles", "Pattern", "Tubes", "Pattern"])
        self.stage_table.setVerticalHeaderLabels(["Pass 1", ""])

        self.stage_table.setCellWidget(0, 0, QSpinBox(self.stage_table, minimum=1))
        self.stage_table.setCellWidget(0, 1, Cycle_Button(self.stage_table, "Tube Pattern", Pattern))
        self.stage_table.setCellWidget(0, 2, QSpinBox(self.stage_table, minimum=1))
        self.stage_table.setCellWidget(0, 3, Cycle_Button(self.stage_table, "Tube Pattern", Pattern))

        self.stage_table.cellWidget(0, 0).valueChanged.connect(self.update_heat_exchanger)
        self.stage_table.cellWidget(0, 1).enum_update_signal.connect(self.update_heat_exchanger)
        self.stage_table.cellWidget(0, 2).valueChanged.connect(self.update_heat_exchanger)
        self.stage_table.cellWidget(0, 3).enum_update_signal.connect(self.update_heat_exchanger)

        self.hot_passes = 1
        self.cold_passes = 1

    def load_heat_exchanger(self, heat_exchanger):
        self.loading = True

        self.reset_table()

        for i in range(1, heat_exchanger.cold_flow_sections):
            self.on_stage_cell_clicked(i, 0)
        i = 0
        for element in heat_exchanger.cold_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                self.stage_table.cellWidget(i, 0).setValue(
                    int(element.baffles))
                self.stage_table.cellWidget(i, 1).setCurrentValue(
                    element.pattern)
                i += 1

        for i in range(1, heat_exchanger.hot_flow_sections):
            self.on_stage_cell_clicked(i, 2)
        
        i = 0
        for element in heat_exchanger.hot_path.elements:
            if isinstance(element, Heat_Transfer_Element):
                self.stage_table.cellWidget(i, 2).setValue(
                    int(element.tubes))
                self.stage_table.cellWidget(i, 3).setCurrentValue(element.pattern)
                i += 1

        self.input_side.setCurrentValue(heat_exchanger.flow_path_entries_side)
        self.length_input.setText(str(heat_exchanger.L_hot_tube))

        self.HE_update_signal.emit(heat_exchanger)

        self.loading = False

    def update_heat_exchanger(self):
        
        # if update detected while loading, ignore
        # this is not a user initiated update
        if self.loading:
            return

        baffles_per_stage = []
        cold_tube_pattern = []
        tubes_per_stage = []
        hot_tube_pattern = []

        for i in range(self.cold_passes):
            w = self.stage_table.cellWidget(i, 0)
            baffles_per_stage.append(w.value())
            wp = self.stage_table.cellWidget(i, 1)
            cold_tube_pattern.append(wp.current_value)
        
        for i in range(self.hot_passes):
            w = self.stage_table.cellWidget(i, 2)
            tubes_per_stage.append(w.value())
            wp = self.stage_table.cellWidget(i, 3)
            hot_tube_pattern.append(wp.current_value)

        flow_path_entries_side = self.input_side.current_value
        length = self.length_input.text()
        
        try:
            assert length != ""
            length = float(length)
        except AssertionError:
            return 
        except ValueError:
            logging.warning(f"Unable to convert {length} to a float")
            return
        
        heat_exchanger = build_heat_exchanger(
            tubes_per_stage, baffles_per_stage, length, flow_path_entries_side, hot_tube_pattern, cold_tube_pattern
        )

        self.HE_update_signal.emit(heat_exchanger)

        return heat_exchanger


class Heat_Exchanger_Diagram(QWidget):
    def __init__(self, parent, width, height):
        super().__init__(parent)
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
        self.cold_inlet_box.setStyleSheet("color: blue;")
        self.hot_inlet_box.setPlaceholderText("T2in")
        self.hot_inlet_box.setStyleSheet("color: red;")
        self.cold_outlet_box.setPlaceholderText("T1out")
        self.cold_outlet_box.setStyleSheet("color: blue;")
        self.hot_outlet_box.setPlaceholderText("T2out")
        self.hot_outlet_box.setStyleSheet("color: red;")

        self.mdot_hot_box = QLineEdit()
        self.mdot_cold_box = QLineEdit()
        self.Qdot_box = QLineEdit()
        self.effectiveness_box = QLineEdit()
        self.mass_box = QLineEdit()
        self.total_tube_length_box = QLineEdit()

        self.mdot_hot_box.setReadOnly(True)
        self.mdot_cold_box.setReadOnly(True)
        self.Qdot_box.setReadOnly(True)
        self.effectiveness_box.setReadOnly(True)
        self.mass_box.setReadOnly(True)
        self.total_tube_length_box.setReadOnly(True)

        self.mdot_hot_box.setPlaceholderText("mdot_hot")
        self.mdot_cold_box.setPlaceholderText("mdot_cold")
        self.Qdot_box.setPlaceholderText("Qdot")
        self.effectiveness_box.setPlaceholderText("Effectiveness")
        self.mass_box.setPlaceholderText("mass")
        self.total_tube_length_box.setPlaceholderText("Total Tube Length")

        self.mdot_hot_label = QLabel("Hot mass flow rate:")
        self.mdot_cold_label = QLabel("Cold mass flow rate:")
        self.Qdot_label = QLabel("Heat transfer rate:")
        self.effectiveness_label = QLabel("Effectiveness:")
        self.mass_label = QLabel("HE Mass (kg)")
        self.total_tube_length_label = QLabel("Total Tube Length (m)")

        # Set up the layout
        layout = QGridLayout()
        layout.addWidget(self.cold_inlet_box, 0, 0)
        layout.addWidget(self.hot_inlet_box, 0, 1)
        layout.addWidget(self.cold_outlet_box, 1, 0)
        layout.addWidget(self.hot_outlet_box, 1, 1)


        for i,w in enumerate(self.get_output_widgets()):
            layout.addWidget(w, i//2+2, i%2, 1, 2)

        self.setLayout(layout)
    
    def get_output_widgets(self):

        return [
            self.mdot_hot_label,
            self.mdot_hot_box,
            self.mdot_cold_label,
            self.mdot_cold_box,
            self.Qdot_label,
            self.Qdot_box,
            self.effectiveness_label,
            self.effectiveness_box,
            self.mass_label,
            self.mass_box,
            self.total_tube_length_label,
            self.total_tube_length_box
        ]
        
    def set_heat_exchanger(self, heat_exchanger):
        self.heat_exchanger = heat_exchanger
        self.recompute()
        self.update()
    
    def set_conditions(self, conditions):
        self.cold_inlet_box.setText(str(conditions[0]))
        self.hot_inlet_box.setText(str(conditions[1]))
        self.recompute()
        self.update()
    
    def recompute(self):

        try:
            T1_text = self.cold_inlet_box.text()
            T2_text = self.hot_inlet_box.text()
            assert T1_text != "" and T2_text != ""
            T1in = float(T1_text)
            T2in = float(T2_text)
        except AssertionError:
            return # as nothing has been entered
        except ValueError:
            logging.warning(f"Failed to convert {T1_text} or {T2_text} to a float")
            return

        self.heat_exchanger.set_conditions([T1in, T2in])
        res = self.heat_exchanger.compute_effectiveness(
            method='LMTD', optimiser = 'fsolve'
        )

        if not res:
            #logging.warning("Failed to compute effectiveness")

            self.cold_outlet_box.setText("N/A")
            self.hot_outlet_box.setText("N/A")
            self.mdot_cold_box.setText("N/A")
            self.mdot_hot_box.setText("N/A")
            self.Qdot_box.setText("N/A")
            self.effectiveness_box.setText("N/A")
            return

        T1out, T2out = self.heat_exchanger.Tout
        mdot_cold, mdot_hot = self.heat_exchanger.mdot
        Qdot = self.heat_exchanger.Qdot
        effectiveness = self.heat_exchanger.effectiveness

        mass = self.heat_exchanger.calc_mass()
        tube_length = self.heat_exchanger.total_tubes * self.heat_exchanger.L_hot_tube
        print(self.heat_exchanger.total_tubes, tube_length)

        dpoints = 4

        self.cold_outlet_box.setText(
            str(np.round(T1out,1))
            )
        self.hot_outlet_box.setText(
            str(np.round(T2out,1))
            )
        
        self.mdot_cold_box.setText(
            str(np.round(mdot_cold,dpoints))
            )
        self.mdot_hot_box.setText(
            str(np.round(mdot_hot,dpoints))
            )
        self.Qdot_box.setText(
            str(np.round(Qdot,dpoints))
            )
        self.effectiveness_box.setText(
            str(np.round(effectiveness,dpoints))
            )

        if mass > max_HE_mass:
            self.mass_box.setStyleSheet("color: red;")
        else:
            self.mass_box.setStyleSheet("color: black;")
        self.mass_box.setText(
            str(np.round(mass,dpoints))
            )
        if tube_length > max_total_tube_length:
            self.total_tube_length_box.setStyleSheet("color: red;")
        else:
            self.total_tube_length_box.setStyleSheet("color: black;")
        self.total_tube_length_box.setText(
            str(np.round(tube_length,dpoints))
            )
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        #painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not hasattr(self, 'heat_exchanger'):
            return

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

        
        num_baffles = []
        for e in self.heat_exchanger.cold_path.elements:
            if isinstance(e, Heat_Transfer_Element):
                num_baffles.append(e.baffles)
        
        num_tubes = []
        for e in self.heat_exchanger.hot_path.elements:
            if isinstance(e, Heat_Transfer_Element):
                num_tubes.append(e.tubes)
                

        hot_channel_width = 200 / self.heat_exchanger.hot_flow_sections
        cold_channel_width = 200 / self.heat_exchanger.cold_flow_sections
        largest_width = max(hot_channel_width, cold_channel_width)

        # draw baffles
        painter.setPen(QPen(Qt.GlobalColor.darkGray, 4))
        for i,n in enumerate(num_baffles):
            y_1 = 100 + i * cold_channel_width
            y_2 = 100 + (i+1) * cold_channel_width
            for j in range(int(n)):
                x_coord = (150 + 500 * (j + 1) / (n + 1)) * scale_x
                painter.drawLine(int(x_coord), int(y_1 * scale_y), int(x_coord), int(y_2 * scale_y))


        if (self.heat_exchanger.hot_flow_sections - self.heat_exchanger.cold_flow_sections) % 2 == 0:
            v_offset = 0.08 * largest_width
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
        
        # set positions of inlets and outlet boxes
        box_x_offset = -20
        self.cold_inlet_box.setGeometry(int((x_cold_in + box_x_offset) * scale_x), int(25 * scale_y), 40, 25)
        self.hot_inlet_box.setGeometry(int((x_hot_in + box_x_offset) * scale_x), int(25 * scale_y), 40, 25)
        self.cold_outlet_box.setGeometry(int((x_cold_out + box_x_offset) * scale_x), int(350 * scale_y), 40, 25)
        self.hot_outlet_box.setGeometry(int((x_hot_out + box_x_offset) * scale_x), int(350 * scale_y), 40, 25)

        # set position of output boxes
        label_width = 120
        for i,w in enumerate(self.get_output_widgets()):
            w.setGeometry(int(100 * scale_x + label_width * (i % 2)), int((500 + 50 * (i//2)) * scale_y), label_width, 25)

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
            x_per_zigzag = 200 * self.heat_exchanger.hot_flow_sections * self.heat_exchanger.cold_flow_sections / self.heat_exchanger.total_tubes
            num_segments1 = 2 + int((end_point1.x() - start_point1.x()) / x_per_zigzag)
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
            x_per_zigzag = 200 * self.heat_exchanger.hot_flow_sections / num_tubes[i]
            num_segments1 = 2 + int((end_point1.x() - start_point1.x()) // x_per_zigzag)
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

