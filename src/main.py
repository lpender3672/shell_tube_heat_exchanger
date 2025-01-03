from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow, QLabel, QFileDialog, QPushButton, QListWidget
from PyQt6 import QtGui
from PyQt6.QtCore import Qt

import ctypes
myappid = 'cued.lwp26.shell_and_tube_heat_exchanger.1.0.0'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

import numpy as np

from constants import *
from heat_exchanger import build_heat_exchanger
from diagram import Heat_Exchanger_Diagram, Heat_Exchanger_Definition
from fluid_path import Fluid_Path, Entry_Constriction, Exit_Expansion, U_Bend, Heat_Transfer_Element
from optimiser import Optimise_Widget
from results import Results_Widget

## Hydraulic Analysis


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GA3 Heat Exchanger")
        self.icon = QtGui.QIcon("paint_icon.jpg")
        if self.icon: # check if file found
            self.setWindowIcon(self.icon)

        self.setGeometry(100, 100, 1600, 900)

        layout = QtWidgets.QGridLayout()

        self.optimise_widget = Optimise_Widget(self)

        self.HE_definition = Heat_Exchanger_Definition(self)        
        layout.addWidget(self.HE_definition, 0, 0, 3, 2)

        
        self.HE_diagram = Heat_Exchanger_Diagram(self, 600, 400)

        diagram_label = QLabel("Heat Exchanger Diagram")

        self.HE_diagram.setFixedWidth(600)
        
        self.results_widget = Results_Widget(self)

    
        layout.addWidget(diagram_label, 0, 2, 1, 2)
        layout.addWidget(self.HE_diagram, 2, 2, 6, 2)

        layout.addWidget(self.optimise_widget, 3, 0, 3, 2)

        layout.addWidget(self.results_widget, 0, 4, 6, 4)


        # set the central widget of the Window

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

        self.show()

        HXchanger = build_heat_exchanger([2,3,4,5], [2,2], 0.3, Side.OPPOSITE, Pattern.SQUARE)


        self.optimise_widget.set_design_template(HXchanger)
        self.optimise_widget.set_conditions([20,60])
        
        #print(HXchanger.calc_mdot())
        HXchanger.set_conditions([20,60])
        success = HXchanger.compute_effectiveness(method='LMTD')
        if success:
            print(HXchanger.Qdot)
            print(HXchanger.DT_min/(HXchanger.cold_flow_sections*HXchanger.Fscale *HXchanger.LMTD))
            print(HXchanger.effectiveness)
            print(HXchanger.Fscale)


        success = HXchanger.compute_effectiveness(method='E_NTU')
        if success:
            print(HXchanger.Qdot)
            print(HXchanger.ntu)
            print(HXchanger.effectiveness)
            print(HXchanger.Qdot/(HXchanger.area_times_H*HXchanger.LMTD))


        self.attach_signals()
        self.HE_definition.load_heat_exchanger(HXchanger)
        self.HE_diagram.set_conditions([20,60])

        
    def attach_signals(self):
        # This attaches the heat exchanger update signal to the diagram update function
        self.HE_definition.HE_update_signal.connect(self.HE_diagram.set_heat_exchanger)

        # This sets the design template for the optimiser to the manually set heat exchanger
        self.HE_definition.HE_update_signal.connect(self.optimise_widget.set_design_template)

        # This attaches the heat exchanger update signal to the results widget
        self.HE_definition.HE_update_signal.connect(self.results_widget.set_heat_exchanger)


        # This attaches the graph update functions called each iteration
        self.optimise_widget.add_iteration_callback(
            self.results_widget.convergence_graph.new_data
            )
        self.optimise_widget.add_iteration_callback(
            self.results_widget.state_space_graph.new_data
            )
        
        # When the optimal heat exchanger is found, load it into the definition widget
        self.optimise_widget.start_optimise_button.clicked.connect(self.results_widget.convergence_graph.clear)
        self.optimise_widget.start_optimise_button.clicked.connect(self.results_widget.state_space_graph.clear)
        self.optimise_widget.optimal_found.connect(self.HE_definition.load_heat_exchanger)

    def line_update(self, i):
        self.list_widget.setCurrentRow(i)
        # highlight current line
        self.list_widget.item(i).setSelected(True)

    def on_exit(self):
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    window = MainWindow()
    app.aboutToQuit.connect(window.on_exit)

    app.exec()