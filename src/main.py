from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow, QLabel, QFileDialog, QPushButton, QListWidget
from PyQt6 import QtGui
from PyQt6.QtCore import Qt

import ctypes
myappid = 'cued.lwp26.shell_and_tube_heat_exchanger.1.0.0'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

import numpy as np

from constants import *
from heat_exchanger import Heat_Exchanger
from diagram import Heat_Exchanger_Diagram, Heat_Exchanger_Definition
from fluid_path import Fluid_Path, Entry_Constriction, Exit_Expansion, U_Bend, Heat_Transfer_Element
from optimiser import Optimise_Widget

## Hydraulic Analysis


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GA3 Heat Exchanger")
        self.icon = QtGui.QIcon("paint_icon.jpg")
        if self.icon: # check if file found
            self.setWindowIcon(self.icon)

        self.setGeometry(100, 100, 800, 800)

        layout = QtWidgets.QGridLayout()

        self.optimise_widget = Optimise_Widget()

        self.HE_definition = Heat_Exchanger_Definition()        
        layout.addWidget(self.HE_definition, 0, 0, 1, 2)

        
        self.HE_diagram = Heat_Exchanger_Diagram(600, 400)

        self.HE_definition.HE_update_signal.connect(self.HE_diagram.set_heat_exchanger)

        diagram_label = QLabel("Heat Exchanger Diagram")

        self.HE_diagram.setFixedWidth(600)
        layout.addWidget(diagram_label, 0, 2, 1, 1)
        layout.addWidget(self.HE_diagram, 1, 2, 4, 1)

        layout.addWidget(self.optimise_widget, 1, 0, 1, 2)


        # set the central widget of the Window

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

        self.show()

        HXchanger = self.HE_definition.set_heat_exchanger(1, 2, 13, 9)

        self.optimise_widget.set_design_template(HXchanger)
        self.optimise_widget.set_conditions([20,60])

 
        print(HXchanger.calc_mass())
        HXchanger.set_conditions([20,60])
        success = HXchanger.compute_effectiveness(method='LMTD')
        if success:
            print(HXchanger.Qdot)
            print(HXchanger.LMTD)
            print(HXchanger.effectiveness)

        #print(HXchanger.DT_min/(HXchanger.cold_flow_sections*HXchanger.Fscale*HXchanger.LMTD))

        success = HXchanger.compute_effectiveness(method='E_NTU')
        if success:
            print(HXchanger.Qdot)
            print(HXchanger.NTU)


        self.HE_definition.load_heat_exchanger(HXchanger)
        

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


