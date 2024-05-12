from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow, QLabel, QFileDialog, QPushButton, QListWidget
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRunnable
from PyQt6 import QtGui

import ctypes
myappid = 'cued.lwp26.shell_and_tube_heat_exchanger.1.0.0'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

import numpy as np

from constants import *
from heat_exchanger import Heat_Exchanger
from diagram import Heat_Exchanger_Diagram, Heat_Exchanger_Definition
from fluid_path import Fluid_Path, Entry_Constriction, Exit_Expansion, U_Bend, Heat_Transfer_Element
from optimiser import Optimise_Worker

## Hydraulic Analysis

def generate_heat_exchanger(hot_stages = 1, cold_stages = 1, tubes = 13, baffles = 9):
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

    HXchanger = Heat_Exchanger(Cold_path, Hot_path, 
                            flow_path_entries_side = Side.OPPOSITE)

    return HXchanger


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GA3 Heat Exchanger")
        self.icon = QtGui.QIcon("paint_icon.jpg")
        if self.icon: # check if file found
            self.setWindowIcon(self.icon)

        self.setGeometry(100, 100, 800, 800)

        layout = QtWidgets.QGridLayout()

        optimise_label = QLabel("Optimise Heat Exchanger")
        optimise_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.HE_definition = Heat_Exchanger_Definition()        
        layout.addWidget(self.HE_definition, 0, 0, 1, 2)

        self.start_optimise_button = QPushButton("Run Optimisation")

        self.cancel_optimise_button = QPushButton("Cancel Optimisation")
        self.cancel_optimise_button.setEnabled(False)

        self.start_optimise_button.clicked.connect(self.start_optimiser)
        self.cancel_optimise_button.clicked.connect(self.cancel_optimise)
        

        self.list_widget = QListWidget()

        self.HE_diagram = Heat_Exchanger_Diagram(600, 400)

        self.HE_definition.HE_update_signal.connect(self.HE_diagram.set_heat_exchanger)

        diagram_label = QLabel("Heat Exchanger Diagram")

        self.HE_diagram.setFixedWidth(600)
        layout.addWidget(diagram_label, 0, 2, 1, 1)
        layout.addWidget(self.HE_diagram, 1, 2, 4, 1)


        # make list widget uneditable
        self.list_widget.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # make list widget unselectable
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        layout.addWidget(optimise_label, 0, 6, 1, 2)
        layout.addWidget(self.start_optimise_button, 1, 6, 1, 2)
        layout.addWidget(self.cancel_optimise_button, 2, 6, 1, 2)
        layout.addWidget(self.list_widget, 3, 6, 1, 2)



        # set the central widget of the Window

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

        self.show()


        HXchanger = generate_heat_exchanger(1,1, 13, 9)
        epsilon = HXchanger.compute_effectiveness(20,60)
        print(epsilon)

        self.HE_definition.load_heat_exchanger(HXchanger)
    

    def start_optimiser(self):
        
        self.statusBar().showMessage("Starting optimiser")
        self.start_optimise_button.setEnabled(False)
        self.cancel_optimise_button.setEnabled(True)

 
    def cancel_optimise(self):
        self.start_optimise_button.setEnabled(True)
        self.cancel_optimise_button.setEnabled(False)

    def line_update(self, i):
        self.list_widget.setCurrentRow(i)
        # highlight current line
        self.list_widget.item(i).setSelected(True)

    def on_optimising_finished(self):
        self.start_optimise_button.setEnabled(True)
        self.cancel_optimise_button.setEnabled(False)

    def on_exit(self):
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    window = MainWindow()
    app.aboutToQuit.connect(window.on_exit)

    app.exec()


