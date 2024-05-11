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
from fluid_path import Fluid_Path, Entry_Constriction, Exit_Expansion, L_Bend, U_Bend, Heat_Transfer_Element
from optimiser import Optimise_Worker

## Hydraulic Analysis


Hot_path = Fluid_Path(rho_w, mu, cp, k_w)
Hot_path.add_element(Entry_Constriction())
Hot_path.add_element(
    Heat_Transfer_Element(13, 5, Direction.COUNTERFLOW, Pattern.SQUARE)
)
Hot_path.add_element(Exit_Expansion())

Cold_path = Fluid_Path(rho_w, mu, cp, k_w)

Cold_path.add_element(
    Heat_Transfer_Element(13, 5, Direction.COUNTERFLOW, Pattern.SQUARE)
)

HXchanger = Heat_Exchanger(Cold_path, Hot_path)
HXchanger.compute_effectiveness(1,2)



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GA3 Heat Exchanger")
        self.icon = QtGui.QIcon("paint_icon.jpg")
        if self.icon: # check if file found
            self.setWindowIcon(self.icon)

        layout = QtWidgets.QVBoxLayout()

        label = QLabel("Optimise Heat Exchanger")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)


        self.start_optimise_button = QPushButton("Run Optimisation")

        self.cancel_optimise_button = QPushButton("Cancel Optimisation")
        self.cancel_optimise_button.setEnabled(False)

        self.start_optimise_button.clicked.connect(self.start_optimiser)
        self.cancel_optimise_button.clicked.connect(self.cancel_optimise)
        

        self.list_widget = QListWidget()

        # make list widget uneditable
        self.list_widget.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # make list widget unselectable
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        layout.addWidget(label)
        layout.addWidget(self.start_optimise_button)
        layout.addWidget(self.cancel_optimise_button)
        layout.addWidget(self.list_widget)

        # set the central widget of the Window

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

        self.show()
    

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


