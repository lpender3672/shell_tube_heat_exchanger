from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow, QLabel, QFileDialog, QPushButton, QListWidget
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRunnable

import numpy as np

from constants import *

from heat_exchanger import Heat_Exchanger
from optimiser import Optimiser

## Hydraulic Analysis


HXchanger = Heat_Exchanger(Pattern.SQUARE, 13, 9)
HXchanger.compute_effectiveness()

class optimise_worker(QRunnable):
    iteration_update = pyqtSignal(Heat_Exchanger)
    finished = pyqtSignal(Heat_Exchanger)

    def __init__(self, optimiser):
        super().__init__()

        self.optimiser = optimiser
        self.cancelled = False


    def run(self):
        while not self.cancelled:
            self.msleep(1)
        # randomise input conditions


        
        


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GA3 Heat Exchanger")

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


