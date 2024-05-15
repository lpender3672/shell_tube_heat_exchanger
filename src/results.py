from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QBrush
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtWidgets import QLineEdit, QGridLayout, QVBoxLayout, QLabel, QPushButton, QSpinBox, QListWidget

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

from constants import *

class Convergence_Graph(QWidget):
    def __init__(self):
        super().__init__()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)

        self.data = [[] for _ in range(num_threads)]

    def new_data(self, heat_exchanger):

        id = heat_exchanger.id
        self.data[id].append(heat_exchanger.Qdot)

        self.plot()
    
    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for i in range(1):
            npoints = len(self.data[i])
            ax.plot(np.arange(0, npoints, 1), self.data[i], label = f"Thread {i}")

        ax.relim()
        ax.legend()
        ax.grid()
        self.canvas.draw()


class State_Space_Graph(QWidget):
    def __init__(self):
        super().__init__()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)

        self.data = np.zeros((num_threads, 0))

    

class Results_Widget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QGridLayout()

        self.convergence_graph = Convergence_Graph()
        self.state_space_graph = State_Space_Graph()

        layout.addWidget(self.convergence_graph, 0, 0, 1, 2)
        layout.addWidget(self.state_space_graph, 1, 0, 1, 2)

        self.setLayout(layout)

    def on_exit(self):
        pass