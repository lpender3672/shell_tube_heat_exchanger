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
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Qdot (W)")
        self.ax.set_ylim(2000, 20000)
        self.ax.set_xlim(0, 200)
        
        self.index = 0
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def new_data(self, data):

        Qdot = data[1][0]
        tubes, baffles = data[0]
        
        # add new data to graph without clearing
        self.ax.plot(self.index, Qdot, 'ro')
        self.ax.grid()
        self.canvas.draw()

        self.index += 1

class State_Space_Graph(QWidget):
    def __init__(self):
        super().__init__()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # add colourbar to graph
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Tubes per stage")
        self.ax.set_ylabel("Baffles per stage")
        self.ax.set_ylim(0, 21)
        self.ax.set_xlim(0, 10)

        # display a colour map legend
        

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)

        self.data = np.zeros((num_threads, 0))
    
    def new_data(self, data):

        Qdot = data[1][0]
        tubes, baffles = data[0]
        
        # add new data to graph without clearing
        # get colour from Qdot
        c = plt.cm.jet((Qdot - 2000) / 18000)
        self.ax.plot(tubes, baffles, 'o', color = c)
        self.ax.grid()
        self.canvas.draw()

    

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