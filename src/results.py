from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QBrush
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QTimer
from PyQt6.QtWidgets import QLineEdit, QGridLayout, QVBoxLayout, QLabel, QPushButton, QSpinBox, QListWidget

import matplotlib
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT
)
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use('QtAgg')

import numpy as np

from constants import *


class Convergence_Graph(QWidget):
    def __init__(self):
        super().__init__()
        
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=True)
        self.ax = self.figure.add_subplot(111)

        self.draw_timer = QTimer()
        self.draw_timer.timeout.connect(self.graph_update)
        
        self.input_data = None
        self.output_data = np.zeros((0, 2))
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

        self.draw_timer.start(1000)
        self.clear()
    
    def set_heat_exchanger(self, heat_exchanger):

        if (self.input_data is None or 
            self.cold_flow_sections != heat_exchanger.cold_flow_sections or 
            self.hot_flow_sections != heat_exchanger.hot_flow_sections):

            self.cold_flow_sections = heat_exchanger.cold_flow_sections
            self.hot_flow_sections = heat_exchanger.hot_flow_sections
            self.input_data = np.zeros((0, self.cold_flow_sections + self.hot_flow_sections + 1))
            self.output_data = np.zeros((0, 2))
            self.clear()

    def new_data(self, data):

        if self.input_data is None:
            return
        
        inputs, outputs = data
        reshaped_inputs = np.reshape(inputs, (1, -1))

        self.input_data = np.append(self.input_data,
                                    reshaped_inputs, axis=0)
        
        reshaped_outputs = np.reshape(outputs, (1, -1))

        self.output_data = np.append(self.output_data, 
                                     reshaped_outputs, axis=0)
    
    def graph_update(self):
        
        if self.input_data is None:
            return
        if self.output_data.shape[0] == 0 or self.input_data.shape[0] == 0:
            return
        
        self.ax.clear()
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Qdot (kW)")
        self.ax.grid()

        n = self.output_data.shape[0]
        iterations = np.arange(n)
        lengths = self.input_data[:, 0]
        lengths = 50 * (lengths / np.max(lengths)) ** 2
        Qdot = self.output_data[:, 0] / 1e3 # kW

        self.ax.scatter(iterations, Qdot, s = lengths, c=self.output_data[:,0], cmap='plasma')
        self.ax.autoscale()
        self.canvas.draw_idle()

        self.figure.subplots_adjust(
            top=0.957,
            bottom=0.167,
            left=0.12,
            right=0.971,
            hspace=0.2,
            wspace=0.2
        )
    
    def clear(self):

        if self.input_data is not None:
            self.input_data = np.zeros((0, self.cold_flow_sections + self.hot_flow_sections + 1))
        self.output_data = np.zeros((0, 2))

        self.ax.clear()
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Qdot (W)")
        self.ax.grid()
        self.canvas.draw_idle()

        self.figure.subplots_adjust(
            top=0.957,
            bottom=0.167,
            left=0.12,
            right=0.971,
            hspace=0.2,
            wspace=0.2
        )

class State_Space_Graph(QWidget):
    def __init__(self):
        super().__init__()

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=True)
        # add colourbar to graph
        self.ax = self.figure.add_subplot(111)
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes('right', size='5%', pad=0.15)
        self.cax.set_axis_off()
        self.figure.tight_layout()

        self.draw_timer = QTimer()
        self.draw_timer.timeout.connect(self.graph_update)
        
        self.input_data = None
        self.output_data = np.zeros((0, 2))

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        
        self.setLayout(layout)

        self.draw_timer.start(1000)
        self.clear()

    def set_heat_exchanger(self, heat_exchanger):

        if (self.input_data is None or 
            self.cold_flow_sections != heat_exchanger.cold_flow_sections or 
            self.hot_flow_sections != heat_exchanger.hot_flow_sections):

            self.cold_flow_sections = heat_exchanger.cold_flow_sections
            self.hot_flow_sections = heat_exchanger.hot_flow_sections
            self.input_data = np.zeros((0, self.cold_flow_sections + self.hot_flow_sections + 1))
            self.output_data = np.zeros((0, 2))
            self.clear()
    
    def new_data(self, data):

        if self.input_data is None:
            return
        
        inputs, outputs = data
        reshaped_inputs = np.reshape(inputs, (1, -1))

        self.input_data = np.append(self.input_data,
                                    reshaped_inputs, axis=0)
        
        reshaped_outputs = np.reshape(outputs, (1, -1))

        self.output_data = np.append(self.output_data, 
                                     reshaped_outputs, axis=0)

    def graph_update(self):

        if self.input_data is None:
            return
        if self.output_data.shape[0] == 0:
            return
        
        # sort all data by Qdot
        # sorted_indices = np.argsort(self.input_data[:, 0])[::-1] # length sorted plot
        sorted_indices = np.argsort(self.output_data[:, 0]) # Qdot sorted plot
        self.input_data = self.input_data[sorted_indices]
        self.output_data = self.output_data[sorted_indices]

        self.ax.clear()
        self.ax.set_xlabel("Baffles per stage 1")
        self.ax.set_ylabel("Tubes per stage 1")
        self.ax.grid()

        pass1_baffles = np.rint(self.input_data[:, 1])
        pass1_tubes = np.rint(self.input_data[:, self.cold_flow_sections + 1])
        lengths = self.input_data[:, 0]
        lengths = 50 * (lengths / np.max(lengths)) ** 2
        colours = np.round(self.output_data[:,0] / 1e3, 1) # kW
        
        # set colour to Qdot value
        self.cax.set_axis_on()
        scat = self.ax.scatter(pass1_baffles, pass1_tubes, s = lengths, c=colours, cmap='plasma')
        self.figure.colorbar(scat, cax=self.cax, orientation='vertical')
        self.cax.set_ylabel('Qdot (kW)')
        
        self.ax.autoscale()
        self.canvas.draw_idle()

        self.figure.subplots_adjust(
            top=0.946,
            bottom=0.167,
            left=0.12,
            right=0.88,
            hspace=0.2,
            wspace=0.2
        )
    
    def clear(self):

        if self.input_data is not None:
            self.input_data = np.zeros((0, self.cold_flow_sections + self.hot_flow_sections + 1))
        self.output_data = np.zeros((0, 2))

        self.ax.clear()
        self.ax.set_xlabel("Baffles per stage")
        self.ax.set_ylabel("Tubes per stage")
        self.ax.set_ylim(0, 31)
        self.ax.set_xlim(0, 10)
        self.ax.grid()
        self.canvas.draw_idle()

        self.figure.subplots_adjust(
            top=0.946,
            bottom=0.167,
            left=0.12,
            right=0.88,
            hspace=0.2,
            wspace=0.2
        )


class Results_Widget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        layout = QGridLayout()

        self.convergence_graph = Convergence_Graph()
        self.state_space_graph = State_Space_Graph()

        layout.addWidget(self.convergence_graph, 0, 0, 1, 2)
        layout.addWidget(self.state_space_graph, 1, 0, 1, 2)

        self.setLayout(layout)

    def set_heat_exchanger(self, heat_exchanger):

        self.convergence_graph.set_heat_exchanger(heat_exchanger)
        self.state_space_graph.set_heat_exchanger(heat_exchanger)

    def on_exit(self):
        pass