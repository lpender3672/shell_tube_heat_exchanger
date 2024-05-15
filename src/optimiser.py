from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRunnable, QThreadPool, QObject
from PyQt6.QtWidgets import QWidget, QMainWindow, QLabel, QFileDialog, QPushButton, QListWidget, QGridLayout, QMessageBox
from PyQt6 import QtGui
from PyQt6 import QtWidgets

import copy
import numpy as np

from scipy.optimize import NonlinearConstraint, BFGS
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import shgo as scipy_shgo

from constants import *
from heat_exchanger import Heat_Exchanger


class Optimise_Result():
    def __init__(self, heat_exchanger, success):
        self.heat_exchanger = heat_exchanger
        self.success = success


class Optimise_Widget(QWidget):
    iteration_update = pyqtSignal(Heat_Exchanger)

    def __init__(self):
        super().__init__()

        self.num_threads = num_threads
        self.template = None
        self.conditions = None

        layout = QGridLayout()

        optimise_label = QLabel("Optimise Heat Exchanger")
        optimise_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.list_widget = QListWidget()

        self.start_optimise_button = QPushButton("Run Optimisation")

        self.cancel_optimise_button = QPushButton("Cancel Optimisation")
        self.cancel_optimise_button.setEnabled(False)

        self.start_optimise_button.clicked.connect(self.start_optimiser)
        self.cancel_optimise_button.clicked.connect(self.cancel_optimise)

         # make list widget uneditable
        self.list_widget.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # make list widget unselectable
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)


        layout.addWidget(optimise_label, 0, 6, 1, 2)
        layout.addWidget(self.start_optimise_button, 1, 6, 1, 2)
        layout.addWidget(self.cancel_optimise_button, 2, 6, 1, 2)
        layout.addWidget(self.list_widget, 3, 6, 1, 2)

        self.setLayout(layout)

    def set_design_template(self, heat_exchanger_template):
        self.template = heat_exchanger_template

    def set_conditions(self, conditions):
        self.conditions = conditions


    def start_optimiser(self):

        if not self.template or not self.conditions:
            # raise error
            QMessageBox.critical(self, "Error", "Please set the design template and conditions")
            return
        
        #self.statusBar().showMessage("Starting optimiser")
        self.start_optimise_button.setEnabled(False)
        self.cancel_optimise_button.setEnabled(True)

        # use the thread pool to run the optimiser
        self.thread_pool = QThreadPool.globalInstance()

        self.workers = []
        for i in range(self.num_threads):
            heat_exchanger = self.template.get_random_geometry_copy()
            heat_exchanger.set_conditions(self.conditions)
            heat_exchanger.id = i

            # scipy optimse worker
            worker = Scipy_Global_Optimise_Worker(heat_exchanger)
            worker.build_constraints()
            
            worker.signal.iteration_update.connect(self.on_iteration_update)
            worker.signal.finished.connect(self.on_optimisation_finished)
            
            self.workers.append(worker)
            self.thread_pool.start(worker)

        
        print("Optimisation started")
        

    def on_iteration_update(self, heat_exchanger):
        self.iteration_update.emit(heat_exchanger)


    def on_optimisation_finished(self, result):
        self.cancel_optimise()

        if result.success:
            print("Optimisation Successful")

            L = result.heat_exchanger.L_hot_tube
            pitch = result.heat_exchanger.pitch

            tubes = result.heat_exchanger.total_tubes
            baffles = result.heat_exchanger.total_baffles

            mass = result.heat_exchanger.calc_mass()

            print(f"L = {L}, tubes = {tubes}, baffles = {baffles}, mass = {mass}")
            print(f"mdot_cold = {result.heat_exchanger.mdot[0]}, mdot_hot = {result.heat_exchanger.mdot[1]}")
            print(f"Qdot = {result.heat_exchanger.Qdot}, effectiveness = {result.heat_exchanger.effectiveness}")
        
        else:
            print("Optimisation Failed")

    def cancel_optimise(self):
        self.start_optimise_button.setEnabled(True)
        self.cancel_optimise_button.setEnabled(False)


class Worker_Signals(QObject):
    iteration_update = pyqtSignal(Heat_Exchanger)
    finished = pyqtSignal(Optimise_Result)

class Scipy_Optimise_Worker(QRunnable):

    def __init__(self, heat_exchanger):
        super().__init__()
        QObject.__init__(self)

        self.heat_exchanger = heat_exchanger
        self.cancelled = False
        self.iteration_count = 0
        self.emit_interval = 10

        self.signal = Worker_Signals()

    def build_constraints(self):
        
        constraints = []
    
        def calc_mass(x):
            self.heat_exchanger.set_geometry(0.35, x[0], x[1])
            return self.heat_exchanger.calc_mass()

        # require mass < 1.20kg
        mass_constraint = NonlinearConstraint(calc_mass, 0.5, 1.20, jac='2-point', hess=BFGS())
        constraints.append(mass_constraint)


        # force number of tubes and baffles to take integer values
        def integer_constraints(x):
            # Modify x directly to enforce integer constraints
            x[0] = x[0] % 1
            x[1] = x[1] % 1
            return x
        
        constraints.append({'type':'eq', 'fun': integer_constraints})

        # range constraints
        max_tubes = 24
        max_baffles_per_section = 30
        max_tubes_per_section = max_tubes // self.heat_exchanger.hot_flow_sections

        constraints.append({'type':'ineq', 'fun': lambda x: x[0] - 1})
        constraints.append({'type':'ineq', 'fun': lambda x: x[1] - 1})
        constraints.append({'type':'ineq', 'fun': lambda x: max_tubes_per_section - x[0]})
        constraints.append({'type':'ineq', 'fun': lambda x: max_baffles_per_section - x[1]})

        # require hot and cold compressor rises greater than HX pressure drops (so comp_rise - pressure_drop > 0)
        flow_range_constraint = NonlinearConstraint(self.heat_exchanger.calc_mdot, 
                                                    [cold_side_compressor_characteristic_2024[0,0],hot_side_compressor_characteristic_2024[0,0]], 
                                                    [cold_side_compressor_characteristic_2024[0,-1],hot_side_compressor_characteristic_2024[0,-1]], 
                                                    jac='2-point', hess=BFGS())
        #constraints.append(flow_range_constraint)

        self.constraints = constraints

    def objective_function(self, x):

        self.heat_exchanger.set_geometry(0.35, x[0], x[1])
    
        result = self.heat_exchanger.compute_effectiveness(method = 'LMTD')

        #if not result:  return np.inf
        if self.iteration_count % self.emit_interval == 0:
            self.signal.iteration_update.emit(self.heat_exchanger)
        self.iteration_count += 1

        return 1e4 / self.heat_exchanger.Qdot
        

    def run(self):

        # https://docs.scipy.org/doc/scipy/tutorial/optimize.html
        # 

        tubes = self.heat_exchanger.total_tubes
        baffles = self.heat_exchanger.total_baffles

        res = scipy_minimize(
                        self.objective_function, 
                        [tubes, baffles], 
                        method='trust-constr',
                        jac="2-point",
                        hess=BFGS(),
                        constraints=self.constraints,
                        options={'verbose': 1, 'maxiter':1000}
                        )

        result = Optimise_Result(
            self.heat_exchanger,
            res.success
        )

        self.signal.finished.emit(result)

class Scipy_Global_Optimise_Worker(QRunnable):
    def __init__(self, heat_exchanger):
        super().__init__()
        QObject.__init__(self)

        self.heat_exchanger = heat_exchanger
        self.cancelled = False
        self.iteration_count = 0
        self.emit_interval = 10

        self.signal = Worker_Signals()

    def objective_function(self, x):

        self.heat_exchanger.set_geometry(0.35, x[0], x[1])    
        result = self.heat_exchanger.compute_effectiveness(method = 'LMTD')

        if self.iteration_count % self.emit_interval == 0:
            self.signal.iteration_update.emit(self.heat_exchanger)

        self.iteration_count += 1
        return 1e4 / self.heat_exchanger.Qdot

    def build_constraints(self):
        
        constraints = []

        def calc_mass(x):
            self.heat_exchanger.set_geometry(0.35, x[0], x[1])
            return self.heat_exchanger.calc_mass()

        # require mass < 1.20kg
        mass_constraint = NonlinearConstraint(calc_mass, 0.5, 1.20, jac='2-point')
        constraints.append(mass_constraint)

        def integer_constraints(x):
            x[0] = x[0] % 1
            x[1] = x[1] % 1
            return x
        
        constraints.append({'type':'eq', 'fun': integer_constraints})

        self.constraints = constraints

    def run(self):

        max_tubes = 24
        max_baffles_per_section = 30
        max_tubes_per_section = max_tubes // self.heat_exchanger.hot_flow_sections

        result = scipy_shgo(self.objective_function, 
                            bounds = [(0.5, 50), 
                                      (0.5, 50)],
                            constraints=self.constraints, 
                            iters=100,
                            options={'verbose': 1},
                            sampling_method='sobol'
                            )
        
        self.signal.finished.emit(
            Optimise_Result(self.heat_exchanger, result.success)
            )
        

class Brute_Force_Worker(QRunnable):
    def __init__(self, heat_exchanger, id = 0):
        super().__init__()
        QObject.__init__(self)

        self.heat_exchanger = heat_exchanger
        self.id = id
        self.cancelled = False

        self.signal = Worker_Signals()


    def run(self):

        max_tubes = 24
        max_baffles_per_section = 30

        max_tubes_per_section = max_tubes // self.heat_exchanger.hot_flow_sections
        
        best_design = copy.deepcopy(self.heat_exchanger)
        best_design.Qdot = 0

        for tubes in range(1, max_tubes_per_section):
            for baffles in range(1, max_baffles_per_section):

                self.heat_exchanger.set_geometry(0.35, tubes, baffles)
                result = self.heat_exchanger.compute_effectiveness(method = 'LMTD')

                if not result:
                    continue

                if self.heat_exchanger.Qdot > best_design.Qdot:
                    best_design = copy.deepcopy(self.heat_exchanger)

                self.signal.iteration_update.emit(self.heat_exchanger)

                if self.cancelled:
                    return
        
        self.signal.finished.emit(
            Optimise_Result(best_design, True)
            )
        
