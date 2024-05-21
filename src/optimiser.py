from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRunnable, QThreadPool, QObject
from PyQt6.QtWidgets import QWidget, QMainWindow, QLabel, QFileDialog, QPushButton, QListWidget, QGridLayout, QMessageBox
from PyQt6 import QtGui
from PyQt6 import QtWidgets

import copy
import numpy as np

from scipy.optimize import NonlinearConstraint, BFGS, OptimizeResult
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import shgo as scipy_shgo

import logging
import time

from constants import *
from heat_exchanger import Heat_Exchanger, pitch_from_tubes, build_heat_exchanger


class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class Optimise_Widget(QWidget):
    optimal_found = pyqtSignal(Heat_Exchanger)

    def __init__(self, parent):
        super().__init__(parent)

        self.num_threads = num_threads
        self.template = None
        self.conditions = None

        self.iteration_callbacks = []

        layout = QGridLayout()

        optimise_label = QLabel("Optimise Heat Exchanger")
        optimise_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.log_text = QTextEditLogger(self)

        self.log_text.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(self.log_text)

        logging.getLogger().setLevel(logging.INFO)


        self.start_optimise_button = QPushButton("Run Optimisation")

        self.cancel_optimise_button = QPushButton("Cancel Optimisation")
        self.cancel_optimise_button.setEnabled(False)

        self.start_optimise_button.clicked.connect(self.start_optimiser)
        self.cancel_optimise_button.clicked.connect(self.cancel_optimise)


        layout.addWidget(optimise_label, 0, 6, 1, 2)
        layout.addWidget(self.start_optimise_button, 1, 6, 1, 2)
        layout.addWidget(self.cancel_optimise_button, 2, 6, 1, 2)
        layout.addWidget(self.log_text.widget, 3, 6, 1, 2)

        self.setLayout(layout)

    def set_design_template(self, heat_exchanger_template):
        self.template = heat_exchanger_template

    def set_conditions(self, conditions):
        self.conditions = conditions

    def add_iteration_callback(self, callback):
        self.iteration_callbacks.append(callback)

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
        self.thread_pool.setMaxThreadCount(self.num_threads)

        self.workers = []
        for i in range(self.num_threads):
            heat_exchanger = copy.deepcopy(self.template)
            heat_exchanger.set_conditions(self.conditions)
            heat_exchanger.id = i

            # scipy optimse worker
            # worker = Scipy_Optimise_Worker(heat_exchanger)
            # worker.build_constraints()

            # scipy global optimise worker
            worker = Scipy_Global_Optimise_Worker(heat_exchanger)
            # worker.signal.moveToThread(self.thread_pool.thread())
            
            # brute force worker
            # worker = Brute_Force_Worker(heat_exchanger)

            for callback in self.iteration_callbacks:
                worker.signal.iteration_update.connect(callback, Qt.ConnectionType.QueuedConnection)

            worker.signal.finished.connect(self.on_optimisation_finished, Qt.ConnectionType.QueuedConnection)

            worker.build_constraints()
            
            self.workers.append(worker)
            worker.run()
            #self.thread_pool.start(worker)

        
        logging.info("Optimisation started")

    def on_optimisation_finished(self, result):
        self.cancel_optimise()

        if result.success:
            logging.info("Optimisation Successful")

            mass = result.heat_exchanger.calc_mass()
            l = result.x[0]
            baffles = np.rint(result.x[1:result.heat_exchanger.cold_flow_sections + 1])
            tubes = np.rint(result.x[result.heat_exchanger.cold_flow_sections + 1:])

            logging.info(f"\nTubes = {tubes}, \nBaffles = {baffles}, \nLength = {l}, \nMass = {mass}")
            logging.info(f"\n mdot_cold = {result.heat_exchanger.mdot[0]}\n mdot_hot = {result.heat_exchanger.mdot[1]}")
            logging.info(f"Qdot = {result.heat_exchanger.Qdot}, effectiveness = {result.heat_exchanger.effectiveness}")

            self.optimal_found.emit(result.heat_exchanger)
        
        else:
            logging.info("Optimisation Failed")

    def cancel_optimise(self):
        self.start_optimise_button.setEnabled(True)
        self.cancel_optimise_button.setEnabled(False)


class Worker_Signals(QObject):
    iteration_update = pyqtSignal(np.ndarray)
    finished = pyqtSignal(OptimizeResult)

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
            l = x[0]
            baffles = x[1:self.heat_exchanger.cold_flow_sections + 1]
            tubes = x[self.heat_exchanger.cold_flow_sections + 1:]
            self.heat_exchanger.set_geometry(l, tubes, baffles)
            return self.heat_exchanger.calc_mass()

        # require mass < 1.20kg
        mass_constraint = NonlinearConstraint(calc_mass, 0, max_HE_mass, jac='2-point')
        constraints.append(mass_constraint)

        # require pitch to be greater than D_outer_tube
        def calc_pitch(x):
            l = x[0]
            baffles = x[1:self.heat_exchanger.cold_flow_sections + 1]
            tubes = x[self.heat_exchanger.cold_flow_sections + 1:]
            self.heat_exchanger.set_geometry(l, tubes, baffles)
            return min(self.heat_exchanger.get_pitch())
        
        pitch_constraint = NonlinearConstraint(calc_pitch, D_outer_tube, D_shell, jac='2-point')
        #constraints.append(pitch_constraint)

        # force number of tubes and baffles to take integer values
        def integer_constraints(x):
            f = 100
            x[1:] = f * (x[1:] % 1)
            return x
        
        #constraints.append({'type':'eq', 'fun': integer_constraints})

        # Total tube length constraint
        def total_tube_length(x):
            l = x[0]
            tubes = np.rint(x[self.heat_exchanger.cold_flow_sections + 1:])
            total_length = np.sum(tubes) * l
            return  max_total_tube_length - total_length
        
        length_constraint = {'type':'ineq', 'fun': total_tube_length}
        constraints.append(length_constraint)


        # This constraint ensures that the number of tubes do not decrease as the flow path progresses
        # This is to shrink the search space and make the optimisation faster
        # This is thought to not exclude the optimal solution
        # As temperature difference decreases the area of heat transfer should increase to maintain local heat transfer
        # Increasing the area of 
        def ascending_tubes(x):
            tubes = x[self.heat_exchanger.cold_flow_sections + 1:]
            return np.diff(tubes) + 1e-6

        ascending_tubes_constraint = {'type':'ineq', 'fun': ascending_tubes}
        constraints.append(ascending_tubes_constraint)

        self.constraints = constraints


    def objective_function(self, x, *args):

        l = x[0]
        baffles = x[1:self.heat_exchanger.cold_flow_sections + 1]
        tubes = x[self.heat_exchanger.cold_flow_sections + 1:]

        self.heat_exchanger.set_geometry(l, tubes, baffles)

        result = self.heat_exchanger.compute_effectiveness(method = 'LMTD')

        if not result:
            return np.inf

        if self.iteration_count % self.emit_interval == 0:
            output = [self.heat_exchanger.Qdot, self.heat_exchanger.effectiveness]
            update = np.array([x, output], dtype=object)
            self.signal.iteration_update.emit(update)

        self.iteration_count += 1
        return 1e4 / self.heat_exchanger.Qdot
        

    def run(self):

        # https://docs.scipy.org/doc/scipy/tutorial/optimize.html
        # 
        max_tubes = 24
        max_baffles_per_section = 8
        max_tubes_per_section = max_tubes // self.heat_exchanger.hot_flow_sections
        rand_tubes = np.random.randint(1, max_baffles_per_section)
        rand_baffles = np.random.randint(1, max_tubes_per_section)

        x0 = [0.3]
        x0.extend([rand_baffles for _ in range(self.heat_exchanger.cold_flow_sections)])
        x0.extend([rand_tubes for _ in range(self.heat_exchanger.hot_flow_sections)])

        try:
            result = scipy_minimize(
                            self.objective_function, 
                            x0, 
                            method='trust-constr',
                            jac="2-point",
                            hess=BFGS(),
                            constraints=self.constraints,
                            options={'verbose': 1, 'maxiter':1000}
                            )
        except Exception as e:
            print(e)
        
        else:
            if result.success:            
                self.objective_function(result.x)
                result.heat_exchanger = self.heat_exchanger

            self.signal.finished.emit(result)

class Scipy_Global_Optimise_Worker(Scipy_Optimise_Worker):
    def __init__(self, heat_exchanger):
        super().__init__(heat_exchanger)

    def run(self):

        max_tubes = 24 // self.heat_exchanger.hot_flow_sections
        max_baffles = 8

        lmax = max_HE_length - end_cap_width_nozzle
        if self.heat_exchanger.hot_flow_sections % 2 == 0:
             # for hot flow nozzles on same side the opposite end cap width is smaller, allowing for larger max length
            lmax -= end_cap_width
        else:
            lmax -= end_cap_width_nozzle

        length_bounds = (0, lmax)
        tube_bounds = (1, max_tubes)
        baffle_bounds = (1, max_baffles)

        bounds = [length_bounds]
        bounds.extend([baffle_bounds for _ in range(self.heat_exchanger.cold_flow_sections)])
        bounds.extend([tube_bounds for _ in range(self.heat_exchanger.hot_flow_sections)])

        complexity = 1 + self.heat_exchanger.cold_flow_sections + self.heat_exchanger.hot_flow_sections

        try:
            result = scipy_shgo(self.objective_function, 
                                bounds = bounds,
                                constraints=self.constraints,
                                n = 50,
                                options = {
                                    'maxtime' : 60,
                                    'f_min' : 0.1,
                                    'f_tol' : 0.001,
                                    'constraints_tol': 1e-8,
                                },
                                sampling_method='sobol',
                                )
        except Exception as e:
            print(e)
        
        else:
            print(result)

            self.objective_function(result.x)
            result.heat_exchanger = self.heat_exchanger

            self.signal.finished.emit(
                result
                )        

"""
class Brute_Force_Worker(QRunnable):
    def __init__(self, heat_exchanger, id = 0):
        super().__init__()

        self.heat_exchanger = heat_exchanger
        self.id = id
        self.cancelled = False

        self.log_interval = 1000

        self.signal = Worker_Signals()

    def check_constraints(self):

        if self.heat_exchanger.calc_mass() > 1.20:
            return False

        if np.min(self.heat_exchanger.get_pitch()) < D_outer_tube + pitch_offset:
            return False

        return True

    def run(self):

        max_tubes = 24
        max_baffles = 30
        
        max_baffles_per_section = max_baffles // self.heat_exchanger.cold_flow_sections
        max_tubes_per_section = max_tubes // self.heat_exchanger.hot_flow_sections
        
        best_design_Qdot = 0
        best_design = None

        lengths = np.linspace(0.15, 0.35 - 2 * end_cap_width, 20)
        
        # create a (hot_flow_sections + cold_flow_sections) dimensional meshgrid of all possible geometries
        baffles = np.arange(1, max_baffles_per_section)
        tubes = np.arange(1, max_tubes_per_section)

        mgrid_args = [lengths]
        mgrid_args.extend([baffles for _ in range(self.heat_exchanger.cold_flow_sections)])
        mgrid_args.extend([tubes for _ in range(self.heat_exchanger.hot_flow_sections)])
        
        mgrid = np.meshgrid(*mgrid_args, indexing='ij')
        
        n = np.prod(mgrid[0].shape)
        l_vals = mgrid[0].flatten()
        baffle_vals = np.array([mgrid[i].flatten() for i in range(1, self.heat_exchanger.cold_flow_sections + 1)])
        tube_vals = np.array([mgrid[i].flatten() for i in range(self.heat_exchanger.cold_flow_sections + 1, len(mgrid))])

        for i in range(n):

            x = [l_vals[i], tube_vals[:, i], baffle_vals[:, i]]
            self.heat_exchanger.set_geometry( *x)


            if not self.check_constraints():
                continue

            result = self.heat_exchanger.compute_effectiveness(method = 'LMTD', optimiser = "fsolve")

            #output = [self.heat_exchanger.Qdot, self.heat_exchanger.effectiveness]
            #x = [tubes, baffles]
            #self.signal.iteration_update.emit([x, output])
            
            if not result:
                continue
            
            try:
                Qdot = self.heat_exchanger.Qdot
                assert Qdot is not None
            except (AttributeError, AssertionError):
                continue

            if self.heat_exchanger.Qdot > best_design_Qdot:
                best_design_Qdot = self.heat_exchanger.Qdot
                best_design = x

            if self.cancelled:
                return
        

        result = build_heat_exchanger(  best_design[1], best_design[2], best_design[0], 
                                        self.heat_exchanger.flow_path_entries_side, 
                                        Pattern.SQUARE)
        result.set_conditions(self.heat_exchanger.Tin)
        result.compute_effectiveness(method = 'LMTD', optimiser = 'fsolve')

        self.signal.finished.emit(
            Optimise_Result(result, True)
            )
"""
            
        
