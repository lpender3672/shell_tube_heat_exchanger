from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRunnable, QThreadPool, QObject
from PyQt6.QtWidgets import QWidget, QMainWindow, QLabel, QFileDialog, QPushButton, QListWidget, QGridLayout, QMessageBox
from PyQt6 import QtGui
from PyQt6 import QtWidgets

import copy
import numpy as np

from scipy.optimize import NonlinearConstraint, BFGS
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import shgo as scipy_shgo

import logging

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


class Optimise_Result():
    def __init__(self, heat_exchanger, success):
        self.heat_exchanger = heat_exchanger
        self.success = success


class Optimise_Widget(QWidget):
    optimal_found = pyqtSignal(Heat_Exchanger)

    def __init__(self):
        super().__init__()

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
            # worker = Scipy_Global_Optimise_Worker(heat_exchanger)
            # worker.build_constraints()

            # brute force worker
            worker = Brute_Force_Worker(heat_exchanger)

            for callback in self.iteration_callbacks:
                worker.signal.iteration_update.connect(callback)

            worker.signal.finished.connect(self.on_optimisation_finished)
            
            self.workers.append(worker)
            self.thread_pool.start(worker)

        
        logging.info("Optimisation started")

    def on_optimisation_finished(self, result):
        self.cancel_optimise()

        if result.success:
            logging.info("Optimisation Successful")

            L = result.heat_exchanger.L_hot_tube

            tubes = result.heat_exchanger.hot_path.elements[1].tubes
            baffles = result.heat_exchanger.cold_path.elements[0].baffles

            mass = result.heat_exchanger.calc_mass()


            logging.info(f"L_tube = {L}, tubes per stage = {tubes}, baffles per stage = {baffles}, mass = {mass}")
            logging.info(f"mdot_cold = {result.heat_exchanger.mdot[0]}, mdot_hot = {result.heat_exchanger.mdot[1]}")
            logging.info(f"Qdot = {result.heat_exchanger.Qdot}, effectiveness = {result.heat_exchanger.effectiveness}")

            self.optimal_found.emit(result.heat_exchanger)
        
        else:
            logging.info("Optimisation Failed")

    def cancel_optimise(self):
        self.start_optimise_button.setEnabled(True)
        self.cancel_optimise_button.setEnabled(False)


class Worker_Signals(QObject):
    iteration_update = pyqtSignal(list)
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

        cold_sections = self.heat_exchanger.cold_flow_sections
        hot_sections = self.heat_exchanger.hot_flow_sections

        l = x[0]
        baffles = x[1:cold_sections + 1]
        tubes = x[cold_sections + 1:]

        self.heat_exchanger.set_geometry(0.35, x[0], x[1])
    
        result = self.heat_exchanger.compute_effectiveness(method = 'LMTD')

        #if not result:  return np.inf
        if self.iteration_count % self.emit_interval == 0:
            output = [self.heat_exchanger.Qdot, self.heat_exchanger.effectiveness]
            self.signal.iteration_update.emit([x, output])

        self.iteration_count += 1

        return 1e4 / self.heat_exchanger.Qdot
        

    def run(self):

        # https://docs.scipy.org/doc/scipy/tutorial/optimize.html
        # 
        max_tubes = 24
        max_baffles_per_section = 30
        max_tubes_per_section = max_tubes // self.heat_exchanger.hot_flow_sections
        rand_tubes = np.random.randint(1, max_baffles_per_section)
        rand_baffles = np.random.randint(1, max_tubes_per_section)

        res = scipy_minimize(
                        self.objective_function, 
                        [rand_tubes, rand_baffles], 
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

        self.heat_exchanger = heat_exchanger
        self.cancelled = False
        self.iteration_count = 0
        self.emit_interval = 10

        self.signal = Worker_Signals()

    def objective_function(self, x, *args):

        self.heat_exchanger.set_geometry(0.35, x[0], x[1])    
        result = self.heat_exchanger.compute_effectiveness(method = 'LMTD')

        if self.iteration_count % self.emit_interval == 0:
            output = [self.heat_exchanger.Qdot, self.heat_exchanger.effectiveness]
            self.signal.iteration_update.emit([x, output])

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

        # require pitch to be greater than D_outer_tube
        def calc_pitch(x):
            return pitch_from_tubes(x[0], Pattern.SQUARE)
        
        pitch_constraint = NonlinearConstraint(calc_pitch, D_outer_tube, D_shell, jac='2-point')
        constraints.append(pitch_constraint)

        # force number of tubes and baffles to take integer values
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

            if i % self.log_interval == 0:
                print(f"Worker {self.id} iteration {i} of {n}")
                logging.info(f"Worker {self.id} iteration {i} of {n}")

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

            
        
