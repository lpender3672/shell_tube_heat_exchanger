from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRunnable, QThreadPool, QObject
from PyQt6.QtWidgets import QWidget, QMainWindow, QLabel, QFileDialog, QPushButton, QListWidget, QGridLayout, QMessageBox
from PyQt6 import QtGui
from PyQt6 import QtWidgets


import numpy as np

from scipy.optimize import NonlinearConstraint, BFGS
from scipy.optimize import minimize as scipy_minimize

from heat_exchanger import Heat_Exchanger


class Optimise_Result():
    def __init__(self, heat_exchanger, success):
        self.heat_exchanger = heat_exchanger
        self.success = success


class Optimise_Widget(QWidget):

    def __init__(self, num_threads = 4):
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

    
    def build_constraints(self, heat_exchanger):
        
        constraints = []
    
        # force number of tubes and baffles to take integer values
        # TODO: This doesnt work, make this work
        integer_constraints = lambda x : np.append(x[:4], np.round(x[4:], 0))
        constraints.append({'type':'eq', 'fun': integer_constraints})

        # require hot and cold compressor rises greater than HX pressure drops (so comp_rise - pressure_drop > 0)
        flow_constraints = NonlinearConstraint(heat_exchanger.calc_rel_rise, [0,0], [np.inf, np.inf], jac='2-point', hess=BFGS())
        constraints.append(flow_constraints)

        # require mass < 1.20kg
        mass_constraint = NonlinearConstraint(heat_exchanger.calc_mass, 0, 1.20, jac='2-point', hess=BFGS())
        constraints.append(mass_constraint)
        
        # require length < 0.35

        return constraints


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
            constraints = self.build_constraints(heat_exchanger)
            worker = Optimise_Worker(heat_exchanger, self.conditions, constraints)
            
            worker.signal.iteration_update.connect(self.on_iteration_update)
            worker.signal.finished.connect(self.on_optimisation_finished)
            
            self.workers.append(worker)
            self.thread_pool.start(worker)

        
        print("Optimisation started")
        

    def on_iteration_update(self, heat_exchanger):
        pass


    def on_optimisation_finished(self, result):
        self.cancel_optimise()

        if result.success:
            print("Optimisation Successful")

            L = result.heat_exchanger.L_hot_tube
            pitch = result.heat_exchanger.pitch

            tubes = result.heat_exchanger.total_tubes
            baffles = result.heat_exchanger.total_baffles

            print(f"L = {L}, pitch = {pitch}, tubes = {tubes}, baffles = {baffles}")

 
    def cancel_optimise(self):
        self.start_optimise_button.setEnabled(True)
        self.cancel_optimise_button.setEnabled(False)


class Worker_Signals(QObject):
    iteration_update = pyqtSignal(Heat_Exchanger)
    finished = pyqtSignal(Optimise_Result)

class Optimise_Worker(QRunnable):

    def __init__(self, heat_exchanger, conditions, constraints):
        super().__init__()
        QObject.__init__(self)

        self.heat_exchanger = heat_exchanger
        self.cancelled = False

        self.Tin = conditions
        self.constraints = constraints

        self.signal = Worker_Signals()

    def objective_function(self, x):

        mdot_cold, mdot_hot, L, pitch, tubes, baffles = x

        self.heat_exchanger.set_geometry(L, pitch, tubes, baffles)
        self.heat_exchanger.set_mass_flow([mdot_cold, mdot_hot])
    
        self.heat_exchanger.compute_effectiveness(self.Tin, method = 'LMTD')

        self.signal.iteration_update.emit(self.heat_exchanger)

        return 1e4 / self.heat_exchanger.Qdot
        

    def run(self):

        # https://docs.scipy.org/doc/scipy/tutorial/optimize.html
        # 

        mdot_cold  = self.heat_exchanger.mdot_cold
        mdot_hot = self.heat_exchanger.mdot_hot
        L = self.heat_exchanger.L_hot_tube
        pitch = self.heat_exchanger.pitch

        tubes = self.heat_exchanger.total_tubes
        baffles = self.heat_exchanger.total_baffles

        try:
            res = scipy_minimize(self.objective_function, 
                        [mdot_cold, mdot_hot, L, pitch, tubes, baffles], 
                        method='trust-constr',
                        jac="2-point",
                        hess=BFGS(),
                        constraints=self.constraints,
                        options={'verbose': 1}
                        )
        except Exception as e:
            failed_result = Optimise_Result(
                self.heat_exchanger,
                False
            )
        else:
            result = Optimise_Result(
                self.heat_exchanger,
                res.success
            )

        self.signal.finished.emit(result)

