from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRunnable, QThreadPool, QObject
from PyQt6.QtWidgets import QWidget, QMainWindow, QLabel, QFileDialog, QPushButton, QListWidget, QGridLayout, QMessageBox
from PyQt6 import QtGui
from PyQt6 import QtWidgets


from scipy.optimize import minimize as scipy_minimize

from heat_exchanger import Heat_Exchanger


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
            worker = Optimise_Worker(heat_exchanger, self.conditions)
            worker.iteration_update.connect(self.on_iteration_update)

            self.workers.append(worker)
            self.thread_pool.start(worker)
        

    def on_iteration_update(self, heat_exchanger):
        pass


 
    def cancel_optimise(self):
        self.start_optimise_button.setEnabled(True)
        self.cancel_optimise_button.setEnabled(False)



class Optimise_Worker(QRunnable, QObject):
    iteration_update = pyqtSignal(Heat_Exchanger)
    finished = pyqtSignal(Heat_Exchanger)

    def __init__(self, heat_exchanger, conditions):
        super().__init__()
        QObject.__init__(self)

        self.heat_exchanger = heat_exchanger
        self.cancelled = False

        self.T1in, self.T2in = conditions

    def objective_function(self, x):

        L, pitch, tubes, baffles = x

        self.heat_exchanger.set_geometry(L, pitch, tubes, baffles)
        self.iteration_update.emit(self.heat_exchanger)

        return 1 / self.heat_exchanger.compute_effectiveness(self.T1in, self.T2in)
        

    def run(self):

        # https://docs.scipy.org/doc/scipy/tutorial/optimize.html
        # 

        integer_constraint = lambda x : max([x[i]-int(x[i]) for i in range(len(x))])

        cons = ({'type':'eq', 'fun': integer_constraint},
                {'type':'eq','fun': integer_constraint})

        res = scipy_minimize(self.objective_function, 
                       [0.1, 0.1, 10, 10], 
                       constraints=cons)
        
        if res.success:
            print("Optimisation Successful")

        else:
            print("Optimisation Failed")

        self.finished.emit(self.heat_exchanger)

