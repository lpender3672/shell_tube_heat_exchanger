from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRunnable

from scipy.optimize import minimize as scipy_minimize

from heat_exchanger import Heat_Exchanger



class Optimise_Worker(QRunnable):
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

        # https://docs.scipy.org/doc/scipy/tutorial/optimize.html

        # minimize