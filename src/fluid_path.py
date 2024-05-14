from constants import *


class Entry_Constriction():
    def __init__(self):
        pass

    def loss_coefficient(self, Re, sigma):
        # valid linear range that was estimated from figure 8
        if sigma > 0.35:
            print("Warning: sigma > 0.35 which invalidates simple relation determined from figure 8")
            sigma = 0.35

        Kc = 0.5 - 0.5 * sigma
        return Kc
        

class Exit_Expansion():
    def __init__(self):
        pass

    def loss_coefficient(self, Re, sigma):
        # valid linear range that was estimated from figure 8
        if sigma > 0.35:
            print("Warning: sigma > 0.35 which invalidates simple relation determined from figure 8")
            sigma = 0.35

        Ke = (1-sigma)**2 
        return Ke

class U_Bend():
    def __init__(self):
        pass

    def loss_coefficient(self, Re = None):
        # TODO: find a better correlation for this
        # probably the same for hot and cold side
        return 1

class Heat_Transfer_Element():
    def __init__(self, tubes, baffles, flow_direction, tube_pattern):
        
        self.tubes = tubes
        self.baffles = baffles
        self.direction = flow_direction
        self.pattern = tube_pattern

    def friction_coefficient(self, Re, rel_rough):
        # if Re then apply different rules
        return (1.82 * np.log10(Re) - 1.64)**-2

        # THIS COULD ALSO POSSIBLY SOLVE THE ColeBrook-White equation for potentially better results.


class Fluid_Path():
    def __init__(self, rho, mu, cp, k):

        self.rho = rho
        self.mu = mu
        self.cp = cp
        self.k = k
        
        self.elements = []

    def add_element(self, element):
        
        self.elements.append(element)
        

