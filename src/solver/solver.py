from numerics.timestepping.integrator import IntegratorType

class SolverSettings():
    def __init__(self):
        self.time_integrator = IntegratorType.ForwardEuler


class Solver():
    def __init__(self, settings, physics, mesh):
        self.settings = settings
        self.physics = physics
        self.mesh = mesh

        
        
