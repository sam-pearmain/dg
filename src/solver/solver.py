
from physics import Physics
from meshing import Mesh
from logging import Logger
from utils import todo, Uninit
from numerics.timestepping.integrator import Integrator

class SolverSettings():
    def __init__(
            self,
            max_iters: int
        ):
        self.max_iters = max_iters

    @classmethod
    def default(self):
        return SolverSettings(
            max_iters = 1000
        )

class Solver():
    settings:   SolverSettings
    integrator: Integrator
    physics:    Physics
    logger:     Logger
    mesh:       Mesh
    iter:       int

    def __init__(
            self, 
            settings: SolverSettings,
            integrator: Integrator,
            physics: Physics,
            logger: Logger, 
            mesh: Mesh,
            iter: int = 0,
        ):
        self.settings = settings
        self.integrator = integrator
        self.physics = physics
        self.logger = logger
        self.mesh = mesh
        self.iter = iter
    
    @classmethod
    def load_from_checkpoint(filepath: str):
        todo()

    def solve(self):
        integrator = self.integrator

        while self.iter < self.settings.max_iters:
            """This is the mainloop of the solver"""
            # update the time step size
            integrator.update_time_step(self)

            # integrate in time
            integrator.step(self)

            # increment the number of iterations
            self.iter += 1


    @property
    def time_step(self):
        return self.integrator.time_step