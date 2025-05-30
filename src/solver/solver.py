
from physics import Physics
from meshing import Mesh
from numerics.timestepping.integrator import Integrator

class SolverSettings():
    def __init__(
            self,
            max_iters: int
        ):
        self.max_iters = max_iters

    @classmethod
    def default(self):
        SolverSettings(
            max_iters = 1000
        )

class Solver():
    def __init__(
            self, 
            settings: SolverSettings,
            integrator: Integrator,
            physics: Physics,
            mesh: Mesh,
            iter: int = 0,
        ):
        self.settings = settings
        self.integrator = integrator
        self.physics = physics
        self.mesh = mesh
        self.iter = iter
    
    @classmethod
    def load_from_checkpoint(filepath: str):
        pass

    def solve(self):
        physics = self.physics
        mesh = self.mesh
        integrator = self.integrator

        while self.iter < self.settings.max_iters:
            integrator.dt = integrator.get_time_step(self)

        
        
