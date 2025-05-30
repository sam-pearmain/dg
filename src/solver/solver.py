import jax.numpy as jnp

from typing import Union, Dict
from physics import Physics
from meshing import Mesh
from logging import Logger
from utils import todo, Uninit, isuninit
from numerics.timestepping.integrator import Integrator

class SolverSettings():
    def __init__(
            self,
            max_iters: int, 
            order: int,
        ):
        self.max_iters = max_iters
        self.order = order

    @classmethod
    def default(cls):
        return cls(
            max_iters = 100,
            order = 1
        )

class Solver():
    settings:   SolverSettings
    integrator: Integrator
    physics:    Physics
    logger:     Logger
    mesh:       Mesh
    iter:       int
    u:          Union[Dict[int, jnp.ndarray], Uninit]

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
        self.u = Uninit
    
    @classmethod
    def load_from_checkpoint(filepath: str):
        todo()

    def initialise_solution(self):
        self.u = {}
        n_state_vars = self.physics.n_state_vars

        if isuninit(self.mesh.element_order):
            raise UninitError

        # initialise the solution to an array that has the same length as the amount of degrees of freedom
        for order

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