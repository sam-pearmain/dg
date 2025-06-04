import jax.numpy as jnp

from typing import Union, Dict
from logging import Logger
from physics import Physics
from meshing import Mesh
from utils import *
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
        todo("need also a method to write solution to a checkpoint file")

    def initialise_solution(self):
        self.u = {}
        mesh = self.mesh
        n_state_vars = self.physics.n_state_vars

        if isuninit(mesh.element_order):
            raise UninitError

        # initialise the solution to an array that has the same length as the amount of degrees of freedom
        for order, element_ids in mesh.element_order.items():
            n_elements = element_ids.shape[0]

            if n_elements == 0:
                continue

            n_dofs = mesh.element_type.n_dofs(order)
            self.u[order] = jnp.zeros(
                (n_elements, n_state_vars, n_dofs),
                dtype = jnp.float64
            )

    def run(self):
        integrator = self.integrator

        while self.iter < self.settings.max_iters:
            """This is the mainloop of the solver"""
            # update the time step size
            integrator.update_time_step(self)

            # integrate in time
            integrator.step(self)

            # increment the number of iterations
            self.iter += 1

    def compute_spatial_residual(self):
        physics = self.physics

    @property
    def time_step(self):
        return self.integrator.time_step