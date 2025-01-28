# Author: Lianghao Cao
# Date: 2025-01-22
# Brief: This file implements the parameter-to-observable map. 
# This script is mostly take from hippylib library and modified extensively to fit the needs of the geometric MCMC project.
# Please also cite the following paper if you use this code:
# https://doi.org/10.1145/3428447

import hippylib as hp
import dolfin as dl
import numpy as np
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
from ..utilities.collective import set_global, set_global_mv

class PtOMapJacobian():
    """
    This class implements the Frechet derivative of a parameter-to-observable map.
    This particular implementation assume pointwise observations following the hippylib conventions.
    """
    def __init__(self, problem, observable) -> None:
        self.mpi_comm = problem.Vh[PARAMETER].mesh().mpi_comm()
        self.rhs_fwd = problem.generate_state()
        self.rhs_adj = problem.generate_state()
        self.uhat = problem.generate_state()
        self.phat = problem.generate_state()
        self.yhelp = problem.generate_parameter()
        self.problem = problem
        self.observable = observable
    
    def setLinearizationPoint(self, x: list[dl.Vector]) -> None:
        """"
        Set the linearization point.
        :param x: the tuple :code:`[u, m, p]` for state, parameter and adjoint variable
        the state :code:`u` needs to be the PDE solution at the parameter :code:`m`
        the adjoint variable :code:`p` does not need to be set
        """
        if len(x) == 2:
            x.append(self.problem.generate_state())
        if x[ADJOINT] is None:
            x[ADJOINT] = self.problem.generate_state()
        self.problem.setLinearizationPoint(x, gauss_newton_approx=True)
        self.observable.setLinearizationPoint(x)
        self.linearization_point = x

    def generate_jacobian(self) -> hp.MultiVector:
        """
        Generate an empty Jacobian matrix.
        """
        return hp.MultiVector(self.yhelp, self.observable.dim())
    
    def eval(self, jacobian_mv: hp.MultiVector, mode: str="reverse") -> None:
        """
        Evaluate the Jacobian action.
        :param jacobian_mv: the Jacobian as hp.MultiVector
        :param mode: the differentiation mode, must be either "forward" or "reverse"
        """
        if mode == "forward": # Forward mode differentiation
            out = np.zeros((self.observable.dim(), self.yhelp.size())) # Initialize the output
            for ii in range(self.yhelp.size()): # Loop over the parameter dimension
                unit_vec = np.zeros(self.yhelp.size()) # Initialize the unit vector
                unit_vec[ii] = 1. # Set the unit vector
                set_global(self.mpi_comm, unit_vec, self.yhelp) # Set the unit vector to a dolfin vector
                out[:, ii] = self.mult(self.yhelp) # Compute the Jacobian action
            set_global_mv(self.mpi_comm, out[ii, :], jacobian_mv[ii])
        elif mode == "reverse": # Reverse mode differentiation
            for ii in range(self.observable.dim()): # Loop over the observable dimension
                jacobian_mv[ii].zero() # Zero the output
                unit_vec = np.zeros(self.observable.dim()) # Initialize the unit vector
                unit_vec[ii] = 1. # Set the unit vector
                self.transpmult(unit_vec, jacobian_mv[ii]) # Compute the Jacobian transpose action
        else:
            raise ValueError("Differentiation mode not recognized. Must be either `forward' or `reverse'.")

    def mult(self, x: dl.Vector) -> np.ndarray:
        """
        Compute the Jacobian action.
        :param x: the input parameter variation as dl.Vector
        :return: the output observable variation as np.ndarray
        """
        self.rhs_fwd.zero()
        self.problem.apply_ij(ADJOINT,PARAMETER, x, self.rhs_fwd) # Compute the right-hand for incremental forward
        self.problem.solveIncremental(self.uhat, self.rhs_fwd, False) # Solve the incremental forward
        d_u_hat = self.observable.jacobian_mult(self.linearization_point, STATE, self.uhat) # Compute the observable state jacobian action at the incremental solution
        d_m_hat = self.observable.jacobian_mult(self.linearization_point, PARAMETER, x)  # Compute the observable parameter jacobian action at the parameter input
        return -d_u_hat - d_m_hat # Return the negative sum of the two actions

    def transpmult(self, x: np.ndarray, y: dl.Vector) -> None:
        """
        Compute the Jacobian transpose action.
        :param x: the input observable variation as np.ndarray
        :param y: the output parameter variation as dl.Vector
        """
        y.zero() # Zero the output
        self.rhs_adj.zero() # Zero the adjoint right-hand side
        self.observable.jacobian_transpmult(self.linearization_point, STATE, x, self.rhs_adj) # Compute the observable state jacobian transpose action at the observable input
        self.problem.solveIncremental(self.phat, self.rhs_adj, True) # Solve the incremental adjoint
        self.problem.apply_ij(PARAMETER,ADJOINT, self.phat, self.yhelp) # Compute the varaition in the parameter
        y.axpy(-1., self.yhelp) # Add the negative variation
        self.yhelp.zero() # Zero the help vector
        self.observable.jacobian_transpmult(self.linearization_point, PARAMETER, x, self.yhelp) # Compute the observable parameter jacobian transpose action at the observable input
        y.axpy(-1., self.yhelp) # Add the negative variation