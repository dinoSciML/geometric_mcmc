# Author: Lianghao Cao
# Date: 2025-01-22
# Brief: This file implements the parameter-to-observable map. 
# Almost the entire script is copied from the hippylib library and modified to fit the needs of the geometric MCMC project.
# Please also cite the following paper if you use this code:
# https://doi.org/10.1145/3428447

import hippylib as hp
import dolfin as dl
import numpy as np
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
from ..utilities.collective import set_global

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
    
    def setLinearizationPoint(self, x):
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

    def generate_jacobian(self):
        return hp.MultiVector(self.yhelp, self.observable.dim())
    
    def eval(self, jacobian_mv, mode="reverse"):
        if mode == "forward":
            out = np.zeros((self.observable.dim(), self.yhelp.size()))
            for ii in range(self.yhelp.size()):
                unit_vec = np.zeros(self.yhelp.size())
                unit_vec[ii] = 1.
                set_global(self.mpi_comm, unit_vec, self.yhelp)
                out[:, ii] = self.mult(self.yhelp)
            for ii in range(jacobian_mv.nvec()):
                set_global(self.mpi_comm, out[ii, :], jacobian_mv[ii])
        elif mode == "reverse":
            for ii in range(self.observable.dim()):
                jacobian_mv[ii].zero()
                unit_vec = np.zeros(self.observable.dim())
                unit_vec[ii] = 1.
                self.transpmult(unit_vec, jacobian_mv[ii])
        else:
            raise ValueError("Differentiation mode not recognized. Must be either `forward' or `reverse'.")

    def mult(self, x):
        self.problem.apply_ij(ADJOINT,PARAMETER, x, self.rhs_fwd)
        self.problem.solveIncremental(self.uhat, self.rhs_fwd, False)
        d_u_hat = self.observable.jacobian_mult(self.linearization_point, STATE, self.uhat)
        d_m_hat = self.observable.jacobian_mult(self.linearization_point, PARAMETER, x)
        return -d_u_hat - d_m_hat

    def transpmult(self, x, y):
        y.zero()
        self.rhs_adj.zero()
        self.observable.jacobian_transpmult(self.linearization_point, STATE, x, self.rhs_adj)
        self.problem.solveIncremental(self.phat, self.rhs_adj, True)
        self.problem.apply_ij(PARAMETER,ADJOINT, self.phat, self.yhelp)
        y.axpy(-1., self.yhelp)
        self.yhelp.zero()
        self.observable.jacobian_transpmult(self.linearization_point, PARAMETER, x, self.yhelp)
        y.axpy(-1., self.yhelp)