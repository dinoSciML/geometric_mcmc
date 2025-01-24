import dolfin as dl
import hippylib as hp
import numpy as np
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
from .variables import OBSERVABLE
from ..utilities.collective import set_global, get_global

class Observable(object):
    """
    Abstract class to model the observable.
    In the following :code:`x` will denote the variable :code:`[u, m, p]`, denoting respectively
    the state :code:`u`, the parameter :code:`m`, and the adjoint variable :code:`p`.
    
    The methods in the class Observable will usually access the state :code:`u` and possibly the
    parameter :code:`m`. 
    """

    def dim(self):
        """
        Return the dimension of the observable.
        """
        raise NotImplementedError("Child class should implement method dim")
    
    def eval(self, x):
        """
        Given :code:`x` evaluate the cost functional.
        Only the state :code:`u` and (possibly) the parameter :code:`m` are accessed.
        """
        raise NotImplementedError("Child class should implement method eval")

    def jacobian_mult(self, x, i, dir):
        """
        Evaluate the Jacoban action with respect to the state or the parameter.
        :param x: the tuple :code:`[u, m, p]`
        :param i: the variable to differentiate with respect to (STATE or PARAMETER)
        :param dir: the input variation (:code:`dolfin.Vector`) for the Jacobian action evaluation
        return the output variation (:code:`numpy.ndarray`) for the Jacobian action evaluation
        Only the state :code:`u` and (possibly) the parameter :code:`m` are accessed. 
        """
        raise NotImplementedError("Child class should implement method jacobian_mult")

    def jacobian_transpmult(self, x, j, dir, out):
        """
        Evaluate the Jacoban transpose action with respect to the state or the parameter.
        :param x: the tuple :code:`[u, m, p]`
        :param i: the variable to differentiate with respect to (STATE or PARAMETER)
        :param dir: the input variation (:code:`numpy.ndarray`) for the Jacobian transpose action evaluation
        :param out: the output variation (:code:`dolfin.Vector`) for the Jacobian transpose action evaluation
        Only the state :code:`u` and (possibly) the parameter :code:`m` are accessed. 
        """
        raise NotImplementedError("Child class should implement method grad")

    def setLinearizationPoint(self, x):
        """
        Set the linearization point.
        :param x: the tuple :code:`[u, m, p]` for state, parameter and adjoint variable
        Only the state :code:`u` and (possibly) the parameter :code:`m` are accessed. 
        """
        raise NotImplementedError("Child class should implement method setLinearizationPoint")

    def apply_ijk(self, i, j, k, dir_j, dir_k, out):
        """
        Apply the second variation :math:`\delta_{ijk}` (:code:`i,j,k = STATE,PARAMETER,OBSERVABLE`) of the cost in direction :code:`dir_j` and :code:`dir_k`.
        """
        raise NotImplementedError("Child class should implement method apply_ijk")


class PointwiseObservation(Observable):
    """
    Class to model a pointwise observation operator.    
    """
    def __init__(self, Vu, targets):
        """
        Constructor:
            :code:`B` is the discrete pointwise observation operator in hippylib.
        """
        self.B = hp.assemblePointwiseObservation(Vu, targets)
        self.mpi_comm = self.B.mpi_comm()
        self._help_obs = dl.Vector(self.mpi_comm)
        self.B.init_vector(self._help_obs, 0)
        self._dimension = self._help_obs.size()
    
    def dim(self):
        return self._dimension

    def eval(self, x):
        self.B.mult(x[STATE], self._help_obs)
        return get_global(self.mpi_comm, self._help_obs)
    
    def jacobian_mult(self, x, i, dir):
        if i == STATE:
            self.B.mult(dir, self._help_obs)
            out = get_global(self.mpi_comm, self._help_obs)
        elif i == PARAMETER:
            out = np.zeros(self._dimension)
        else:
            raise ValueError("The variable to differentiate with respect to must be either STATE or PARAMETER")
        return out
    
    def jacobian_transpmult(self, x, i, dir, out):
        if i == STATE:
            set_global(self.mpi_comm, dir, self._help_obs)
            self.B.transpmult(self._help_obs, out)
        elif i == PARAMETER:
            out.zero()
        else:
            raise ValueError("The variable to differentiate with respect to must be either STATE or PARAMETER")
    
    def setLinearizationPoint(self, x):
        pass

    def apply_ijk(self, i, j, k, dir_j, dir_k, out):
        out = np.zeros_like(out) if i == OBSERVABLE else out.zero()


def QoiObservable(Observable):
    """
    Class to model a pointwise observation operator.    
    """
    def __init__(self, Vh, qoi_list):
        """
        Constructor:
            :code:`B` is the discrete pointwise observation operator in hippylib.
        """
        self.mpi_comm = Vh[STATE].mesh().mpi_comm()
        self.qoi_list = qoi_list
        self.n_qoi = len(qoi_list)
    
    def dim(self):
        return self.n_qoi
    
    def eval(self, x):
        out = np.zeros(self.n_qoi)
        for ii in range(self.n_qoi):
            out[ii] = self.qoi_list[ii].eval([x[STATE], x[PARAMETER]])
        return out