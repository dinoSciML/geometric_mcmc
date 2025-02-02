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
    def __init__(self, Vu, targets, components=None, prune_and_sort=False):
        """
        Constructor:
            :code:`B` is the discrete pointwise observation operator in hippylib.
        """
        self.B = hp.assemblePointwiseObservation(Vu, targets, components, prune_and_sort)
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
            raise IndexError("The variable to differentiate with respect to must be either STATE or PARAMETER")
        return out
    
    def jacobian_transpmult(self, x, i, dir, out):
        if i == STATE:
            set_global(self.mpi_comm, dir, self._help_obs)
            self.B.transpmult(self._help_obs, out)
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError("The variable to differentiate with respect to must be either STATE or PARAMETER")
    
    def setLinearizationPoint(self, x):
        pass

    def apply_ijk(self, i, j, k, dir_j, dir_k, out):
        out = np.zeros_like(out) if i == OBSERVABLE else out.zero()

class VariationalQoiObservation(Observable):
    """
    Class to model a qoi scalar observable.    
    """
    def __init__(self, Vh, qoi_varf, bc0 = []):
        """
        Constructor:
            :code:`Vh` is the function space for the state and parameter.
            :code:`qoi_varf` is the qoi variational form
        """
        self.Vh = Vh
        self.mpi_comm = self.Vh[PARAMETER].mesh().mpi_comm()
        self.qoi_varf = qoi_varf
        self._help = [dl.Function(self.Vh[i]).vector() for i in [STATE, PARAMETER]]
        self.L = {}
        if isinstance(bc0, dl.DirichletBC):
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0
    
    def dim(self):
        return 1 # Only consider a scalar qoi. For vector qoi, one may use MultipleObservables
    
    def eval(self, x):
        """
        Given :code:`x` evaluate the cost functional.
        Only the state :code:`u` and (possibly) the parameter :code:`m` are accessed.
        """
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        return np.array([dl.assemble(self.qoi_varf(u,m))])

    def grad(self, i, x, g):
        """
        Given :code:`x` evaluate the gradient of the qoi with respect to the state or the parameter.
        :param x: the tuple :code:`[u, m, p]`
        :param i: the variable to differentiate with respect to (STATE or PARAMETER)
        :param g: the output gradient (:code:`dolfin.Vector`) for the qoi evaluation
        Only the state :code:`u` and (possibly) the parameter :code:`m` are accessed.
        """

        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        g.zero()
        if i == STATE:
            dl.assemble(dl.derivative(self.qoi_varf(u,m), u), tensor=g)
            [bc.apply(g) for bc in self.bc0]
        elif i == PARAMETER:
            dl.assemble(dl.derivative(self.qoi_varf(u,m), m), tensor=g)
        else:
            raise IndexError("The variable to differentiate with respect to must be either STATE or PARAMETER")

    def jacobian_mult(self, x, i, dir):
        """
        Evaluate the Jacoban action with respect to the state or the parameter.
        :param x: the tuple :code:`[u, m, p]`
        :param i: the variable to differentiate with respect to (STATE or PARAMETER)
        :param dir: the input variation (:code:`dolfin.Vector`) for the Jacobian action evaluation
        Note that the adjoint variable is not used in the qoi evaluation.
        """
        self._help[i].zero()
        self.grad(i, x, self._help[i])
        return np.array([self._help[i].inner(dir)])

    def jacobian_transpmult(self, x, i, dir, out):
        """
        Evaluate the Jacoban transpose action with respect to the state or the parameter.
        :param x: the tuple :code:`[u, m, p]`
        :param i: the variable to differentiate with respect to (STATE or PARAMETER)
        :param dir: the input variation (:code:`numpy.ndarray`) for the Jacobian transpose action evaluation
        :param out: the output variation (:code:`dolfin.Vector`) for the Jacobian transpose action evaluation
        Note that the adjoint variable is not used in the qoi evaluation.
        """
        out.zero()
        self._help[i].zero()
        self.grad(i, x, self._help[i])
        out.axpy(dir[0], self._help[i])
    
    def setLinearizationPoint(self, x):
        """
        Set the linearization point.
        :param x: the tuple :code:`[u, m, p]` for state, parameter and adjoint variable
        Note that the adjoint variable is not used in the qoi evaluation.
        """
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        x = [u,m]
        for i in range(2):
            di_form = dl.derivative(self.qoi_varf(*x), x[i])
            for j in range(i,2):
                dij_form = dl.derivative(di_form, x[j])
                self.L[i,j] = dl.assemble(dij_form)

    def apply_ijk(self, i, j, k, dir_j, dir_k, out):
        """
        Apply the second variation :math:`\delta_{ijk}` (:code:`i,j,k = STATE,PARAMETER,OBSERVABLE`) of the cost in direction :code:`dir_j` and :code:`dir_k`.
        """
        if j == OBSERVABLE: # The direction j is with respect to the observable
            if (i, k) in self.L:
                self.L[i, k].mult(dir_k, out)
            else:
                self.L[k, i].transpmult(dir_k, out)
            out *= dir_j[0]
        elif k == OBSERVABLE: # The direction k is with respect to the observable
            if (i, j) in self.L:
                self.L[i, j].mult(dir_j, out) # H_ij
            else:
                self.L[j, i].transpmult(dir_j, out) # H_ji=H_ij^T
            out *= dir_k[0]
        elif i == OBSERVABLE: # The output is with respect to the observable
            self._help[j].zero()
            if (j, k) in self.L:
                self.L[j, k].mult(dir_k, self._help[j]) # H_jk
            else:
                self.L[k, j].transpmult(dir_k, self._help[j]) # H_kj=H_jk^T
            out = np.array([self._help[j].inner(dir_j)]) 
        if i == STATE:
            [bc.apply(out) for bc in self.bc0]

class MultipleObservations(Observable):
    """
    Class to model multiple observables
    """

    def __init__(self, observables: list[Observable, ...]) -> None:
        """
        Constructor:
            :code:`observables` is a list of observables
        """
        self.observables = observables
        self.n_observables = len(observables)
        self._dimension = sum([obs.dim() for obs in observables]) # Total dimension of the observables
        self.mpi_comm = observables[0].mpi_comm
    
    def dim(self):
        return self._dimension
    
    def eval(self, x):
        """
        We loop through the list of observables to compute the stacked observable vector
        """
        out = np.zeros(self._dimension)
        start = 0
        for obs in self.observables:
            dim = obs.dim()
            out[start:(start+dim)] = obs.eval(x)
            start += dim
        return out
    
    def jacobian_mult(self, x, i, dir):
        """
        We loop through the list of observables to compute the Jacobian action.
        """
        out = np.zeros(self._dimension)
        start = 0
        for obs in self.observables:
            dim = obs.dim()
            out[start:(start+dim)] = obs.jacobian_mult(x, i, dir)
            start += dim
        return out
    
    def jacobian_transpmult(self, x, i, dir, out):
        """
        We loop through the list of observables to compute the Jacobian transpose action.
        """
        start = 0
        out.zero()
        self._help = out.copy()
        for obs in self.observables:
            dim = obs.dim()
            self._help.zero()
            obs.jacobian_transpmult(x, i, dir[start:(start+dim)], self._help)
            out.axpy(1., self._help)
            start += dim
    
    def setLinearizationPoint(self, x):
        """
        We loop through the list of observables to set the linearization point
        """
        for obs in self.observables:
            obs.setLinearizationPoint(x)
    
    def apply_ijk(self, i, j, k, dir_j, dir_k, out):
        """
        We loop through the list of observables to apply the second variation :math:`\delta_{ijk}`.
        """
        start = 0
        if i != OBSERVABLE:
            help_out = out.copy()
        for obs in self.observables:
            dim = obs.dim()
            if i == OBSERVABLE: # The output is with respect to the observable
                obs.apply_ijk(i, j, k, dir_j, dir_k, out[start:(start+dim)]) # Apply the second variation
            elif j == OBSERVABLE: # The direction j is with respect to the observable
                help_out.zero()
                obs.apply_ijk(i, j, k, dir_j[start:(start+dim)], dir_k, help_out)
                out.axpy(1., help_out)
            elif k == OBSERVABLE: # The direction k is with respect to the observable
                help_out.zero()
                obs.apply_ijk(i, j, k, dir_j, dir_k[start:(start+dim)], help_out)
                out.axpy(1., help_out)
            start += dim

class TimeDependentObservations(Observable):
    """
    Class to model multiple observables at different times
    """

    def __init__(self, observables: list[Observable, ...], times: np.ndarray) -> None:
        """
        Constructor:
        :param observables: is a list of observables
        :param times: is a numpy array of the times.
        """
        self.observables = observables
        self.times = times
        if isinstance(times, list):
            self.times = np.array(times)
        else: 
            self.times = times
        assert len(observables) == times.size, "The number of observables must be equal to the number of times"
        self._dimension = sum([obs.dim() for obs in self.observables])
        self.mpi_comm = observables[0].mpi_comm
    
    def dim(self):
        return self._dimension

    def eval(self, x):
        out = np.zeros(self._dimension)
        start = 0
        for t, obs in zip(self.times, self.observables): # Loop through the list of observables
            dim = obs.dim()
            out[start:(start+dim)] = obs.eval([x[STATE].view(t), x[PARAMETER]]) # Extact the state at time t for evaluation
            start += dim
        return out
    
    def jacobian_mult(self, x, i, dir):
        out = np.zeros(self._dimension)
        start = 0
        for t, obs in zip(self.times, self.observables):
            dim = obs.dim()
            if i == STATE:
                out[start:(start+dim)] = obs.jacobian_mult([x[STATE].view(t), x[PARAMETER]], i, dir.view(t))
            elif i == PARAMETER:
                out[start:(start+dim)] = obs.jacobian_mult([x[STATE].view(t), x[PARAMETER]], i, dir)
            else:
                raise IndexError("The variable to differentiate with respect to must be either STATE or PARAMETER")
            start += dim
        return out
    
    def jacobian_transpmult(self, x, i, dir, out):
        start = 0
        out.zero()
        for t, obs in zip(self.times, self.observables):
            dim = obs.dim()
            if i == STATE:
                obs.jacobian_transpmult([x[STATE].view(t), x[PARAMETER]], i, dir[start:(start+dim)], out.view(t))
            elif i == PARAMETER:
                obs.jacobian_transpmult([x[STATE].view(t), x[PARAMETER]], i, dir[start:(start+dim)], out)
            else:
                raise IndexError("The variable to differentiate with respect to must be either STATE or PARAMETER")
            start += dim
    
    def setLinearizationPoint(self, x):

        for t, obs in zip(self.times, self.observables):
            obs.setLinearizationPoint([x[STATE].view(t), x[PARAMETER]])
    
    def apply_ijk(self, i, j, k, dir_j, dir_k, out):

        start = 0
        if i == STATE:
            help_out = out.view(self.times[0]).copy()
        elif i != OBSERVABLE:
            help_out = out.copy()

        for t, obs in zip(self.t, self.observables):
            dim = obs.dim()
            if i == OBSERVABLE: # The output is with respect to the observable
                if j == STATE and k == STATE:
                    obs.apply_ijk(i, j, k, dir_j.view(t), dir_k.view(t), out[start:(start+dim)]) # Apply the second variation
                elif j == STATE and k != STATE:
                    obs.apply_ijk(i, j, k, dir_j.view(t), dir_k, out[start:(start+dim)])
                elif j != STATE and k == STATE:
                    obs.apply_ijk(i, j, k, dir_j, dir_k.view(t), out[start:(start+dim)])
                else:
                    obs.apply_ijk(i, j, k, dir_j, dir_k, out[start:(start+dim)])
            if j == OBSERVABLE: # The direction j is with respect to the observable
                if i == STATE and k == STATE:
                    help_out.zero()
                    obs.apply_ijk(i, j, k, dir_j[start:(start+dim)], dir_k.view(t), help_out) # Apply the second variation
                    out.view(t).axpy(1., help_out)
                elif i == STATE and k != STATE:
                    help_out.zero()
                    obs.apply_ijk(i, j, k, dir_j[start:(start+dim)], dir_k, help_out)
                    out.view(t).axpy(1., help_out)
                elif i != STATE and k == STATE:
                    help_out.zero()
                    obs.apply_ijk(i, j, k, dir_j[start:(start+dim)], dir_k.view(t), help_out)
                    out.axpy(1., help_out)
            if k == OBSERVABLE:
                if i == STATE and j == STATE:
                    help_out.zero()
                    obs.apply_ijk(i, j, k, dir_k.view(t),  dir_j[start:(start+dim)], help_out) # Apply the second variation
                    out.view(t).axpy(1., help_out)
                elif i == STATE and j != STATE:
                    help_out.zero()
                    obs.apply_ijk(i, j, k, dir_k, dir_j[start:(start+dim)], help_out)
                    out.view(t).axpy(1., help_out)
                elif i != STATE and j == STATE:
                    help_out.zero()
                    obs.apply_ijk(i, j, k, dir_k.view(t), dir_j[start:(start+dim)], help_out)
                    out.axpy(1., help_out)