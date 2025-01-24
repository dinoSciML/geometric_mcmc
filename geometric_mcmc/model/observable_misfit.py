import dolfin as dl
import hippylib as hp
import numpy as np
from .variables import OBSERVABLE

class ObservableMisfit(hp.Misfit):
    
    def __init__(self, Vh, observable, data=None, noise_precision = None):
        """
        Constructor:
            :code:`Vh` is the tuple of function spaces for the state, parameter and adjoint variables
            :code:`observable` is an observable object
            :code:`data` is the data (:code:`numpy.ndarray`)
            :code:`noise_precision` is the noise precision matrix (:code:`numpy.ndarray` or :code:`scipy.sparse`)
        """
        self.observable = observable
        self.Vh = Vh
        self.data = data
        self.noise_precision = noise_precision
        self._help = [dl.Function(Vh[ii]).vector() for ii in range(2)]
        
    def cost(self, x):
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        temp = self.misfit_vector(x)
        return 0.5*np.inner(temp, self.noise_precision.dot(temp))
    
    def misfit_vector(self, x, weighted=False):
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        out = self.observable.eval(x) - self.data
        if weighted:
            return self.noise_precision.dot(out)
        else:
            return out
    
    def grad(self, i, x, out):
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        out.zero()
        temp = self.misfit_vector(x)
        self.observable.jacobian_transpmult(x, i, self.noise_precision.dot(temp), out)
                
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        self.gauss_newton_approx = gauss_newton_approx
        self.observable.setLinearizationPoint(x)
        self.linearization_point = x
        if not self.gauss_newton_approx:
            self.weighted_misfit_vector_at_x = self.misfit_vector(x, True)

       
    def apply_ij(self, i, j, dir, out):
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        obs_help = self.observable.jacobian_mult(self.linearization_point, j, dir)
        self.observable.jacobian_transpmult(self.linearization_point, i, self.noise_precision.dot(obs_help), out)
        if not self.gauss_newton_approx:
            self._help[i].zero()
            self.observable.apply_ijk(i, j, OBSERVABLE, dir, self.weighted_misfit_vector_at_x, self._help[i])
            out.axpy(1., self._help[i])