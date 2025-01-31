import dolfin as dl
import hippylib as hp
import numpy as np
from .variables import OBSERVABLE
from .observable import Observable

class ObservableMisfit(hp.Misfit):
    """
    This class implements the misfit functional associated with an observable.
    This class is particularly useful for complex and nonlinear observables that requires independent attention.
    """
    
    def __init__(self, Vh: list[dl.FunctionSpace, ...], observable: Observable, data: np.ndarray=None, noise_precision: np.ndarray = None) -> None:
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
        
    def cost(self, x: list[dl.Vector, ...]) -> float:
        """
        Compute the cost functional.
        :param x: the tuple :code:`[u, m, p]` for state, parameter and adjoint variable
        return the cost functional value
        Note that the adjoint variable :code:`p` is not used and not accessed.
        """
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        temp = self.misfit_vector(x)
        return 0.5*np.inner(temp, self.noise_precision.dot(temp))
    
    def misfit_vector(self, x: list[dl.Vector, ...], weighted: bool=False) -> np.ndarray:
        """
        Compute the misfit vector, optionally weighted by the noise precision.
        :param x: the tuple :code:`[u, m, p]` for state, parameter and adjoint variable
        :param weighted: the flag to weight the misfit vector by the noise precision
        return the misfit vector
        Note that the adjoint variable :code:`p` is not used and not accessed.
        """
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        out = self.observable.eval(x) - self.data # Compute the misfit vector
        if weighted: 
            return self.noise_precision.dot(out) # Weight the misfit vector
        else:
            return out # Return the misfit vector
    
    def grad(self, i: int, x: list[dl.Vector, ...], out: dl.Vector) -> None:
        """
        Compute the gradient of the cost functional.
        :param i: the variable to differentiate with respect to (STATE or PARAMETER)
        :param x: the tuple :code:`[u, m, p]` for state, parameter and adjoint variable
        :param out: the output gradient (:code:`dolfin.Vector`)
        Note that the adjoint variable :code:`p` is not used and not accessed.
        """
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        out.zero()
        temp = self.misfit_vector(x)
        self.observable.jacobian_transpmult(x, i, self.noise_precision.dot(temp), out) # Compute the observable Jacobian transpose action at weighted misfit vector

        # import matplotlib.pyplot as plt

        # cbar = dl.plot(hp.vector2Function(out, self.Vh[i]))
        # plt.colorbar(cbar)
        # plt.show()
        # exit()
                
    def setLinearizationPoint(self, x: list[dl.Vector, ...], gauss_newton_approx: bool=False) -> None:
        """
        Set the linearization point for the misfit functional.
        :param x: the tuple :code:`[u, m, p]` for state, parameter and adjoint variable
        :param gauss_newton_approx: the flag to use the Gauss--Newton approximation
        Note that the adjoint variable :code:`p` is not used and not accessed.
        """
        self.gauss_newton_approx = gauss_newton_approx
        self.observable.setLinearizationPoint(x)
        self.linearization_point = x # Save the linearization point for Hessian computation
        if not self.gauss_newton_approx:
            self.weighted_misfit_vector_at_x = self.misfit_vector(x, True) # Compute the weighted misfit vector at the linearization point

    def apply_ij(self, i: int, j: int, dir: dl.Vector, out: dl.Vector) -> None:
        """
        Apply the Hessian action.
        :param i: the output variable to differentiate with respect to (STATE or PARAMETER)
        :param j: the input variable to differentiate with respect to (STATE or PARAMETER)
        :param dir: the input variation (:code:`dolfin.Vector`)
        :param out: the output variation (:code:`dolfin.Vector`)
        Note that the adjoint variable :code:`p` is not used and not accessed.
        """
        if self.noise_precision is None: 
            raise ValueError("Noise precision must be specified")
        if self.data is None:
            raise ValueError("Data must be specified")
        # Here we compute the positive definite part of the Hessian
        obs_help = self.observable.jacobian_mult(self.linearization_point, j, dir) # Compute the observable Jacobian action at the input variation
        self.observable.jacobian_transpmult(self.linearization_point, i, self.noise_precision.dot(obs_help), out) # Compute the observable Jacobian transpose action at the weighted observable Jacobian action
        if not self.gauss_newton_approx: # If we are not using the Gauss--Newton approximation
            self._help[i].zero() # Zero the help vector
            self.observable.apply_ijk(i, j, OBSERVABLE, dir, self.weighted_misfit_vector_at_x, self._help[i]) # Compute the second variation of the cost functional and apply it to the weighted misfit vector
            out.axpy(1., self._help[i]) # Add the second variation to the Hessian action