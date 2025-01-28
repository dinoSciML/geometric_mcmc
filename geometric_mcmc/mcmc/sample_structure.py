import dolfin as dl
import hippylib as hp
import numpy as np
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
from typing import Any

class SampleDataStrtucture(object):
    """
    Abstract class to define the sample data structure
    """
    def assign_parameter(self, m):
        """
        Assign a parameter sample
        :param m: The parameter sample.
        """
        raise NotImplementedError("assign_parameter method is not implemented.")
    
    def assign(self, other):
        """
        Copy the full data structure from :code:`other` to self
        """
        raise NotImplementedError("assign method is not implemented.")
    

class SampleStruct(SampleDataStrtucture):
    """
    Sample data structure that are used to store and exchange information between Metropolis--Hasting steps
    """
    def __init__(self, model: hp.Model, derivative_info: int=0, hessian_rank=None) -> None:
        """
        :param model: The hippylib model class.
        :param derivativeInfo: The level of derivative information.
        """
        self.derivative_info = derivative_info # 0 (derivative-free), 1 (gradient-based), 2 (stochastic Hessian decomposition)
        self.u = model.generate_vector(STATE) # The state vector
        self.m = model.generate_vector(PARAMETER) # The parameter vector
        self.cost = 0 # The cost function value
        if self.derivative_info >= 1: # If gradient-based
            self.p = model.generate_vector(ADJOINT) # The adjoint vector
            self.g = model.generate_vector(PARAMETER) # The gradient vector
            self.Cg = model.generate_vector(PARAMETER) # The preconditioned gradient vector
        if self.derivative_info == 2: # If stochastic Hessian decomposition
            self.eigenvalues = None # The eigenvalues of the Hessian
            assert hessian_rank is not None, "The rank of the Hessian decomposition must be specified."
            self.decoder = hp.MultiVector(self.m, hessian_rank) # The decoder multivector (eigenvectors of the Hessian, R-orthonormal)
            self.encoder = hp.MultiVector(self.m, hessian_rank) # The encoder multivector (the adjoint of the decoder, D^t R)
        elif not (self.derivative_info == 0 or self.derivative_info ==1): # check for wrong derivative info
            raise ValueError("Derivative requirement for kernel incorrect.")
        self._help1 = model.generate_vector(PARAMETER)
        self._help2 = model.generate_vector(PARAMETER)
        self._noise = dl.Vector(model.prior.R.mpi_comm())
        self.prior = model.prior
        self.prior.init_vector(self._noise, "noise")
    
    def assign_paramaeter(self, m: dl.Vector) -> None:
        self.m.zero()
        self.m.axpy(1., m)

    def assign(self, other: SampleDataStrtucture) -> None:
        if isinstance(other, SampleStruct): # If the other data structure is a full space data structure, assign as usual
            self.cost = other.cost
            self.m = other.m.copy()
            self.u = other.u.copy()
            if self.derivative_info >= 1:
                self.g = other.g.copy()
                self.p = other.p.copy()
                self.Cg = other.Cg.copy()
            if self.derivative_info == 2:
                self.eigenvalues = other.eigenvalues
                for ii in range(self.decoder.nvec()):
                    self.decoder[ii].zero()
                    self.decoder[ii].axpy(1., other.decoder[ii])
                    self.encoder[ii].zero()
                    self.encoder[ii].axpy(1., other.encoder[ii])
        elif isinstance(other, ReducedSampleStruct): # If the other data structure is a reduced space data structure, assign the reduced space data structure to the full space data structure
            assert self.derivative_info == 0, "Cannot assign reduced sample data structure to sample data structure with derivative information."
            self.cost = other.cost
            self._help1.zero()
            self._help2.zero()
            self._noise.zero()
            self.m.zero()
            assert other.decoder is not None, "Assign decoder to reduced sample data structure to go from reduced space to full space." 
            assert other.encoder is not None, "Assign encoder to reduced sample data structure to go from reduced space to full space."
            other.decoder.reduce(self.m, other.m) # Filling the complementary space using the prior sample, encoder, and decoder
            hp.parRandom.normal(1.0, self._noise)
            self.prior.sample(self._noise, self._help1)
            other.decoder.reduce(self._help2, other.encoder.dot_v(self._help1))
            self._help1.axpy(-1., self._help2)
            self.m.axpy(1., self._help1)
            self.u.zero()
        else:
            raise ValueError("Unknow sample data structure type.")

class ReducedSampleStruct(SampleDataStrtucture):
    """
    Reduced sample data structure that are used to store and exchange information between Metropolis--Hasting steps
    """
    def __init__(self, derivative_info: int=0, decoder=None, encoder=None) -> None:
        """
        :param derivativeInfo: The level of derivative information
        :param decoder: The decoder multivector
        :param encoder: The encoder multivector
        """
        self.u = None
        self.m = None
        self.cost = 0
        self.derivative_info = derivative_info
        if self.derivative_info >= 1:
            self.g = None
        if self.derivative_info == 2:
            self.eigenvalues = None
            self.rotation = None
        self.encoder = encoder
        self.decoder = decoder
    
    def assign_paramaeter(self, m: Any) -> None:
        if isinstance(m, dl.Vector):
            if self.encoder is not None:
                self.m = self.encoder.dot(m)
            else:
                raise ValueError("Need encoder to assign full parameter vector for reduced sampling.")
        else:
            self.m = m

    def assign(self, other: SampleDataStrtucture) -> None:
        self.cost = other.cost
        self.u = other.u
        self.m = other.m
        if self.derivative_info >= 1:
            self.g = other.g
        if self.derivative_info == 2:
            self.eigenvalues = other.eigenvalues
            self.rotation = other.rotation