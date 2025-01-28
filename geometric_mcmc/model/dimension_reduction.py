#Author: Lianghao Cao
#Date: 2025-01-22
#Brief: This file implements the functions for dimension reduction. 
#In particular, it implements the derivative-informed subspace (DIS) and the Karhunen--Loeve expansion (KLE) of the prior.

import numpy as np
import dolfin as dl
import hippylib as hp
from scipy.linalg import eigh
from .pto_map import PtOMapJacobian
from ..utilities.reduced_basis import check_orthonormality
from mpi4py import MPI

class ExpectedGaussNewtonHessian:
    """
    This class implements the action of the expected Gauss--Newton Hessian operator.
    """
    def __init__(self, jacobian_samples: list[hp.MultiVector], noise_precision: np.ndarray) -> None:
        """
        Constructor:
            :param jacobian_samples: The list of Jacobian samples
            :param noise_precision: The noise precision matrix
        """
        self.jacobian_samples = jacobian_samples
        self.noise_precision = noise_precision
        self.n_samples = len(jacobian_samples)
    def mult(self, x: dl.Vector, y: dl.Vector) -> None:
        """
        Apply the operator to a vector x and assign to y. 
        """
        y.zero()
        for ii, jacobian in enumerate(self.jacobian_samples):
            jacobian.reduce(y, self.noise_precision.dot(jacobian.dot_v(x)))

    def transpmult(self, x, y):
        """
        Apply the transpose of the operator to a vector x and assign to y. Same as mult due to symmetry.
        """
        self.mult(x, y)


def compute_DIS_from_samples(model: hp.Model, jacobian_samples: list[hp.MultiVector,...],
                              input_rank: int, output_rank: int, 
                              oversampling: int=20) -> tuple[np.ndarray, hp.MultiVector, hp.MultiVector]:
    """
    Compute the derivative-informed subspace from samples of the Jacobian matrix
    :param model: The hippylib model class with ObservableMisfit
    :param jacobian_samples: The list of Jacobian samples
    :param input_rank: The rank of the input dimension reduction
    :param output_rank: The rank of the output dimension reduction
    return (input_eigenvalues, input_decoder, input_encoder), (output_eigenvalues, output_decoder, output_encoder)
    """
    # Check the input and output ranks
    assert input_rank <= jacobian_samples[0][0].size()
    assert output_rank <= jacobian_samples[0].nvec()

    n_samples = len(jacobian_samples) # The number of samples
    prior = model.prior # The prior
    noise_precision = model.misfit.noise_precision # The noise precision matrix

    # Compute the eigendecomposition of the expected Gauss--Newton Hessian using a randomized generalized eigensolver
    exp_GNH = ExpectedGaussNewtonHessian(jacobian_samples, noise_precision) # The expected Gauss--Newton Hessian operator
    Omega = hp.MultiVector(model.generate_vector(hp.PARAMETER), input_rank + oversampling) # Initialize the random matrix
    hp.parRandom.normal(1., Omega) # Fill the random matrix with standard normal random variables
    input_eigenvalues, input_decoder = hp.doublePassG(exp_GNH, prior.R, prior.Rsolver, Omega, input_rank) # Compute the eigendecomposition
    input_encoder = hp.MultiVector(input_decoder[0], input_decoder.nvec()) # Initialize the encoder
    hp.MatMvMult(prior.R, input_decoder, input_encoder) # Compute the encoder
    check_orthonormality(input_decoder, input_encoder) # Check the orthonormality of the basis

    # Compute the eigendecomposition of the inside out Gauss--Newton Hessian using scipy generalized eigensolver
    Rinv_jacobian_samples = [hp.MultiVector(jacobian[0], jacobian.nvec()) for jacobian in jacobian_samples] # Initialize the prior-preconditioned Jacobian samples
    Rinv_operator = hp.Solver2Operator(prior.Rsolver) # Initialize the prior-preconditioned operator
    for ii, jacobian in enumerate(jacobian_samples): # Compute the prior-preconditioned Jacobian samples
        hp.MatMvMult(Rinv_operator, jacobian, Rinv_jacobian_samples[ii])
    JCJt = [jacobian.dot_mv(R_inv_jacobian) for jacobian, R_inv_jacobian in zip(jacobian_samples, Rinv_jacobian_samples)] # Compute the prior-preconditioned inside out Gauss--Newton Hessian
    expected_mat = sum([1./n_samples*noise_precision@mat@noise_precision for mat in JCJt]) # Pad the Gauss--Newton Hessian on the left and right with noise precision
    output_eigenvalues, output_decoder = eigh(expected_mat, b=noise_precision) # Solve the generalized eigenvalue problem
    output_eigenvalues = output_eigenvalues[::-1] # Reverse the eigenvalues, descending order
    output_decoder = output_decoder[:, ::-1] # Reverse the eigenvectors too
    output_decoder = output_decoder[:, :output_rank] # Truncate the eigenvectors
    output_encoder = output_decoder.T@noise_precision # Compute the encoder
    assert  np.allclose(output_encoder@output_decoder, np.eye(output_rank)) # Check the orthonormality of the basis

    return (input_eigenvalues, input_decoder, input_encoder), (output_eigenvalues[:output_rank], output_decoder, output_encoder)

def compute_DIS(model:hp.Model, n_samples:int, input_rank:int, output_rank: int, 
                burn_in: int=0, oversampling: int=20, mode: str="reverse",
                return_observables: bool = False) -> tuple[np.ndarray, hp.MultiVector, hp.MultiVector]:
    """
    Compute the derivative-informed subspace
    :param comm_sampler: The MPI communicator for the sampler
    :param model: The hippylib model class with ObservableMisfit
    :param n_samples: The number of samples for subspace estimation
    :param input_rank: The rank of the input dimension reduction
    :param output_rank: The rank of the output dimension reduction
    :param burn_in: The number of burn-in samples
    :param oversampling: The oversampling factor for the randomized eigensolver
    :param mode: The mode of the Jacobian computation, must be either "forward" or "reverse"
    :param return_observables: Whether to return the observable samples
    return (input_eigenvalues, input_decoder, input_encoder), (output_eigenvalues, output_decoder, output_encoder)
    """
    pto_map = PtOMapJacobian(model.problem, model.misfit.observable) # initialize the parmeter-to-observable map
    x = model.generate_vector() # The state, parameter, and adjoint tuple
    noise = dl.Vector(model.prior.R.mpi_comm()) # The noise vector
    model.prior.init_vector(noise, "noise") # Initialize the noise vector
    jacobian_samples = [] # The list of Jacobian samples
    if return_observables: observables = []
    for ii in range(burn_in): # Burn-in samples
        hp.parRandom.normal(1.0, noise)
    for ii in range(n_samples): # Generate the samples
        hp.parRandom.normal(1.0, noise) # Generate the noise
        model.prior.sample(noise, x[hp.PARAMETER]) # Sample the parameter
        model.solveFwd(x[hp.STATE], x) # Solve the forward problem
        if return_observables: observables.append(model.misfit.observable.eval(x))
        pto_map.setLinearizationPoint(x) # Set the linearization point
        jacobian_samples.append(pto_map.generate_jacobian()) # Generate a empty Jacobian
        pto_map.eval(jacobian_samples[-1], "reverse") # Compute the Jacobian at the linearization point
    
    input_res, output_res = compute_DIS_from_samples(model, jacobian_samples, input_rank, output_rank, oversampling) # Compute the derivative-informed subspace from the generated samples
    if return_observables:
        return input_res, output_res, np.vstack(observables)
    else:
        return input_res, output_res
    
def compute_KLE(prior: hp.prior, rank: int) -> tuple[np.ndarray, hp.MultiVector, hp.MultiVector]:
    """
    Compute the Karhunen--Loeve expansion of the prior. 
    Note that by default the basis is C^{-1} orthonormal, where C is the prior covariance. THIS IS DIFFERENT FROM THE USUAL KLE!!!!
    :param prior: A hippylib Gaussian random function
    :param rank: The number of expansion
    :return: The eigenvalues, the decoder and the encoder
    """
    kle_constructor = KLESubspaceConstructorSLEPc(prior) # Initialize the KLE subspace constructor
    eigenvalues, decoder = kle_constructor.compute_kle_subspace(rank) # Compute the KLE subspace
    encoder = hp.MultiVector(decoder[0], decoder.nvec()) # Initialize the encoder
    hp.MatMvMult(prior.R, decoder, encoder) # Compute the encoder

    return eigenvalues, decoder, encoder

class KLESubspaceConstructorSLEPc:
    """
    KLE subspace computation with Cameron--Martin space orthnormality.
    """
    def __init__(self, prior):
        assert hasattr(prior, "R")
        assert hasattr(prior, "A")
        assert hasattr(prior, "M")
        self._prior = prior
        self.R = self._prior.R
        self.Vh = self._prior.Vh
        self.A = dl.as_backend_type(self._prior.A)
        self.M = dl.as_backend_type(self._prior.M)
        self.eigensolver = dl.SLEPcEigenSolver(self.A, self.M)
        self.eigensolver.parameters["solver"] = "krylov-schur"
        self.eigensolver.parameters["problem_type"] = "gen_hermitian"
        self.eigensolver.parameters["spectrum"] = "target magnitude"
        self.eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        self.eigensolver.parameters["spectral_shift"] = 0.0

        self.m = dl.Function(self.Vh).vector()

    def mpi_comm(self):
        return self.R.mpi_comm()

    def compute_kle_subspace(self, rank):
        """
        Compute the KLE basis using :code:`dl.SLEPcEigenSolver`
        :param rank: number of eigenpairs
        """
        self.eigensolver.solve(rank)
        sqrt_precision_eigenvalues = np.zeros(rank)
        kle_basis = hp.MultiVector(self.m, rank)
        for ii in range(rank):
            sqrt_precision_eigenvalues[ii], _, basis_i, _ = self.eigensolver.get_eigenpair(ii)
            kle_basis[ii].zero()
            kle_basis[ii].axpy(1.0/sqrt_precision_eigenvalues[ii], basis_i)
        covariance_eigenvalues = 1/sqrt_precision_eigenvalues**2
        return covariance_eigenvalues, kle_basis