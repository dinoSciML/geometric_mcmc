import hippylib as hp
import dolfin as dl
import math
import numpy as np
import matplotlib.pyplot as plt
from .sample_structure import SampleStruct, ReducedSampleStruct
from .kernel_utilities import decomposeGaussNewtonHessian
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
from ..utilities.reduced_basis import check_orthonormality
from ..model.pto_map import PtOMapJacobian
from ..model.surrogate_wrapper import ReducedBasisSurrogate
from scipy.linalg import eigh

class Kernel(object):
    """
        Abstract class for the metropolis--Hasting kernel in MCMC
    """

    def derivativeInfo(self) -> int:
        """
        Return the level of derivative information of the kernel
        """
        raise NotImplementedError("Child class must implement method 'derivativeInfo'")

    def reduced(self) -> bool:
        """
        Return True if the kernel operates in a finite dimension reduce parameter vector space, False otherwise
        """
        raise NotImplementedError("Child class must implement method 'reduced'")

    def generate_sample(self) -> SampleStruct:
        """
        Generate a sample data structure
        """
        raise NotImplementedError("Child class must implement method 'generate_sample'")

    def init_sample(self, s: SampleStruct) -> None:
        """
        Initialize the sample structure by computing the cost (derivativeInfo=0) or gradients (derivativeInfo=1) or Hessian decomposition (derivativeInfo=2).
        :param s: The sample structure
        """
        raise NotImplementedError("Child class must implement method 'init_sample'")

    def sample(self, current: SampleStruct, proposed: SampleStruct) -> None:
        """
        Sample a new proposal given the current sample
        :param current: The current sample
        :param proposed: The proposed sample
        """
        raise NotImplementedError("Child class must implement method 'sample'")

    def accept_or_reject(self, current: SampleStruct, proposed: SampleStruct) -> int:
        """
        Accept or reject the proposed sample
        :param current: The current sample
        :param proposed: The proposed sample
        """
        raise NotImplementedError("Child class must implement method 'accept_or_reject'")
    
    def consume_random(self) -> None:
        """
        Consume random numbers for each Metroplis--Hasting step
        """
        raise NotImplementedError("Child class must implement method 'consume_random'")

class mMALAKernel(Kernel):
    """
        This class implement the kernel class of Reimannian Manifold (state dependent local reference measure) MALA.
        Reference:
        `A. Beskos, M. Girolami, S. Lan, P. E. Farrell, A. M. Stuart (2017).
         Geometric MCMC for infinite-dimensional inverse problems.
         Journal of Computational Physics, 335, 327–351. https://doi.org/10.1016/j.jcp.2016.12.041`
    """

    def __init__(self, model: hp.Model, form_jacobian=True, mode: str = "reverse") -> None:
        """
        Construction:
        :param model: The hIPPYlib model class for bayesian inverse problems
        :param form_jacobian: Whether to form the Jacobian matrix
        :param derivative: The derivative class for the parameter-to-observable map
        :param mode: The mode of differentiation. Must be forward or reverse
        """
        self.model = model
        if not hasattr(self.model.misfit, "observable"):
            raise Exception("Please use the ObservableMisfit class.")
        
        self._noise = dl.Vector(self.model.prior.R.mpi_comm())
        self.model.prior.init_vector(self._noise, "noise")
        self._help1, self._help2 = dl.Vector(self.model.prior.R.mpi_comm()), dl.Vector(self.model.prior.R.mpi_comm())
        self.model.prior.init_vector(self._help1, 1)
        self.model.prior.init_vector(self._help2, 1)

        self.parameters = {}
        self.parameters["h"] = 0.1
        self.parameters["oversampling"] = 20
        self.form_jacobian = form_jacobian
        if self.form_jacobian:
            self.parameters["hessian_rank"] = min(self.model.misfit.observable.dim(), self._help1.size())
            self.parameters["gauss_newton_approximation"] = True
            self._Rinv_operator = hp.Solver2Operator(self.model.prior.Rsolver)
            self.pto_map_jac = PtOMapJacobian(self.model.problem, self.model.misfit.observable)
            self.jac_mode = mode
        else:
            self.mode = mode
            self.parameters["hessian_rank"] = None # The rank of the Hessian approximation must be set by the user
            self.parameters["gauss_newton_approximation"] = False

    def name(self) -> str:
        """
        Return the name of the kernel
        """
        return "mMALA"

    def derivativeInfo(self) -> int:
        """
        Return the derivative information of the kernel
        """
        return 2

    def reduced(self) -> bool:
        """
        Return True if the kernel is reduced, False otherwise
        """
        return False
    
    def generate_sample(self) -> SampleStruct:
        """
        Generate a sample data structure
        """
        return SampleStruct(self.model, self.derivativeInfo(), self.parameters["hessian_rank"])

    def init_sample(self, s: SampleStruct) -> None:
        """
        Initialize the sample structure by computing the cost, gradient, eigenvalues and eigenbases.
        :param s: The sample structure
        """
        self.model.solveFwd(s.u, [s.u, s.m, None]) # solve the forward problem
        s.cost = self.model.cost([s.u, s.m, None])[2] # compute the misfit value
        if self.form_jacobian:
            J = self.pto_map_jac.generate_jacobian() # generate the Jacobian multivector
            Rinv_J = self.pto_map_jac.generate_jacobian() # compute the prior preconditioned Jacobian
            self.pto_map_jac.setLinearizationPoint([s.u, s.m, None]) # set the linearization point
            self.pto_map_jac.eval(J, mode = self.jac_mode) # compute the Jacobian
            hp.MatMvMult(self._Rinv_operator, J, Rinv_J) # compute the encoder
            weighted_misfit_vector = self.model.misfit.misfit_vector([s.u, s.m, None], True) # compute the weighted misfit vector
            s.g.zero()
            J.reduce(s.g, weighted_misfit_vector) # compute the gradient
            s.Cg.zero()
            Rinv_J.reduce(s.Cg, weighted_misfit_vector) # compute the preconditioned gradient
            s.eigenvalues = decomposeGaussNewtonHessian(J, Rinv_J, s.decoder, s.encoder, self.model.prior, self.model.misfit.noise_precision, oversampling=self.parameters["oversampling"]) # compute the eigenvalues and eigenvectors
        else: # use randomized eigendecomposition of the Hessian using a matrix free approach
            self.model.solveAdj(s.p, [s.u, s.m, s.p]) # solve the adjoint problem
            self.model.setPointForHessianEvaluations([s.u, s.m, s.p], gauss_newton_approx=self.parameters["gauss_Newton_approximation"]) # set the point for Hessian evaluations
            Hmisfit = hp.ReducedHessian(self.model, misfit_only=True) # compute the Hessian approximation
            assert self.parameters["hessian_rank"] is not None, "The rank of the Hessian approximation must be set."
            Omega = hp.MultiVector(s.m, self.parameters["hessian_rank"] + self.parameters["oversampling"]) # initialize the multivector
            hp.parRandom.normal(1., Omega)
            s.decoder = None
            s.eigenvalues, s.decoder = hp.doublePassG(Hmisfit, self.model.prior.R, self.model.prior.Rsolver, Omega, self.parameters["hessian_rank"]) # compute the eigenvalues and eigenvectors
            hp.MatMvMult(self.model.prior.R, s.decoder, s.encoder) # compute the encoder

    def sample(self, current: SampleStruct, proposed: SampleStruct) -> None:
        """
        Sample a new proposal given the current sample
        :param current: The current sample
        :param proposed: The proposed sample
        """
        self.proposal(current, proposed.m)  # Coming up with a proposal step

    def accept_or_reject(self, current: SampleStruct, proposed: SampleStruct) -> int:
        """
        Accept or reject the proposed sample
        :param current: The current sample
        :param proposed: The proposed sample
        return: 1 if the proposed sample is accepted, 0 otherwise
        """
        rate_pc = self.acceptance_ratio(proposed, current)  # log transition rate from proposal to current
        rate_cp = self.acceptance_ratio(current, proposed)  # log transition rate from current to proposal
        al = np.sum(rate_pc - rate_cp)  # log ratio of transition rate
        if (al > math.log(np.random.rand())):  # Metropolis--Hasting acceptance--rejection step
            return 1
        else:
            return 0

    def proposal(self, current: SampleStruct, out: dl.Vector) -> None:
        """
        Compute the proposal step given the current sample
        :param current: The current sample
        :param out: The output vector, i.e., the proposed sample
        """
        h = self.parameters["h"] # step size
        rho = (4.0-h)/(4.0+h) # the scaling factor
        hp.parRandom.normal(1., self._noise)  # sample from i.i.d. random noise vector
        s_prior = dl.Vector(self.model.prior.R.mpi_comm()) # initialize prior sample
        s_post = dl.Vector(self.model.prior.R.mpi_comm())  # initialize proposal sample
        self.model.prior.init_vector(s_prior, 0) # initialize the prior sample
        self.model.prior.init_vector(s_post, 0) # initialize the proposal sample
        self.model.prior.sample(self._noise, s_prior, add_mean=False)  # sample from the prior distribution
        # compute the diagonal scaling for sample transformation
        invsqrt1plusdmin1 = -1.0 + 1. / (np.sqrt(1.0 + current.eigenvalues))
        VtRs = current.encoder.dot_v(s_prior)  # encode the prior samples
        s_post.zero()
        current.decoder.reduce(s_post, invsqrt1plusdmin1 * VtRs)  # decode the prior samples
        s_post.axpy(1., s_prior)
        self.compute_local_gradient(current, self._help1)  # compute the local gradient vector
        # Compute the proposed sample!
        out.zero()
        out.axpy(rho, current.m)
        out.axpy(1.0 - rho, self._help1)
        out.axpy(math.sqrt(1.0 - rho ** 2), s_post)

    def acceptance_ratio(self, origin: SampleStruct, destination: SampleStruct) -> list[float, ...]:
        """
        Compute the acceptance ratio given the origin and destination samples
        :param origin: The origin sample
        :param destination: The destination sample
        return: A list of terms in the acceptance ratio
        """
        rate = self.density_wrt_pcn(origin, destination)  # Compute the RN derivative with pCN
        rate[-1] = -origin.cost  # log transition rate
        return rate

    def compute_local_gradient(self, current: SampleStruct, out: dl.Vector) -> None:
        """
        Compute the local gradient given the current sample
        :param current: The current sample
        :param out: The output vector
        """
        out.zero()
        out.axpy(-1., current.Cg)
        self._help2.zero()
        self._help2.axpy(1., current.m)
        self._help2.axpy(1., current.Cg)
        dplus1invd = current.eigenvalues / (1.0 + current.eigenvalues)
        VtRmpg = current.encoder.dot_v(self._help2)
        self._help2.zero()
        current.decoder.reduce(self._help2, dplus1invd * VtRmpg)
        out.axpy(1., self._help2)

    def density_wrt_pcn(self, origin: SampleStruct, destination: SampleStruct) -> list[float, ...]:
        """
        Compute the density with respect to the preconditioned Crank-Nicolson proposal
        :param origin: The origin sample
        :param destination: The destination sample
        return: A list of terms in the density with respect to the preconditioned Crank-Nicolson proposal
        """
        h = self.parameters["h"]
        rho = (4.0-h)/(4.0+h)
        Vtg = origin.decoder.dot_v(origin.g)
        VtRm = origin.encoder.dot_v(origin.m)
        self._help1.zero()
        self._help1.axpy(-rho / math.sqrt(1 - rho ** 2), origin.m)
        self._help1.axpy(1.0 / math.sqrt(1 - rho ** 2), destination.m)
        VtRmhat = origin.encoder.dot_v(self._help1)
        dplus1inv = 1. / (1.0 + origin.eigenvalues)
        dplus1 = origin.eigenvalues + 1.0
        rate = np.zeros(5)
        rate[0] = -0.125 * h * (np.inner(dplus1inv * (origin.eigenvalues * VtRm), origin.eigenvalues * VtRm) \
                                - np.inner(origin.eigenvalues * dplus1inv * Vtg, VtRm) \
                                + origin.g.inner(origin.Cg))
        rate[1] = 0.5 * math.sqrt(h) * (np.inner(VtRmhat, origin.eigenvalues * VtRm) - self._help1.inner(origin.g))
        rate[2] = -0.5 * (np.inner(VtRmhat, origin.eigenvalues * VtRmhat))
        rate[3] = 0.5 * np.sum(np.log(dplus1))
        return rate

    def consume_random(self) -> None:
        """
        Consume random numbers for the kernel
        """
        hp.parRandom.normal(1., self._noise)
        Omega = hp.MultiVector(self._help1, self.parameters["hessian_rank"] + self.parameters["oversampling"])
        hp.parRandom.normal(1., Omega)
        np.random.rand()
        if hasattr(self.model, "consume_random") and callable(getattr(self.model, "consume_random")):
            self.model.consume_random()

class FixedmMALAKernel(Kernel):

    """
        This class implement the kernel class of Reimannian manifold MALA with a fixed metric tensor. This is the infinite-dimensional preconditioned MALA.
    """

    def __init__(self, model: hp.Model, eigenvalues: np.ndarray, decoder: hp.MultiVector, encoder : hp.MultiVector = None) -> None:
        """
        Construction:
        :param model: The model class for bayesian inverse problems
        :param decoder: The decoder multivector (e.g., R-orthonormal eigenbasis)
        :param eigenvalues: The eigenvalues of the Hessian approximation
        """
        self.model = model
        self.decoder = decoder
        self.eigenvalues = eigenvalues
        self.parameters = {}
        self.parameters["h"] = 0.1
        self._noise = dl.Vector(self.model.prior.R.mpi_comm())
        self.model.prior.init_vector(self._noise, "noise")
        self._help1, self._help2 = dl.Vector(self.model.prior.R.mpi_comm()), dl.Vector(self.model.prior.R.mpi_comm())
        self.model.prior.init_vector(self._help1, 1)
        self.model.prior.init_vector(self._help2, 1)

        if encoder is None:
            self.encoder = hp.MultiVector(self.model.prior.R.mpi_comm(), self.decoder.nvec())
            hp.MatMvMult(self.model.prior.Rsolver, self.decoder, self.encoder)
        else:
            self.encoder = encoder
        check_orthonormality(self.decoder, self.encoder)

    def name(self) -> str:
        """
        Return the name of the kernel
        """
        return "Fixed-mMALA"

    def derivativeInfo(self) -> int:
        """
        Return the level of derivative information of the kernel
        """
        return 1
    
    def reduced(self) -> bool:
        """
        Return True if the kernel is reduced, False otherwise
        """
        return False 

    def generate_sample(self) -> SampleStruct:
        """
        Generate a sample data structure
        """
        return SampleStruct(self.model, self.derivativeInfo())


    def init_sample(self, s: SampleStruct) -> None:
        """
        Initialize the sample structure by computing the cost, gradient, eigenvalues and eigenbases.
        :param s: The sample structure
        """
        self.model.solveFwd(s.u, [s.u, s.m, None]) # solve the forward problem
        s.cost = self.model.cost([s.u, s.m, None])[2] # compute the misfit value
        self.model.solveAdj(s.p, [s.u, s.m, s.p]) # solve the adjoint problem
        self.model.evalGradientParameter([s.u, s.m, s.p], s.g, misfit_only=True) # compute the gradient
        self.model.prior.Rsolver.solve(s.Cg, s.g) # compute the preconditioned gradient

    def sample(self, current: SampleStruct, proposed: SampleStruct) -> None:
        """
        Sample a new proposal given the current sample
        """
        self.proposal(current, proposed.m)  # Coming up with a proposal step

    def accept_or_reject(self, current: SampleStruct, proposed: SampleStruct) -> int:
        """
        Accept or reject the proposed sample
        :param current: The current sample
        :param proposed: The proposed sample
        return: 1 if the proposed sample is accepted, 0 otherwise
        """
        rate_pc = self.acceptance_ratio(proposed, current)  # log transition rate from proposal to current
        rate_cp = self.acceptance_ratio(current, proposed)  # log transition rate from current to proposal
        al = np.sum(rate_pc - rate_cp)  # log ratio of transition rate
        if (al > math.log(np.random.rand())):  # Metropolis--Hasting acceptance--rejection step
            return 1
        else:
            return 0

    def proposal(self, current: SampleStruct, out: dl.Vector) -> None:
        """
        Compute the proposal step given the current sample
        :param current: The current sample
        :param out: The output vector, i.e., the proposed sample
        """
        h = self.parameters["h"] # step size
        rho = (4.0-h)/(4.0+h) # the scaling factor
        hp.parRandom.normal(1., self._noise)  # sample from i.i.d. random noise vector
        s_prior = dl.Vector(self.model.prior.R.mpi_comm()) # initialize prior sample
        s_post = dl.Vector(self.model.prior.R.mpi_comm())  # initialize proposal sample
        self.model.prior.init_vector(s_prior, 0) # initialize the prior sample
        self.model.prior.init_vector(s_post, 0) # initialize the proposal sample
        self.model.prior.sample(self._noise, s_prior, add_mean=False)  # sample from the prior distribution
        # compute the diagonal scaling for sample transformation
        invsqrt1plusdmin1 = -1.0 + 1. / (np.sqrt(1.0 + self.eigenvalues))
        VtRs = self.encoder.dot_v(s_prior)  # encode the prior samples
        s_post.zero()
        self.decoder.reduce(s_post, invsqrt1plusdmin1 * VtRs)  # decode the posterior component
        s_post.axpy(1., s_prior)
        self.compute_local_gradient(current, self._help1)  # compute local gradient in help1
        # Compute the proposed sample!
        out.zero()
        out.axpy(rho, current.m)
        out.axpy(1.0 - rho, self._help1)
        out.axpy(math.sqrt(1.0 - rho ** 2), s_post)

    def acceptance_ratio(self, origin: SampleStruct, destination: SampleStruct) -> list[float, ...]:
        """
        Compute the acceptance ratio given the origin and destination samples
        :param origin: The origin sample
        :param destination: The destination sample
        return: A list of terms in the acceptance ratio
        """
        rate = self.density_wrt_pcn(origin, destination)  # Compute the RN derivative with pCN
        rate[-1] = -origin.cost  # log transition rate
        return rate

    def compute_local_gradient(self, current: SampleStruct, out: dl.Vector) -> None:
        """
        Compute the local gradient given the current sample
        :param current: The current sample
        :param out: The output vector
        """
        out.zero()
        Vtg = self.decoder.dot_v(current.g)
        VtRm = self.encoder.dot_v(current.m)
        dplus1inv = 1. / (1.0 + self.eigenvalues)
        self.decoder.reduce(out, dplus1inv * (self.eigenvalues * VtRm - Vtg))

    def density_wrt_pcn(self, origin: SampleStruct, destination: SampleStruct) -> list[float, ...]:
        """
        Compute the density with respect to the preconditioned Crank-Nicolson proposal
        :param origin: The origin sample
        :param destination: The destination sample
        return: A list of terms in the density with respect to the preconditioned Crank-Nicolson proposal
        """
        h = self.parameters["h"]
        rho = (4.0-h)/(4.0+h)
        Vtg = self.decoder.dot_v(origin.g)
        VtRm = self.encoder.dot_v(origin.m)
        VtRmhat = -rho / math.sqrt(1 - rho ** 2) * VtRm
        VtRmhat += 1.0 / math.sqrt(1 - rho ** 2) * self.encoder.dot_v(destination.m)
        dplus1 = self.eigenvalues + 1.0
        dplus1inv = 1. / dplus1
        rate = np.zeros(5)
        rate[0] = -0.125 * h * np.inner(dplus1inv * (self.eigenvalues * VtRm - Vtg), self.eigenvalues * VtRm - Vtg)
        rate[1] = 0.5 * math.sqrt(h) * (np.inner(VtRmhat, self.eigenvalues * VtRm - Vtg))
        rate[2] = -0.5 * (np.inner(VtRmhat, self.eigenvalues * VtRmhat))
        rate[3] = 0.5 * np.sum(np.log(dplus1))
        return rate
    
    def consume_random(self) -> None:
        """
        Consume random numbers for the kernel
        """
        hp.parRandom.normal(1., self._noise)
        np.random.rand()
        if hasattr(self.model, "consume_random") and callable(getattr(self.model, "consume_random")):
            self.model.consume_random()

class ReducedBasisSurrogatemMALAKernel(Kernel):
    """
        This class implement the reduced mMALA kernel, which uses DINO as a surrogate for everything, including the acceptance ratio.
    """
    def __init__(self, surrogate: ReducedBasisSurrogate, parameter_encoder: hp.MultiVector = None, parameter_decoder: hp.MultiVector =None) -> None:
        """
        Construction:
        :param surrogate: The surrogate model with methods eval, cost, and misfit_vector
        :param parameter_encoder: The encoder multivector. This is pass to the data structure to handle assignment of parameter between full and reduced spaces.
        :param parameter_decoder: The decoder multivector. This is pass to the data structure to handle assignment of parameter between full and reduced spaces.
        """
        self.surrogate = surrogate
        self.parameters = {}
        self.parameters["h"] = 0.1
        self.input_dim, self.output_dim = surrogate.dim()
        self.parameter_encoder = parameter_encoder
        self.parameter_decoder = parameter_decoder
        self.rank = min(self.output_dim, self.input_dim)  # dimension in the reduced space only

    def name(self):
        """
        Return the name of the kernel
        """
        return "Reduced-basis surrogate-driven mMALA (without correction)"
    
    def reduced(self):
        """
        Return True if the kernel is reduced, False otherwise
        """
        return True

    def derivativeInfo(self):
        """
        Return the level of derivative information of the kernel
        """
        return 2

    def init_sample(self, s: ReducedSampleStruct) -> None:
        """
        Initialize the surrogate sample structure by computing the cost, gradient, eigenvalues and eigenbases.
        :param s: The surrogate sample structure
        """
        s.u, J = self.surrogate.eval(s.m, derivative_order=1)
        s.cost = self.surrogate.cost(s.u, s.m)[2]
        weighted_misfit_vector = self.surrogate.misfit_vector(s.u, True)
        s.g = J.T @ weighted_misfit_vector
        s.eigenvalues, s.rotation = eigh(J.T@self.surrogate.noise_precision@J)
        s.eigenvalues = s.eigenvalues[::-1]
        s.rotation = s.rotation[:, ::-1]

    def generate_sample(self) -> ReducedSampleStruct:
        """
        Generate a surrogate sample data structure
        """
        return ReducedSampleStruct(derivative_info=self.derivativeInfo(), encoder=self.parameter_encoder, decoder=self.parameter_decoder)

    def sample(self, current: ReducedSampleStruct, proposed: ReducedSampleStruct) -> None:
        """
        Generate a new proposal given the current sample
        """
        proposed.m = self.proposal(current)  # Coming up with a proposal step

    def accept_or_reject(self, current: ReducedSampleStruct, proposed: ReducedSampleStruct) -> int:
        """
        Accept or reject the proposed sample
        :param current: The current sample
        :param proposed: The proposed sample
        """
        rate_pc = self.acceptance_ratio(proposed, current)  # log transition rate from proposal to current
        rate_cp = self.acceptance_ratio(current, proposed)  # log transition rate from current to proposal
        al = np.sum(rate_pc - rate_cp)  # log ratio of transition rate
        if (al > math.log(np.random.rand())):  # Metropolis--Hasting acceptance--rejection step
            return 1
        else:
            return 0

    def proposal(self, current: ReducedSampleStruct) -> np.ndarray:
        """
        Compute the proposal step given the current sample
        :param current: The current sample
        return: The proposed sample
        """
        h = self.parameters["h"]
        rho = (4.0-h)/(4.0+h)
        noise = np.random.normal(scale=np.sqrt((1 - rho ** 2) / (current.eigenvalues + 1)), size=self.input_dim)
        mean = (current.eigenvalues + rho) / (current.eigenvalues + 1) * current.rotation.T @ current.m - \
               (1.0 - rho) / (current.eigenvalues + 1) * current.rotation.T @ current.g
        return current.rotation @ (mean + noise)

    def acceptance_ratio(self, origin : ReducedSampleStruct, destination: ReducedSampleStruct) -> np.ndarray:
        """
        Compute the acceptance ratio given the origin and destination samples
        :param origin: The origin sample
        :param destination: The destination sample
        return: A list of terms in the acceptance ratio
        """
        rate = self.density_wrt_pcn(origin, destination)  # Compute the RN derivative with pCN
        rate[-1] = -origin.cost  # log transition rate
        return rate

    def density_wrt_pcn(self, origin :ReducedSampleStruct, destination: ReducedSampleStruct) -> np.ndarray:
        """
        Compute the density with respect to the preconditioned Crank-Nicolson proposal
        :param origin: The origin sample
        :param destination: The destination sample
        return: A list of terms in the density with respect to the preconditioned Crank-Nicolson proposal
        """
        h = self.parameters["h"]
        rho = (4.0-h)/(4.0+h)
        dplus1 = origin.eigenvalues + 1.0
        dplus1inv = 1. / dplus1
        rotated_mr = origin.rotation.T @ origin.m
        rotated_gr = origin.rotation.T @ origin.g
        rotated_mhat = origin.rotation.T @ (destination.m - rho * origin.m) /math.sqrt(1.0-rho**2)
        rate = np.zeros(5)
        rate[0] = -0.125 * h * np.inner(dplus1inv * (origin.eigenvalues * rotated_mr - rotated_gr),
                                        origin.eigenvalues * rotated_mr - rotated_gr)
        rate[1] = 0.5 * math.sqrt(h) * (np.inner(rotated_mhat, origin.eigenvalues * rotated_mr - rotated_gr))
        rate[2] = -0.5 * (np.inner(rotated_mhat, origin.eigenvalues * rotated_mhat))
        rate[3] = 0.5 * np.sum(np.log(dplus1))
        return rate

    def consume_random(self):
        """
        Consume random numbers for the kernel
        """
        np.random.rand()
        np.random.normal(size=self.input_dim)
        if hasattr(self.surrogate, "consume_random") and callable(getattr(self.surrogate, "consume_random")):
            self.surrogate.consume_random()


class pCNKernel(hp.pCNKernel):
    """
    This class implements the preconditioned Crank-Nicolson algorithm inherited from the hippylib implementation.
    """
    def __init__(self, model: hp.Model) -> None:
        super(pCNKernel, self).__init__(model)
        self.parameters["h"] = 0.1
    
    def name(self) -> str:
        return "pCN"
    
    def derivativeInfo(self) -> int:
        return 0
    
    def reduced(self) -> bool:
        return False
    
    def generate_sample(self) -> SampleStruct:
        return SampleStruct(self.model, self.derivativeInfo())

    def sample(self, current: SampleStruct, proposed: SampleStruct) -> None:
        h = self.parameters["h"]
        self.parameters["s"] = 4*math.sqrt(h)/(4.0+h)
        proposed.m = self.proposal(current)

    def accept_or_reject(self, current: SampleStruct, proposed: SampleStruct) -> int:
        al = -proposed.cost + current.cost
        if (al > math.log(np.random.rand())):
            return 1
        else:
            return 0

class MALAKernel(hp.MALAKernel):
    """
    This class implements the MALA algorithm inherited from the hippylib implementation.
    """
    def __init__(self, model: hp.Model) -> None:
        super(MALAKernel, self).__init__(model)
        self.parameters["h"] = 0.1
    
    def name(self) -> str:
        return "MALA"
    
    def derivativeInfo(self) -> int:
        return 1
    
    def reduced(self) -> bool:
        return False
    
    def generate_sample(self) -> SampleStruct:
        return SampleStruct(self.model, self.derivativeInfo())

    def sample(self, current: SampleStruct, proposed: SampleStruct) -> None:
        h = self.parameters["h"]
        self.parameters["delta_t"] = 2.0*h
        proposed.m = self.proposal(current)

    def accept_or_reject(self, current: SampleStruct, proposed: SampleStruct) -> int:
        rho_mp = self.acceptance_ratio(current, proposed)
        rho_pm = self.acceptance_ratio(proposed, current)
        al = rho_mp - rho_pm
        if (al > math.log(np.random.rand())):
            return 1
        else:
            return 0

class gpCNKernel(hp.gpCNKernel):
    """
    This class implements the generalized preconditioned Crank--Nicolson algorithm inherited from the hippylib implementation.
    """
    def __init__(self, model: hp.Model, nu) -> None:

        super(gpCNKernel, self).__init__(model, nu)
        self.parameters["h"] = 0.1
    
    def name(self) -> str:
        return "gpCN"
    
    def derivativeInfo(self) -> int:
        return 1
    
    def reduced(self) -> bool:
        return False
    
    def generate_sample(self):
        return SampleStruct(self.model, self.derivativeInfo())

    def sample(self, current: SampleStruct, proposed: SampleStruct) -> None:
        h = self.parameters["h"]
        self.parameters["s"] = 4 * math.sqrt(h) / (4.0 + h)
        proposed.m = self.proposal(current)

    def accept_or_reject(self, current: SampleStruct, proposed: SampleStruct) -> int:
        al = self.delta(current) - self.delta(proposed)
        if (al > math.log(np.random.rand())):
            return 1
        else:
            return 0

class DelayedAcceptanceKernel(pCNKernel):
    """
    This class implements the final stage of the delayed acceptance algorithm inherited from the hippylib implementation.
    """
    def __init__(self, model: hp.Model) -> None:
        super(DelayedAcceptanceKernel, self).__init__(model)
    
    def name(self) -> str:
        return "Delayed Acceptance"
    
    def reduced(self) -> bool:
        return False

    def sample(self, *args, **kwargs):
        pass

    def accept_or_reject(self, current: SampleStruct, proposed: SampleStruct) -> int:
        assert len(current) == 2 and len(proposed) == 2
        al = current[1].cost - proposed[1].cost - current[0].cost + proposed[0].cost
        if (al > math.log(np.random.rand())):
            return 1
        else:
            return 0