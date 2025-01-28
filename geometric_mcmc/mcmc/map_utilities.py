import hippylib as hp
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
from ..model.pto_map import PtOMapJacobian
from .kernel_utilities import decomposeGaussNewtonHessian
import warnings

def compute_MAP(model):
    """
    Compute the MAP point using inexact Newton-CG method with inexact line search globalization and Gauss-Newton approximation warm start.
    """
    m = model.prior.mean.copy()
    solver = hp.ReducedSpaceNewtonCG(model)
    solver.parameters["rel_tolerance"] = 1e-6
    solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["max_iter"] = 25
    solver.parameters["GN_iter"] = 5
    solver.parameters["globalization"] = "LS"
    solver.parameters["LS"]["c_armijo"] = 1e-4
    solver.parameters["print_level"] = -1
    x_MAP = solver.solve([None, m, None])
    return x_MAP

def compute_Hessian_decomposition_at_sample(model, x, gauss_newton_approx=False, form_jacobian=True, mode="reverse", rank=None, oversampling=20):
    """
    Compute the Hessian decomposition of the model at the MAP point.
    :param model: The hippylib model class with ObservableMisfit
    :param x: The sample point. Must be a list of [state, parameter, adjoint] consists of :code:`dolfin.Vector`.
    :param gauss_newton_approx: Whether to use the Gauss-Newton approximation.
    :param form_jacobian: Whether to form the Jacobian or use matrix-free randomized eigensolver. Only used if the Gauss-Newton approximation is assumed.
    :param mode: The differentiation mode for forming the jacobian matrix. Must be either `forward' or `reverse'.
    :param rank: The rank of the dimension reduction. Must assign a value if the Gauss-Newton approximation is not used.
    :param oversampling: The oversampling factor.
    """

    if gauss_newton_approx and form_jacobian: # If the Gauss-Newton approximation is used and the Jacobian is formed, the directly solve the eigenvalue problem.

        if rank is not None: warnings.warn("The rank is specified but not used for Gauss--Newton approximation computed using Jacobian.")
        pto_map_jac = PtOMapJacobian(model.problem, model.misfit.observable)
        Rinv_operator = hp.Solver2Operator(model.prior.Rsolver)

        J = pto_map_jac.generate_jacobian() # generate the Jacobian multivector
        Rinv_J = pto_map_jac.generate_jacobian() # compute the prior preconditioned Jacobian

        pto_map_jac.setLinearizationPoint(x) # set the linearization point
        pto_map_jac.eval(J, mode = mode) # compute the Jacobian
        hp.MatMvMult(Rinv_operator, J, Rinv_J) # compute the encoder

        encoder = pto_map_jac.generate_jacobian()
        decoder = pto_map_jac.generate_jacobian()

        eigenvalues = decomposeGaussNewtonHessian(J, Rinv_J, decoder, encoder, Rinv_operator, model.misfit.noise_precision) # compute the eigenvalues and eigenvectors
    
    else:
        if gauss_newton_approx:
            rank = model.misfit.observable.dim()
        assert rank is not None, "The rank must be specified if the Gauss-Newton approximation is not used."
        model.setPointForHessianEvaluations(x, gauss_newton_approx)
        Hmisfit = hp.ReducedHessian(model, misfit_only=True) # compute the Hessian approximation
        Omega = hp.MultiVector(x[PARAMETER], rank+oversampling) # initialize the multivector
        hp.parRandom.normal(1., Omega)
        eigenvalues, decoder = hp.doublePassG(Hmisfit, model.prior.R, model.prior.Rsolver, Omega, rank) # compute the eigenvalues and eigenvectors
        encoder = hp.MultiVector(x[PARAMETER], decoder.nvec())
        hp.MatMvMult(model.prior.R, decoder, encoder) # compute the encoder

    return eigenvalues, decoder, encoder