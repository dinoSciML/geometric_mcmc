import hippylib as hp
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
from ..model.pto_map import PtOMapJacobian
from ..mcmc.kernel_utilities import decomposeHessian

def compute_GNH_at_MAP(model, mode="reverse"):
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

    pto_map_jac = PtOMapJacobian(model.problem, model.misfit.observable)
    Rinv_operator = hp.Solver2Operator(model.prior.Rsolver)

    J = pto_map_jac.generate_jacobian() # generate the Jacobian multivector
    Rinv_J = pto_map_jac.generate_jacobian() # compute the prior preconditioned Jacobian

    pto_map_jac.setLinearizationPoint(x_MAP) # set the linearization point
    pto_map_jac.eval(J, mode = mode) # compute the Jacobian
    hp.MatMvMult(Rinv_operator, J, Rinv_J) # compute the encoder

    encoder = pto_map_jac.generate_jacobian()
    decoder = pto_map_jac.generate_jacobian()

    eigenvalues = decomposeHessian(J, Rinv_J, decoder, encoder, Rinv_operator, model.misfit.noise_precision) # compute the eigenvalues and eigenvectors

    return x_MAP, eigenvalues, decoder, encoder