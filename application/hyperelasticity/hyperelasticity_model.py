# Author: Lianghao Cao
# Date: 01/24/2024
# This code (the line search algoirthm) is partially provided by Tom O'Leary and written by D.C. Luo
import os, sys
import ufl, math
import dolfin as dl
dl.set_log_level(dl.LogLevel.ERROR) # delete this will print the Newton iterations info for the forward problem
import numpy as np
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
import matplotlib.pyplot as plt
sys.path.insert(0, "../../")
import geometric_mcmc as gmc

def hyperelasticity_settings(settings={}):
    settings["seed"] = 0  # Random seed
    settings["aspect_ratio"] = 2.0
    settings["nx"] = 64  # Number of cells in each direction

    # Prior statistics
    settings["sigma"] = 1.0  # pointwise variance
    settings["rho"] = 0.3  # spatial correlation length

    # Anisotropy of prior samples
    settings["theta0"] = 2.0
    settings["theta1"] = 0.5
    settings["alpha"] = math.atan(settings["aspect_ratio"])

    # Likelihood specs
    settings["ntargets"] = 32
    settings["rel_noise"] = 0.01

    # Printing and saving
    settings["verbose"] = True
    settings["output_path"] = "./"
    settings["save_setup"] = True

    return settings

def left_boundary(x, on_boundary):
    return on_boundary and (x[0] < dl.DOLFIN_EPS)

def right_boundary(x, on_boundary):
    return on_boundary and (x[0] > 2.0 - dl.DOLFIN_EPS)

class HyperelasticityVarf:
    """
    The class for the hyperelasticity variational form

    """
    def __init__(self):
        pass

    def parameter_map(self, m):
        """
        The parameter-to-modulus map
        """
        return dl.Constant(3.0) * (ufl.erf(m) + dl.Constant(1.0)) + dl.Constant(1.0)

    def __call__(self, u, m, p):
        Pi = self.energy_form(u, m)
        return dl.derivative(Pi, u, p)

    def energy_form(self, u, m):
        d = u.geometric_dimension()
        Id = dl.Identity(d)
        F = Id + dl.grad(u)
        C = F.T * F
        # Lame parameters
        # -------------------
        # The new parameter-to-modulus map
        # -------------------
        E = self.parameter_map(m)
        nu = 0.4
        mu = E / (2.0 * (1.0 + nu))
        lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Invariants of the deformation tensors
        Ic, J = dl.tr(C), dl.det(F)

        # Stored strain energy density
        psi = (mu / 2.0) * (Ic - 3.0) - mu * dl.ln(J) + (lmbda / 2.0) * (dl.ln(J)) ** 2

        Pi = psi * dl.dx
        return Pi

class CustomHyperelasticityProblem(hp.PDEVariationalProblem):
    def __init__(self, Vh, stretch_length):
        bc = [dl.DirichletBC(Vh[hp.STATE], dl.Constant((stretch_length, 0.0)), right_boundary), \
              dl.DirichletBC(Vh[hp.STATE], dl.Constant((0.0, 0.0)), left_boundary)]
        bc0 = [dl.DirichletBC(Vh[hp.STATE], dl.Constant((0.0, 0.0)), left_boundary), \
               dl.DirichletBC(Vh[hp.STATE], dl.Constant((0.0, 0.0)), right_boundary)]
        hyperelasticity_varf_hander = HyperelasticityVarf()
        super(CustomHyperelasticityProblem, self).__init__(Vh, hyperelasticity_varf_hander, bc, bc0,
                                                           is_fwd_linear=False)
        u_init_expr = dl.Expression(("0.5*L*x[0]", "0.0"), L=stretch_length, degree=5)
        u_init_func = dl.interpolate(u_init_expr, Vh[hp.STATE])
        self.u_init = u_init_func.vector()
        self.iterations = 0

    def parameter_map(self, m):
        m_out = self.generate_parameter()
        m_func_in = dl.Function(self.Vh[hp.PARAMETER])
        m_func_in.vector().zero()
        m_func_in.vector().axpy(1., m)
        m_func_out = dl.project(self.varf_handler.parameter_map(m_func_in), self.Vh[hp.PARAMETER])
        m_out.axpy(1., m_func_out.vector())
        return m_out

    def solveFwd(self, state, x):

        x[hp.STATE].zero()
        x[hp.STATE].axpy(1., self.u_init)
        u = hp.vector2Function(x[hp.STATE], self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = dl.TestFunction(self.Vh[hp.ADJOINT])
        F = self.varf_handler(u, m, p)
        du = dl.TrialFunction(self.Vh[hp.STATE])
        JF = dl.derivative(F, u, du)
        problem = dl.NonlinearVariationalProblem(F, u, self.bc, JF)
        solver = dl.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['nonlinear_solver'] = 'snes'
        prm['snes_solver']['line_search'] = 'bt'
        prm['snes_solver']['linear_solver'] = 'lu'
        prm['snes_solver']['report'] = False
        prm['snes_solver']['error_on_nonconvergence'] = True
        prm['snes_solver']['absolute_tolerance'] = 1E-10
        prm['snes_solver']['relative_tolerance'] = 1E-6
        prm['snes_solver']['maximum_iterations'] = 1000
        prm['newton_solver']['absolute_tolerance'] = 1E-10
        prm['newton_solver']['relative_tolerance'] = 1E-6
        prm['newton_solver']['maximum_iterations'] = 1000
        prm['newton_solver']['relaxation_parameter'] = 1.0

        iterations, converged = solver.solve()
        self.iterations += iterations
        state.zero()
        state.axpy(1., u.vector())


def HyperelasticityPrior(Vh_PARAMETER, pointwise_std, correlation_length, mean=None, anis_diff=None):
    delta = 1.0 / (pointwise_std * correlation_length)
    gamma = delta * correlation_length ** 2
    if anis_diff is None:
        theta0 = 1
        theta1 = 1
        alpha = math.pi / 4.
        anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
        anis_diff.set(theta0, theta1, alpha)
    if mean is None:
        return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, robin_bc=True)
    else:
        return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, mean=mean, robin_bc=True)

def HyperelasticityMisfit(Vh, state, settings):

    ndim = 2

    ntargets = settings["ntargets"]
    aspect_ratio = settings["aspect_ratio"]

    ny_targets = math.floor(math.sqrt(ntargets / aspect_ratio))
    nx_targets = math.floor(ntargets / ny_targets)
    if not ny_targets * nx_targets == ntargets:
        raise Exception("The number of obaservation points cannot lead to a regular grid \
        that is compatible with the aspect ratio.")
    targets_x = np.linspace(0.0, settings["aspect_ratio"], nx_targets + 2)
    targets_y = np.linspace(0.0, 1.0, ny_targets + 2)
    targets_xx, targets_yy = np.meshgrid(targets_x[1:-1], targets_y[1:-1])
    targets = np.zeros([ntargets, ndim])
    targets[:, 0] = targets_xx.flatten()
    targets[:, 1] = targets_yy.flatten()
    observables = gmc.PointwiseObservation(Vh[hp.STATE], targets)
    observable_predictions = observables.eval([state, None])
    MAX = np.max(np.abs(observable_predictions))
    noise_std_dev = settings["rel_noise"] * MAX
    noisy_data = observable_predictions + np.random.normal(0, noise_std_dev, ntargets*2)
    noise_precision = (1.0 / noise_std_dev ** 2) * np.eye(ntargets*2)
    misfit = gmc.ObservableMisfit(Vh, observables, data=noisy_data, noise_precision=noise_precision)

    return misfit, observable_predictions, targets

def hyperelasticity_model(comm_mesh, settings):

    np.random.seed(settings["seed"])
    output_path = settings["output_path"]

    rank = comm_mesh.rank # Get the rank of the current processor
    size = comm_mesh.size # Get the size of the MPI communicator
    output_path = settings["output_path"] # Get the output path
    if settings["save_setup"] and output_path is None: 
        raise Exception("The output path is not set in the model setting.") # Check if the output path is set
    if settings["save_setup"] and not os.path.exists(output_path +  "bip_setup/"): # Create the BIP setup subfolder
        os.makedirs(output_path + "bip_setup/", exist_ok=True)

    nx = settings["nx"]
    mesh = dl.RectangleMesh(comm_mesh, dl.Point(0, 0), dl.Point(settings["aspect_ratio"], 1.0), \
                            nx, math.floor(nx / settings["aspect_ratio"]))
    Vh_STATE = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Vh_PARAMETER = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]

    if settings["verbose"] and rank == 0:
        print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
            Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()))

    pde = CustomHyperelasticityProblem(Vh, 1.5)

    theta0 = settings["theta0"]
    theta1 = settings["theta1"]
    alpha = settings["alpha"]
    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=5)
    anis_diff.set(theta0, theta1, alpha)
    prior = HyperelasticityPrior(Vh[hp.PARAMETER], settings["sigma"], settings["rho"], anis_diff=anis_diff)

    utrue = pde.generate_state()
    mtrue = true_parameter(prior)
    x = [utrue, mtrue, None]
    pde.solveFwd(x[hp.STATE], x)

    misfit, observable_prediction, targets = HyperelasticityMisfit(Vh, x[hp.STATE], settings)

    if settings["save_setup"]:
        true_param_array = gmc.get_global(comm_mesh, mtrue, root=0)
        if size==1:
            cbar = dl.plot(hp.vector2Function(mtrue, Vh[hp.PARAMETER]))
            loc = np.linspace(np.min(mtrue.get_local()), np.max(mtrue.get_local()), 5)
            cmap = plt.colorbar(cbar, ticks=loc, location='top')
            label = list(np.round(loc, 2))
            cmap.ax.set_xticklabels(label)
            plt.axis("off")
            plt.savefig(output_path + "bip_setup/" + "true_parameter.pdf", bbox_inches="tight")
            plt.close()
            E_true = hp.vector2Function(pde.parameter_map(mtrue), Vh[hp.PARAMETER])
            cbar = dl.plot(E_true)
            loc = np.linspace(np.min(E_true.vector().get_local()), np.max(E_true.vector().get_local()), 5)
            cmap = plt.colorbar(cbar, ticks=loc, location='top')
            label = list(np.round(loc, 2))
            cmap.ax.set_xticklabels(label)
            plt.axis("off")
            plt.savefig(output_path + "bip_setup/" + "true_modulus.pdf", bbox_inches="tight")
            plt.close()
            cbar = dl.plot(hp.vector2Function(utrue, Vh[hp.STATE]), mode="displacement", vmin=0.0, vmax=1.5)
            loc = np.linspace(0.0, 1.5, 4)
            cmap = plt.colorbar(cbar, ticks=loc, location='top')
            label = list(np.round(loc, 2))
            cmap.ax.set_xticklabels(label)
            reshaped_observable = observable_prediction.reshape(misfit.observable.dim()//2, 2)
            reshaped_data = misfit.data.reshape(misfit.observable.dim()//2, 2)
            plt.scatter(targets[:, 0] + reshaped_observable[:, 0], targets[:, 1] + reshaped_observable[:, 1], 
                        c=np.linalg.norm(reshaped_data, axis=1), marker="o", s=30, edgecolors="k", linewidths= 2)
            plt.axis("off")
            plt.savefig(output_path + "bip_setup/" + "true_state_with_observations.pdf", bbox_inches="tight")
            plt.close()
        else:
            to_vertex, _ = gmc.get_vertex_order(Vh[hp.PARAMETER], root=0) # if mesh parallel, get the indices for image order of the parameter entires.
        if rank==0:
            plt.plot(observable_prediction, "-", label="True observable") 
            plt.plot(misfit.data, "o", label="Noisy data")
            plt.legend(loc="best")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.savefig(output_path + "bip_setup/" + "observation_data.pdf", bbox_inches="tight")
            plt.close()
            np.save(output_path + "bip_setup/" +  "targets.npy", targets)
            np.save(output_path + "bip_setup/" +  "data.npy", misfit.data)
            np.save(output_path + "bip_setup/" +  "noise_precision.npy", misfit.noise_precision)
            np.save(output_path + "bip_setup/" +  "true_observable.npy", observable_prediction)
            np.save(output_path + "bip_setup/" +  "true_parameter.npy", true_param_array)
            if size>1: np.save(output_path + "bip_setup/" +  "to_vertex.npy", to_vertex)
    return hp.Model(pde, prior, misfit), mtrue

def true_parameter(prior):
    noise = dl.Vector(prior.R.mpi_comm())
    prior.init_vector(noise, "noise")
    hp.parRandom.normal(1.0, noise)
    mtrue = dl.Vector(prior.R.mpi_comm())
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    return mtrue