#Author: Lianghao Cao
#Date: Sep. 26, 2023
#This script is copied from hippylib tutorial, with minor modificiations.
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

def nonlinear_diffusion_reaction_settings(settings ={}):
    """
    Settings for the nonlinear diffusion reaction model
    """
    settings["seed"] = 0 #Random seed
    settings["nx"] = 40 # Number of cells in each direction

    #Prior statistics
    settings["sigma"] = 3 # pointwise variance
    settings["rho"] = 0.1 # spatial correlation length

    #Anisotropy of prior samples
    settings["theta0"] = 1.0
    settings["theta1"] = 1.0
    settings["alpha"] = 0.25*math.pi

    #Likelihood specs
    settings["ntargets"] = 25
    settings["rel_noise"] = 0.02

    #Printing and saving
    settings["verbose"] = False
    settings["output_path"] = None
    settings["save_setup"] = False

    return settings

def u_boundary(x, on_boundary): # Define the boundary with Dirichlet condition
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def pde_varf(u, m, p): # Define the PDE variational form
    return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx + u * u * u * p * ufl.dx - dl.Constant(0.0) * p * ufl.dx

def nonlinear_diffusion_reaction_model(mpi_comm, settings):
    """
    Create the nonlinear diffusion reaction hippylib model class.
    :param mpi_comm: The MPI communicator for possible domain decomposition.
    :param settings: The settings for the model
    return: The model class and the 'true' parameter
    """

    np.random.seed(settings["seed"]) # Set the random seed
    rank = mpi_comm.rank # Get the rank of the current processor
    size = mpi_comm.size # Get the size of the MPI communicator

    output_path = settings["output_path"] # Get the output path
    if settings["save_setup"] and output_path is None: 
        raise Exception("The output path is not set in the model setting.") # Check if the output path is set
    if settings["save_setup"] and not os.path.exists(output_path +  "bip_setup/"): # Create the BIP setup subfolder
        os.makedirs(output_path + "bip_setup/", exist_ok=True)
    
    # Define the mesh and function spaces
    ndim = 2
    nx = settings["nx"]
    mesh = dl.UnitSquareMesh(mpi_comm, nx, nx)
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]

    if settings["verbose"] and rank == 0:
        print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
            Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()))

    # Define the boundary conditions
    u_bdr = dl.Expression("x[1]", degree=1, mpi_comm=mpi_comm)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[hp.STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, u_boundary)

    # Define the PDE variational problem
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)

    # Define the prior
    delta = 1.0/(settings["sigma"]*settings["rho"])
    gamma = delta*settings["rho"]**2
    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
    anis_diff.set( settings["theta0"], settings["theta1"], settings["alpha"])

    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, anis_diff, robin_bc=True)

    # Define the observation points and the observable
    ntargets = settings["ntargets"]
    rel_noise = settings["rel_noise"]

    targets = np.random.uniform(0.1, 0.9, [ntargets, ndim] )
    if settings["verbose"] and rank==0: print("Number of observation points: {0}".format(ntargets))
    observable = gmc.PointwiseObservation(Vh[hp.STATE], targets)

    # Generate the synthetic data set
    utrue = pde.generate_state()
    mtrue = true_parameter(prior, random=False) # We use a very out of distribution parameter sample. See LogPermField class below.
    x = [utrue, mtrue, None]
    pde.solveFwd(x[hp.STATE], x) # Solve the forward problem
    observable_prediction = observable.eval(x) # Evaluate the observable
    MAX = np.max(np.abs(observable_prediction)) # Get the maximum value of the observable
    noise_std_dev = rel_noise * MAX # Set the noise standard deviation
    data = observable_prediction + np.random.normal(scale=noise_std_dev, size=observable_prediction.shape) # Add noise to the observable
    noise_precision = np.diag(1.0/(noise_std_dev**2)*np.ones_like(data)) # Set the noise precision

    misfit = gmc.ObservableMisfit(Vh, observable, data=data, noise_precision=noise_precision) # Define the misfit

    # Save the setup and visualize if no mesh parallel
    if settings["save_setup"]:
        true_param_array = gmc.get_global(mpi_comm, mtrue, root=0)
        if size==1:
            cbar = dl.plot(hp.vector2Function(utrue, Vh[hp.STATE]))
            plt.colorbar(cbar)
            plt.scatter(targets[:, 0], targets[:, 1], c=data, marker="o", s=20, edgecolors="k", linewidths= 2)
            plt.axis("off")
            plt.savefig(output_path + "bip_setup/" + "true_state_with_observation.pdf", bbox_inches="tight")
            plt.close()
            cbar = dl.plot(hp.vector2Function(mtrue, Vh[hp.PARAMETER]))
            plt.colorbar(cbar)
            plt.axis("off")
            plt.savefig(output_path + "bip_setup/" + "true_parameter.pdf", bbox_inches="tight")
            plt.close()
        else:
            to_vertex, _ = gmc.get_vertex_order(Vh[hp.PARAMETER], root=0) # if mesh parallel, get the indices for image order of the parameter entires.
        if rank==0:
            plt.plot(observable_prediction, "-", label="True observable") 
            plt.plot(data, "o", label="Noisy data")
            plt.legend(loc="best")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.savefig(output_path + "bip_setup/" + "observation_data.pdf", bbox_inches="tight")
            plt.close()
            np.save(output_path + "bip_setup/" +  "targets.npy", targets)
            np.save(output_path + "bip_setup/" +  "data.npy", data)
            np.save(output_path + "bip_setup/" +  "noise_precision.npy", noise_precision)
            np.save(output_path + "bip_setup/" +  "true_observable.npy", observable_prediction)
            np.save(output_path + "bip_setup/" +  "true_parameter.npy", true_param_array)
            if size>1: np.save(output_path + "bip_setup/" +  "to_vertex.npy", to_vertex)
    
    return hp.Model(pde, prior, misfit), mtrue

class LogPermField(dl.UserExpression):
    """
    A class for generating a log permeability field with a ring and a square.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def inside_ring(self, x, length):
        dist_sq = (x[0] - 0.8*length) ** 2 + (x[1] - 0.2*length) ** 2
        return (dist_sq <= (0.6*length) ** 2) and (dist_sq  >= (0.5*length)**2) and x[1] >= 0.2*length and x[0] <= 0.8*length
    def inside_square(self, x, length):
        return (x[0]>=0.6*length and x[0] <= 0.8*length and x[1] >= 0.2*length and x[1] <= 0.4*length)
    def eval(self, value, x):
        if self.inside_ring(x, 1.0) or self.inside_square(x, 1.0):
            value[0] = 3.0
        else:
            value[0] = -3.0
    def value_shape(self):
        return ()

def true_parameter(prior, random = True):
    """
    Generate a true parameter sample.
    :param prior: The prior
    :param random: Whether to generate a random sample or using the LogPermField class
    return: The true parameter as :coder:`dolfin.Vector`
    """
    if random:
        noise = dl.Vector(prior.R.mpi_comm())
        prior.init_vector(noise, "noise")
        hp.parRandom.normal(1.0, noise)
        mtrue = dl.Vector(prior.R.mpi_comm())
        prior.init_vector(mtrue, 0)
        prior.sample(noise, mtrue)
        return mtrue
    else:
        mtrue_expression = LogPermField()
        mtrue_func = dl.interpolate(mtrue_expression, prior.Vh)
        return mtrue_func.vector()
    