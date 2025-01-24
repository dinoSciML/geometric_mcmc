#Author: Lianghao Cao
#Date: Sep. 26, 2023
#This script is copied from hippylib tutorial, with minor modificiations.

import os, sys
import ufl, math
import dolfin as dl
import numpy as np
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
import matplotlib.pyplot as plt
sys.path.insert(0, "../../")
import geometric_mcmc as gmc


def nonlinear_diffusion_reaction_settings(settings ={}):
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
    settings["verbose"] = True
    settings["output_path"] = "./result/"
    settings["plot"] = True


    return settings


def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def pde_varf(u, m, p):
    return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx + u * u * u * p * ufl.dx - dl.Constant(0.0) * p * ufl.dx

def nonlinear_diffusion_reaction_model(settings):

    np.random.seed(settings["seed"])

    output_path = settings["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    ndim = 2
    nx = settings["nx"]
    mesh = dl.UnitSquareMesh(nx, nx)
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]

    rank = mesh.mpi_comm().rank

    if settings["verbose"] and rank==0:
        print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
            Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()))

    u_bdr = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[hp.STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, u_boundary)

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)

    sigma = settings["sigma"]
    rho = settings["rho"]
    delta = 1.0/(sigma*rho)
    gamma = delta*rho**2

    theta0 = settings["theta0"]
    theta1 = settings["theta1"]
    alpha = settings["alpha"]

    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
    anis_diff.set(theta0, theta1, alpha)

    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, anis_diff, robin_bc=True)

    ntargets = settings["ntargets"]
    rel_noise = settings["rel_noise"]

    targets = np.random.uniform(0.1, 0.9, [ntargets, ndim] )
    if settings["verbose"] and rank==0:
        print("Number of observation points: {0}".format(ntargets))
    observable = gmc.PointwiseObservation(Vh[hp.STATE], targets)

    utrue = pde.generate_state()
    mtrue = true_parameter(prior, random=False)
    x = [utrue, mtrue, None]
    pde.solveFwd(x[hp.STATE], x)
    data = observable.eval(x)
    MAX = np.max(np.abs(data))
    noise_std_dev = rel_noise * MAX
    data += np.random.normal(scale=noise_std_dev, size=data.shape)
    noise_precision = np.diag(1.0/(noise_std_dev**2)*np.ones_like(data))

    misfit = gmc.ObservableMisfit(Vh, observable, data=data, noise_precision=noise_precision)

    if settings["plot"] and rank==0:
        cbar = plt.scatter(targets[:, 0], targets[:, 1], c=data, marker=",", s=10)
        plt.colorbar(cbar)
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.gca().set_aspect('equal')
        plt.savefig(output_path + "obs_sample.pdf", bbox_inches="tight")
        plt.close()
        np.save(output_path + "targets.npy", targets)
    
    return hp.Model(pde, prior, misfit), mtrue

class LogPermField(dl.UserExpression):
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
    if random:
        noise = dl.Vector()
        prior.init_vector(noise, "noise")
        hp.parRandom.normal(1.0, noise)
        mtrue = dl.Vector()
        prior.init_vector(mtrue, 0)
        prior.sample(noise, mtrue)
        return mtrue
    else:
        mtrue_expression = LogPermField()
        mtrue_func = dl.interpolate(mtrue_expression, prior.Vh)
        return mtrue_func.vector()
    