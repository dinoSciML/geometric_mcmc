import numpy as np
import dolfin as dl
dl.set_log_level(dl.LogLevel.ERROR)
import sys, os, argparse
from mpi4py import MPI
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.insert(0, '../../')
import geometric_mcmc as gmc
from ndr_model import nonlinear_diffusion_reaction_model, nonlinear_diffusion_reaction_settings
import matplotlib.pyplot as plt
import matplotlib
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass

step_sizes = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='nonlinear diffusion--reaction inference')

    parser.add_argument('--n_samples',
                        default=1000,
                        type=int,
                        help="Number of samples for each processer")

    parser.add_argument('--n_processes',
                        default=1,
                        type=int,
                        help="The total number of process MPI communicator")
    
    parser.add_argument('--n_subdomains',
                        default=1,
                        type=int,
                        help="The size of the mesh MPI communicator")

    parser.add_argument('--output_path',
                        default="./step_size_tuning/",
                        type=str,
                        help="The output path for saving")

    parser.add_argument('--output_frequency',
                        default=50,
                        type=int,
                        help="The number of samples for saving")
    
    parser.add_argument('--method',
                        default=0,
                        type=str,
                        help="The method to use in sampling")

    args = parser.parse_args()
    n_processes = args.n_processes
    n_subdomains = args.n_subdomains
    n_samples = args.n_samples
    output_path = args.output_path

    comm_mesh, comm_sampler = gmc.split_mpi_comm(MPI.COMM_WORLD, n_subdomains, n_processes)

    if comm_sampler.rank == 0:
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

    comm_sampler.barrier()

    settings = nonlinear_diffusion_reaction_settings()
    settings["output_path"] = output_path

    model, _ = nonlinear_diffusion_reaction_model(comm_mesh, settings)

    Vh = model.problem.Vh

    x_MAP = gmc.compute_MAP(model)
    map_eigenvalues, map_decoder, map_encoder = gmc.compute_Hessian_decomposition_at_sample(model, x_MAP, gauss_newton_approx = True, form_jacobian=True)

    LA_posterior = hp.GaussianLRPosterior(model.prior, map_eigenvalues, map_decoder)
    LA_posterior.mean = x_MAP[hp.PARAMETER]

    noise, m_prior, m0 = dl.Vector(comm_mesh), dl.Vector(comm_mesh), dl.Vector(comm_mesh)
    model.prior.init_vector(noise, "noise")
    model.prior.init_vector(m_prior, 0)
    model.prior.init_vector(m0, 0)

    for ii in range(comm_sampler.rank):
        hp.parRandom.normal(1., noise)
    hp.parRandom.normal(1., noise)
    LA_posterior.sample(noise, m_prior, m0)


    kernel = gmc.mMALAKernel(model, form_jacobian=True)

    index = np.arange(model.misfit.data.size)[::5]
    qoi = gmc.ObservableQoi(model.misfit.observable, index)

    tuned_step_size = gmc.step_size_tuning(comm_sampler, model, kernel, step_sizes, n_samples, output_path, 
                         m0=m0, n_burn_in = n_samples//2, qoi = qoi)

    if comm_sampler.rank == 0: np.save(output_path + "tuned_step_size.npy", tuned_step_size)
