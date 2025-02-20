import numpy as np
import dolfin as dl
dl.set_log_level(dl.LogLevel.ERROR)
import sys, os, argparse
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.insert(0, '../../')
import geometric_mcmc as gmc
sys.path.insert(0, '../mcmc_utilities/')
from model_driven_mcmc import model_driven_mcmc_settings, run_model_driven_mcmc
from ndr_model import nonlinear_diffusion_reaction_model, nonlinear_diffusion_reaction_settings
import matplotlib.pyplot as plt
import matplotlib
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass

from mpi4py import MPI
import pickle

if __name__ == "__main__":
    settings = model_driven_mcmc_settings()
    settings["method"] = "mMALA"       # The method to use in sampling. Can be "pCN", "MALA", "LA-pCN", "DIS-MALA", "mMALA"
    settings["n_samples"] = 10000        # Number of samples for each processer
    settings["tune_step_size"] = 0      # Whether to tune the step size
    settings["step_size_tuning"]["step_size_max"] = 2       # The maximum step size for tuning. Used when tune_step_size is 1
    settings["step_size_tuning"]["step_size_min"] = 0.1     # The minimum step size for tuning. Used when tune_step_size is 1
    settings["step_size_tuning"]["n_samples"] = 4000 # The number of samples for tuning the step size
    settings["step_size_tuning"]["n_burn_in"] = 1000 # The burn-in period for tuning the step size
    settings["step_size"] = 0.1         # The step size parameter in sampling
    settings["DIS-MALA"]["n_dis_samples"] = 500     # Number of samples for the DIS eigenvalue and eigenvector estimation. Only used when method is DIS-mMALA
    settings["DIS-MALA"]["parameter_rank"] = 200    # The rank of the input dimension reduction. Used for both MAP Hessian approximation and DIS-mMALA
    settings["DIS-MALA"]["decoder_file"] = "./reduction_result/dis_input_decoder.xdmf"    # The file for the decoder in the input dimension reduction if it is precomputed
    settings["DIS-MALA"]["encoder_file"] = "./reduction_result/dis_input_encoder.xdmf"    # The file for the encoder in the input dimension reduction if it is precomputed
    settings["DIS-MALA"]["eigenvalues_file"] = "./reduction_result/dis_input_eigenvalues.npy"    # The file for the decoder in the input dimension reduction if it is precomputed
    settings["mMALA"]["gauss_newton_approximation"] = False # Whether to use Gauss-Newton approximation for the Hessian
    settings["mMALA"]["form_jacobian"] = True    # Whether to form the Jacobian matrix
    settings["mMALA"]["jacobian_mode"] = "reverse" # The mode for forming the Jacobian matrix
    settings["mMALA"]["parameter_rank"] = 200    # The rank of the input dimension reduction. Used for both low rank Hessian approximation.
    settings["laplace_approximation_rank"] = 200    # The rank of the input dimension reduction. Used MAP Hessian approximation
    settings["n_subdomains"] = 1        # The size of the mesh MPI communicator
    settings["output_path"] = "./" + settings["method"] + "_results/" # The output path for saving
    settings["output_frequency"] = 100  # The number of samples for saving
    settings["verbose"] = 0             # whether to print the information
    settings["qoi_index"] = [0, -1]         # The index of observables to track as qoi during MCMC

    comm_mesh, comm_sampler = gmc.split_mpi_comm(MPI.COMM_WORLD, settings["n_subdomains"], MPI.COMM_WORLD.size//settings["n_subdomains"])

    if comm_sampler.rank == 0:
        if not os.path.exists(settings["output_path"]):
            os.makedirs(settings["output_path"], exist_ok=True)
    
    comm_sampler.Barrier()

    model_settings = nonlinear_diffusion_reaction_settings()
    model_settings["output_path"] = settings["output_path"]
    model_settings["save_setup"] = False
    model_settings["verbose"] = settings["verbose"]

    model, _ = nonlinear_diffusion_reaction_model(comm_mesh, model_settings)

    if comm_sampler.rank == 0:
        with open(settings["output_path"] + "mcmc_settings.pkl", "wb") as f:
            pickle.dump(settings, f)
        with open(settings["output_path"] + "model_settings.pkl", "wb") as f:
            pickle.dump(model_settings, f)

    run_model_driven_mcmc(comm_sampler, model, settings)