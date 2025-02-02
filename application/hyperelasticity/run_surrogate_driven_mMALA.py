import numpy as np
import dolfin as dl
from mpi4py import MPI
import torch
dl.set_log_level(dl.LogLevel.ERROR)
import sys, os, argparse
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.insert(0, '../../')
import geometric_mcmc as gmc
sys.path.insert(0, '../learning_utilities/')
from surrogate_training import FFN
sys.path.insert(0, '../mcmc_utilities/')
from surrogate_driven_mcmc import surrogate_driven_mcmc_settings, run_surrogate_driven_mMALA
from hyperelasticity_model import hyperelasticity_model, hyperelasticity_settings
import matplotlib.pyplot as plt
import matplotlib
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass
import pickle

if __name__ == "__main__":

    settings = surrogate_driven_mcmc_settings()
    settings["n_samples"] = 10000
    settings["n_subdomains"] = 1
    settings["tune_step_size"] = 1
    settings["step_size_tuning"]["step_size_max"] = 4
    settings["step_size_tuning"]["step_size_min"] = 0.5
    settings["step_size_tuning"]["n_samples"] = 2000
    settings["step_size_tuning"]["n_burn_in"] = 1000
    settings["step_size"] = 0.1
    settings["output_frequency"] = 200
    settings["output_path"] = "./DA-DINO-mMALA_results/"
    settings["surrogate"]["model_file"] = "./training_result/surrogate.pt"
    settings["surrogate"]["data_file"] = "./training_result/surrogate_results.pkl"
    settings["verbose"] = 0
    settings["qoi_index"] = [0, -1]

    comm_mesh, comm_sampler = gmc.split_mpi_comm(MPI.COMM_WORLD, settings["n_subdomains"], MPI.COMM_WORLD.size//settings["n_subdomains"])

    model_settings = hyperelasticity_settings()
    model_settings["output_path"] = settings["output_path"]
    model_settings["save_setup"] = False
    model_settings["verbose"] = False

    model, _ = hyperelasticity_model(comm_mesh, model_settings)

    if comm_sampler.rank == 0:
        with open(settings["output_path"] + "mcmc_settings.pkl", "wb") as f:
            pickle.dump(settings, f)
        with open(settings["output_path"] + "model_settings.pkl", "wb") as f:
            pickle.dump(model_settings, f)

    run_surrogate_driven_mMALA(comm_sampler, model, settings)