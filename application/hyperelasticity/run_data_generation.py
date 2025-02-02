import numpy as np
import dolfin as dl
dl.set_log_level(dl.LogLevel.ERROR)
import sys, os, argparse
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
from mpi4py import MPI
sys.path.insert(0, '../../')
import geometric_mcmc as gmc
sys.path.insert(0, '../learning_utilities/')
from data_generation import generate_data_serial, data_generation_settings
from hyperelasticity_model import hyperelasticity_model, hyperelasticity_settings
import matplotlib.pyplot as plt
import matplotlib
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass

if __name__=="__main__":

    data_settings = data_generation_settings()
    data_settings["n_samples"] = 5000
    data_settings["n_input_bases"] = 200
    data_settings["n_output_bases"] = 64
    data_settings["n_dis_samples"] = 2000
    data_settings["output_path"] = "./data/"
    data_settings["verbose"] = True

    model_settings = hyperelasticity_settings()
    model_settings["output_path"] = "./data/"
    model_settings["save_setup"] = True

    comm_mesh, comm_sampler = gmc.split_mpi_comm(MPI.COMM_WORLD, 1, 1)

    model, _ = hyperelasticity_model(comm_mesh, model_settings)
    generate_data_serial(model, data_settings)