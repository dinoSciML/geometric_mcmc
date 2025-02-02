import numpy as np
import dolfin as dl
dl.set_log_level(dl.LogLevel.ERROR)
import sys, os, argparse
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
from mpi4py import MPI
sys.path.insert(0, '../../')
import geometric_mcmc as gmc
from hyperelasticity_model import hyperelasticity_model, hyperelasticity_settings
import matplotlib.pyplot as plt
import matplotlib
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)

    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Running the reduction comparison')

    parser.add_argument('--n_samples',
                        default=500,
                        type=int,
                        help="Number of samples for derivative-informed subspace computation")

    parser.add_argument('--input_rank',
                        default=250,
                        type=int,
                        help="Number of ranks to keep in the input dimension reduction")
    
    parser.add_argument('--output_rank',
                        default=64,
                        type=int,
                        help="Number of ranks to keep in the output dimension reduction")

    parser.add_argument('--output_path',
                        default="./reduction_result/",
                        type=str,
                        help="The output path for saving the results")

    args = parser.parse_args()
    output_path = args.output_path
    input_rank = args.input_rank
    output_rank = args.output_rank
    n_samples = args.n_samples

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    mpi_comm = MPI.COMM_WORLD

    settings = hyperelasticity_settings()
    settings["output_path"] = output_path
    settings["save_setup"] = True

    model, _ = hyperelasticity_model(mpi_comm, settings)

    Vh = model.problem.Vh

    gmc.dimension_reduction_comparison(model, input_rank, output_rank, output_path, n_samples)