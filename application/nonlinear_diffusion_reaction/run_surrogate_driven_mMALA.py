import numpy as np
import dolfin as dl
import torch
dl.set_log_level(dl.LogLevel.ERROR)
import sys, os, argparse
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.insert(0, '../../')
import geometric_mcmc as gmc
sys.path.insert(0, '../learning/')
from surrogate_training import FFN
from ndr_model import nonlinear_diffusion_reaction_model, nonlinear_diffusion_reaction_settings
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

    parser = argparse.ArgumentParser(description='nonlinear diffusion--reaction inference')

    parser.add_argument('--n_samples',
                        default=1000,
                        type=int,
                        help="Number of samples for each processer")

    parser.add_argument('--step_size',
                        default=0.1,
                        type=float,
                        help="The step size parameter in sampling")

    parser.add_argument('--process_id',
                        default=0,
                        type=int,
                        help="The processor ID")

    parser.add_argument('--output_path',
                        default="./mcmc_result/",
                        type=str,
                        help="The output path for saving")

    parser.add_argument('--output_frequency',
                        default=500,
                        type=int,
                        help="The number of samples for saving")
    
    parser.add_argument('--method',
                        default=0,
                        type=str,
                        help="The method to use in sampling")

    args = parser.parse_args()
    process_id = args.process_id
    n_samples = args.n_samples
    step_size = args.step_size
    output_path = args.output_path
    output_frequency = args.output_frequency

    output_path += "/chain_" + str(process_id) + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    settings = nonlinear_diffusion_reaction_settings()
    settings["output_path"] = output_path

    model, mtrue = nonlinear_diffusion_reaction_model(settings)

    Vh = model.problem.Vh

    with open("./training_result/surrogate_results.pkl", "rb") as f:
        data_file = pickle.load(f)
    
    input_encoder, output_encoder = data_file["input_encoder"], data_file["output_encoder"]
    input_decoder, output_decoder = data_file["input_decoder"].T, data_file["output_decoder"]

    input_dim = input_encoder.shape[0]
    output_dim = output_encoder.shape[0]

    width = [input_dim]
    for ii in range(data_file["depth"]):
        width.append(data_file["width"])
    width.append(output_dim)
    surrogate = FFN(width = width)
    surrogate.load_state_dict(torch.load("./training_result/surrogate.pt", map_location="cpu", weights_only=True))
    surrogate.eval()
    surrogate.to(data_file["device"])

    reduced_data = output_encoder@model.misfit.data

    surrogate_model = gmc.PyTorchSurrogateModel(surrogate, input_dim, output_dim,
                                                 data=reduced_data, device=data_file["device"])

    x_MAP, map_eigenvalues, map_decoder, map_encoder = gmc.compute_Hessian_decomposition_at_MAP(model, gauss_newton_approx = True)

    posterior = hp.GaussianLRPosterior(model.prior, map_eigenvalues, map_decoder)
    posterior.mean = x_MAP[hp.PARAMETER]

    kernels = []

    input_encoder_mv = hp.MultiVector(x_MAP[hp.PARAMETER], input_dim)
    input_decoder_mv = hp.MultiVector(x_MAP[hp.PARAMETER], input_dim)
    gmc.set_global_mv(model.prior.R.mpi_comm(), input_encoder, input_encoder_mv)
    gmc.set_global_mv(model.prior.R.mpi_comm(), input_decoder, input_decoder_mv)
    kernels.append(gmc.ReducedBasisSurrogatemMALAKernel(surrogate_model, parameter_encoder = input_encoder_mv, parameter_decoder = input_decoder_mv))
    kernels.append(gmc.DelayedAcceptanceKernel(model))

    kernels[0].parameters["h"] = step_size

    chain = gmc.MultiLevelDelayedAcceptanceMCMC(kernels)
    chain.parameters["number_of_samples"] = n_samples  # The total number of samples
    chain.parameters["print_progress"] = 20  # The number of time to print to screen
    chain.parameters["print_level"] = 1  if settings["verbose"] else -1

    for i in range(process_id):
        chain.consume_random()
    noise, m_prior, m0 = dl.Vector(), dl.Vector(), dl.Vector()
    model.prior.init_vector(m_prior, 0)
    model.prior.init_vector(m0, 0)
    model.prior.init_vector(noise, "noise")
    hp.parRandom.normal(1., noise)
    hp.parRandom.normal(1., noise)
    posterior.sample(noise, m_prior, m0, add_mean=True)

    qoi_index = np.arange((model.misfit.data.size))[::5]
    qoi = gmc.ObservableQoi(model.misfit.observable, qoi_index)

    param_fid = dl.XDMFFile(Vh[hp.PARAMETER].mesh().mpi_comm(), output_path + "parameter_samples.xdmf")
    param_fid.parameters["functions_share_mesh"] = True
    param_fid.parameters["rewrite_function_mesh"] = False

    tracer = gmc.FullTracer(Vh, parameter_file=param_fid, qoi_reference=model.misfit.data[qoi_index])
    tracer.parameters["output_frequency"] = output_frequency  # sample freqeucny for saving the visualization check,
    tracer.parameters["output_path"] = output_path
    tracer.parameters["moving_average_window_size"] = 500

    chain.run(m0, qoi=qoi, tracer=tracer)

    param_fid.close()
