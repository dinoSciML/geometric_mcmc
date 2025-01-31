import numpy as np
import dolfin as dl
from mpi4py import MPI
import torch
dl.set_log_level(dl.LogLevel.ERROR)
import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.insert(0, '../../')
import geometric_mcmc as gmc
sys.path.insert(0, '../learning_utilities/')
from surrogate_training import FFN
import pickle

def surrogate_driven_mcmc_settings(settings = {}):
    settings["n_samples"] = 10000
    settings["n_subdomains"] = 1
    settings["tune_step_size"] = 0
    settings["step_size_tuning"] = {}
    settings["step_size_tuning"]["step_size_max"] = 2
    settings["step_size_tuning"]["step_size_min"] = 0.1
    settings["step_size_tuning"]["n_samples"] = 2000
    settings["step_size_tuning"]["n_burn_in"] = 1000
    settings["step_size"] = 0.1
    settings["output_frequency"] = 100
    settings["output_path"] = "./DA-DINO-mMALA_results/"
    settings["surrogate"] = {}
    settings["surrogate"]["model_file"] = "./training_result/surrogate.pt"
    settings["surrogate"]["data_file"] = "./training_result/surrogate_results.pkl"
    settings["qoi_index"] = [0, -1]
    return settings

def run_surrogate_driven_mMALA(comm_sampler, model, mcmc_settings):
    
    Vh = model.problem.Vh
    comm_mesh = model.prior.R.mpi_comm()

    with open(mcmc_settings["surrogate"]["data_file"], "rb") as f:
        data_file = pickle.load(f)
    
    input_encoder, output_encoder = data_file["input_encoder"], data_file["output_encoder"]
    input_decoder, output_decoder = data_file["input_decoder"].T, data_file["output_decoder"]
    output_mean_shift = data_file["observable_mean_shift"]

    input_dim = input_encoder.shape[0]
    output_dim = output_encoder.shape[0]

    width = [input_dim]
    for ii in range(data_file["depth"]):
        width.append(data_file["width"])
    width.append(output_dim)
    surrogate = FFN(width = width)
    surrogate.load_state_dict(torch.load(mcmc_settings["surrogate"]["model_file"], map_location="cpu"))
    surrogate.eval()
    surrogate.to(data_file["device"])

    reduced_data = output_encoder@(model.misfit.data - output_mean_shift)

    surrogate_model = gmc.PyTorchSurrogateModel(surrogate, input_dim, output_dim,
                                                 data=reduced_data, device=data_file["device"])

    x_MAP = gmc.compute_MAP(model, rel_tolerance=1e-10)
    map_eigenvalues, map_decoder, _ = gmc.compute_Hessian_decomposition_at_sample(model, x_MAP, gauss_newton_approx = False, rank=input_dim)

    posterior = hp.GaussianLRPosterior(model.prior, map_eigenvalues, map_decoder)
    posterior.mean = x_MAP[hp.PARAMETER]

    kernels = []

    input_encoder_mv = hp.MultiVector(x_MAP[hp.PARAMETER], input_dim)
    input_decoder_mv = hp.MultiVector(x_MAP[hp.PARAMETER], input_dim)
    gmc.set_global_mv(model.prior.R.mpi_comm(), input_encoder, input_encoder_mv)
    gmc.set_global_mv(model.prior.R.mpi_comm(), input_decoder, input_decoder_mv)
    kernels.append(gmc.ReducedBasisSurrogatemMALAKernel(surrogate_model, parameter_encoder = input_encoder_mv, parameter_decoder = input_decoder_mv))
    kernels.append(gmc.DelayedAcceptanceKernel(model))

    noise, m_prior, m0 = dl.Vector(comm_mesh), dl.Vector(comm_mesh), dl.Vector(comm_mesh)
    model.prior.init_vector(m_prior, 0)
    model.prior.init_vector(m0, 0)
    model.prior.init_vector(noise, "noise")

    if mcmc_settings["tune_step_size"]:
        step_size_caniadates = np.linspace(mcmc_settings["step_size_tuning"]["step_size_min"], mcmc_settings["step_size_tuning"]["step_size_max"], comm_sampler.size)
        for ii in range(step_size_caniadates.size+1):
            hp.parRandom.normal(1., noise)
        posterior.sample(noise, m_prior, m0, add_mean=True)
        tuned_step_size = gmc.step_size_tuning(comm_sampler, model, kernels, 
                                        step_size_caniadates, mcmc_settings["step_size_tuning"]["n_samples"],
                                        mcmc_settings["output_path"], n_burn_in=mcmc_settings["step_size_tuning"]["n_burn_in"], m0=m0)
        kernels[0].parameters["h"] = tuned_step_size
    else:
        kernels[0].parameters["h"] = mcmc_settings["step_size"]

    chain_path = mcmc_settings["output_path"] + "/chain_" + str(comm_sampler.rank) + "/"
    if not os.path.exists(chain_path):
        os.makedirs(chain_path, exist_ok=True)

    chain = gmc.MultiLevelDelayedAcceptanceMCMC(kernels)
    chain.parameters["number_of_samples"] = mcmc_settings["n_samples"]  # The total number of samples
    chain.parameters["print_progress"] = 20  # The number of time to print to screen
    chain.parameters["print_level"] = 1  if mcmc_settings["verbose"] else -1

    for i in range(comm_sampler.rank):
        hp.parRandom.normal(1., noise)
        chain.consume_random()
    hp.parRandom.normal(1., noise)
    posterior.sample(noise, m_prior, m0, add_mean=True)

    qoi = gmc.ObservableQoi(model.misfit.observable, mcmc_settings["qoi_index"])

    tracer = gmc.FullTracer(Vh[hp.PARAMETER], chain_path, qoi_reference=model.misfit.data[mcmc_settings["qoi_index"]])
    tracer.parameters["save_frequency"] = mcmc_settings["output_frequency"]  # sample freqeucny for saving the visualization check,
    tracer.parameters["visual_frequency"] = mcmc_settings["output_frequency"]
    tracer.parameters["moving_average_window_size"] = 500

    chain.run(m0, qoi=qoi, tracer=tracer)