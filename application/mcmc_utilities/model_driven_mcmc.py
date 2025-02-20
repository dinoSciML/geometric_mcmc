import numpy as np
import dolfin as dl
dl.set_log_level(dl.LogLevel.ERROR)
import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.insert(0, '../../')
import geometric_mcmc as gmc
from mpi4py import MPI

method_list = ["pCN", "MALA", "LA-pCN", "DIS-MALA", "mMALA"]

def model_driven_mcmc_settings(settings = {}):
    settings["method"] = "mMALA"       # The method to use in sampling
    settings["n_samples"] = 5000        # Number of samples for each processer
    settings["tune_step_size"] = 0      # Whether to tune the step size. Only used in parallel MCMC run with more than one processor.
    settings["step_size_tuning"] = {}
    settings["step_size_tuning"]["step_size_max"] = 2       # The maximum step size for tuning. Used when tune_step_size is 1
    settings["step_size_tuning"]["step_size_min"] = 0.1     # The minimum step size for tuning. Used when tune_step_size is 1
    settings["step_size_tuning"]["n_samples"] = 4000 # The number of samples for tuning the step size
    settings["step_size_tuning"]["n_burn_in"] = 1000 # The burn-in period for tuning the step size
    settings["step_size"] = 0.1         # The step size parameter in sampling. Used when tune_step_size is 0 or in serial run
    settings["DIS-MALA"] = {}
    settings["DIS-MALA"]["n_dis_samples"] = 500     # Number of samples for the DIS eigenvalue and eigenvector estimation. Only used when method is DIS-mMALA
    settings["DIS-MALA"]["parameter_rank"] = 200    # The rank of the input dimension reduction. Used for both MAP Hessian approximation and DIS-mMALA
    settings["DIS-MALA"]["decoder_file"] = None    # The file for the decoder in the input dimension reduction if it is precomputed
    settings["DIS-MALA"]["encoder_file"] = None    # The file for the encoder in the input dimension reduction if it is precomputed
    settings["DIS-MALA"]["eigenvalues_file"] = None    # The file for the decoder in the input dimension reduction if it is precomputed
    settings["mMALA"] = {}
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
    return settings

def run_model_driven_mcmc(comm_sampler, model, mcmc_settings):

    assert any([mcmc_settings["method"]==method for method in method_list]), "The method is not supported. Choose from %s"%method_list

    comm_mesh = model.prior.R.mpi_comm()
    process_id = comm_sampler.rank

    if process_id == 0:
        if not os.path.exists(mcmc_settings["output_path"]):
            os.makedirs(mcmc_settings["output_path"], exist_ok=True)

    comm_sampler.Barrier()

    Vh = model.problem.Vh

    x_MAP = gmc.compute_MAP(model)
    map_eigenvalues, map_decoder, _ = gmc.compute_Hessian_decomposition_at_sample(model, x_MAP, gauss_newton_approx = False, rank=mcmc_settings["laplace_approximation_rank"])

    posterior = hp.GaussianLRPosterior(model.prior, map_eigenvalues, map_decoder)
    posterior.mean = x_MAP[hp.PARAMETER]

    if mcmc_settings["method"] == "pCN":
        kernel = gmc.pCNKernel(model)
    elif mcmc_settings["method"] == "MALA":
        kernel = gmc.MALAKernel(model)
    elif mcmc_settings["method"] == "LA-pCN":
        kernel = gmc.gpCNKernel(model, posterior)
    elif mcmc_settings["method"] == "DIS-MALA":
        if mcmc_settings["DIS-MALA"]["decoder_file"] is not None and mcmc_settings["DIS-MALA"]["encoder_file"] is not None and mcmc_settings["DIS-MALA"]["eigenvalues_file"] is not None:
            dis_input_decoder = gmc.load_mv_from_XDMF(Vh[hp.PARAMETER], mcmc_settings["DIS-MALA"]["decoder_file"], mcmc_settings["DIS-MALA"]["parameter_rank"])
            dis_input_encoder = gmc.load_mv_from_XDMF(Vh[hp.PARAMETER], mcmc_settings["DIS-MALA"]["encoder_file"], mcmc_settings["DIS-MALA"]["parameter_rank"])
            dis_input_eigenvalues = np.load(mcmc_settings["DIS-MALA"]["eigenvalues_file"])[:mcmc_settings["DIS-MALA"]["parameter_rank"]]
            gmc.check_orthonormality(dis_input_decoder, dis_input_encoder)
        else:
            dis_input_res = gmc.compute_DIS(model, mcmc_settings["DIS-MALA"]["n_dis_samples"], mcmc_settings["DIS-MALA"]["parameter_rank"])
            dis_input_eigenvalues, dis_input_decoder, dis_input_encoder = dis_input_res
        kernel = gmc.FixedmMALAKernel(model, dis_input_eigenvalues, dis_input_decoder, encoder= dis_input_encoder)
    elif mcmc_settings["method"] == "mMALA":
        kernel = gmc.mMALAKernel(model, form_jacobian=mcmc_settings["mMALA"]["form_jacobian"], 
                                 mode=mcmc_settings["mMALA"]["jacobian_mode"])

    noise, m_prior, m0 = dl.Vector(comm_mesh), dl.Vector(comm_mesh), dl.Vector(comm_mesh)
    model.prior.init_vector(m_prior, 0)
    model.prior.init_vector(m0, 0)
    model.prior.init_vector(noise, "noise")

    if mcmc_settings["tune_step_size"] and comm_sampler.size > 1:
        step_size_caniadates = np.exp(np.linspace(np.log(mcmc_settings["step_size_tuning"]["step_size_min"]), np.log(mcmc_settings["step_size_tuning"]["step_size_max"]), comm_sampler.size))
        for ii in range(step_size_caniadates.size+1): 
            hp.parRandom.normal(1., noise)
        posterior.sample(noise, m_prior, m0, add_mean=True)

        tuned_step_size = gmc.step_size_tuning(comm_sampler, model, kernel, step_size_caniadates, 
                                                mcmc_settings["step_size_tuning"]["n_samples"], mcmc_settings["output_path"],
                                                n_burn_in=mcmc_settings["step_size_tuning"]["n_burn_in"], 
                                                m0=m0)
        kernel.parameters["h"] = tuned_step_size
    else:
        kernel.parameters["h"] = mcmc_settings["step_size"]

    chain_path = mcmc_settings["output_path"] +  "/chain_" + str(process_id) + "/"
    if not os.path.exists(chain_path):
        os.makedirs(chain_path, exist_ok=True)

    chain = gmc.MCMC(kernel)
    chain.parameters["number_of_samples"] = mcmc_settings["n_samples"]  # The total number of samples
    chain.parameters["print_progress"] = 20  # The number of time to print to screen
    chain.parameters["print_level"] = 1  if mcmc_settings["verbose"] else -1

    for ii in range(process_id):
        hp.parRandom.normal(1., noise)
        chain.consume_random()

    hp.parRandom.normal(1., noise)
    posterior.sample(noise, m_prior, m0, add_mean=True)

    qoi = gmc.ObservableQoi(model.misfit.observable, mcmc_settings["qoi_index"])

    tracer = gmc.FullTracer(Vh[hp.PARAMETER], chain_path, qoi_reference=model.misfit.data[mcmc_settings["qoi_index"]])
    tracer.parameters["visual_frequency"] = mcmc_settings["output_frequency"]  # sample freqeucny for saving the visualization check,
    tracer.parameters["save_frequency"] = mcmc_settings["output_frequency"]
    tracer.parameters["moving_average_window"] = 500
    
    chain.run(m0, qoi=qoi, tracer=tracer)