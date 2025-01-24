
import numpy as np
import dolfin as dl
dl.set_log_level(dl.LogLevel.ERROR)
import sys, os, argparse
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.insert(0, '../../')
import geometric_mcmc as gmc
from ndr_model import nonlinear_diffusion_reaction_model, nonlinear_diffusion_reaction_settings

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

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
                        default=50,
                        type=int,
                        help="The number of samples for saving")

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

    model, _ = nonlinear_diffusion_reaction_model(settings)

    Vh = model.problem.Vh

    x_MAP, map_eigenvalues, map_decoder, map_encoder = gmc.compute_GNH_at_MAP(model)

    map_decoder.export(Vh[hp.PARAMETER], output_path + "map_gnh_encoder.xdmf", normalize=True)
    map_encoder.export(Vh[hp.PARAMETER], output_path + "map_gnh_decoder.xdmf", normalize=True)

    plt.figure(figsize=(5,4))
    plt.semilogy(map_eigenvalues)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(":")
    plt.savefig(output_path + "map_gnh_eigenvalue.pdf", bbox_inches="tight")
    plt.close()

    posterior = hp.GaussianLRPosterior(model.prior, map_eigenvalues, map_decoder)
    posterior.mean = x_MAP[hp.PARAMETER]

    kernel = gmc.mMALAKernel(model)
    kernel.parameters["h"] = step_size

    chain = gmc.MCMC(kernel)
    chain.parameters["number_of_samples"] = n_samples  # The total number of samples
    chain.parameters["print_progress"] = 20  # The number of time to print to screen
    if settings["verbose"]:
        chain.parameters["print_level"] = 1  # Negative to not print
    else:
        chain.parameters["print_level"] = -1  # Negative to not print

    for i in range(process_id):
        chain.consume_random()
    noise, m_prior, m0 = dl.Vector(), dl.Vector(), dl.Vector()
    model.prior.init_vector(m_prior, 0)
    model.prior.init_vector(m0, 0)
    model.prior.init_vector(noise, "noise")
    hp.parRandom.normal(1., noise)
    posterior.sample(noise, m_prior, m0, add_mean=True)
    if settings["plot"]:
        cbar = dl.plot(hp.vector2Function(m0, Vh[hp.PARAMETER]))
        plt.colorbar(cbar)
        plt.axis("off")
        plt.title(r"Initial sample")
        plt.savefig(output_path + "initial_sample.pdf", bbox_inches="tight")
        plt.close()

    qoi = hp.NullQoi()

    param_fid = dl.XDMFFile(Vh[hp.PARAMETER].mesh().mpi_comm(), output_path + "parameter_samples.xdmf")
    param_fid.parameters["functions_share_mesh"] = True
    param_fid.parameters["rewrite_function_mesh"] = False

    tracer = gmc.FullTracer(Vh, parameter_file = param_fid)
    tracer.parameters["output_frequency"] = output_frequency  # sample freqeucny for saving the visualization check,
    tracer.parameters["output_path"] = output_path

    chain.run(m0, qoi=qoi, tracer=tracer)