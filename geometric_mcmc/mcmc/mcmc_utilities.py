import numpy as np
import os, sys
import dolfin as dl
import hippylib as hp
import time
import h5py
import warnings
from hippylib.modeling.variables import PARAMETER
from .tracer import FullTracer, ReducedTracer
from .chain import MCMC, MultiLevelDelayedAcceptanceMCMC
from .diagnostic import SingleChainESS

def find_first_valley(arr):
    """
    A function to find the idx of the first non-decreasing value in an array
    :param arr: The array
    :return: The index of the first non-positive element
    """
    arr = np.diff(arr)
    out = np.where(np.diff(arr) <= 0.0)[0]
    return out[0] + 1 if out.size > 0 else 1

def step_size_tuning(comm_sampler, model, kernel, step_sizes, n_samples, output_path, 
                     m0 = None, n_burn_in = None, qoi = None):
    """
    A function to tune the step size for the MCMC step size for the given kernel.
    :param comm_sampler: The MPI communicator for the sampler. Should have the same size as the number of step sizes
    :param model: The hippylib model object
    :param kernel: The kernel object for the MCMC or a list of kernel objects for the multilevel delayed acceptance scheme
    :param step_sizes: The step sizes to tune. Should be a 2D numpy array with shape (n_levels, n_processors). A list will be converted to a numpy array
    :param n_samples: The total number of samples to generate
    :param output_path: The output path for saving the data
    :param m0: The initial parameter :code:`dolfin.Vector`. If None, a random prior sample will be generated as the initial parameter
    :param n_burn_in: The burn-in period. If None, half of the total samples will be used
    :param qoi: The quantity of interest object.
    :return: The tuned step size
    """
    time0 = time.time()

    if isinstance(step_sizes, list):
        step_sizes = np.array(step_sizes)
    if step_sizes.ndim == 1:
        step_sizes = np.expand_dims(step_sizes, axis=0)
    if isinstance(kernel, list) and step_sizes.shape[0] != len(kernel)-1:
        raise ValueError("For multilevel delayed acceptance scheme, the number of rows in step sizes should be the same as the number of kernels minus one." +
                         "Got {0} step sizes and {1} kernels".format(step_sizes.shape[0], len(kernel)))
    assert comm_sampler.size == step_sizes.shape[1], "The number of processors should be the same as the number of step sizes"
    assert np.all(np.diff(step_sizes, axis=1) >= 0), "The step sizes should be in non-decreasing order. For multilevel schemes with more than 2 leveles, each level should have non-decreasing step sizes."

    if comm_sampler.rank == 0: np.save(output_path + "step_sizes.npy", step_sizes)

    output_path += "size_%d/"%(comm_sampler.rank)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if n_burn_in is None: 
        n_burn_in = n_samples // 2
        warnings.warn("Burn-in period is not set. Using half of the total number of samples as the burn-in period")

    comm_mesh = model.prior.R.mpi_comm()
    
    if isinstance(kernel, list):
        for ii, ker in range(kernel[:-1]):
            ker[ii].parameters["h"] = step_sizes[ii, comm_sampler.rank]
        flag_reduced = kernel[-1].reduced()
        chain = MultiLevelDelayedAcceptanceMCMC(kernel)
    else:
        kernel.parameters["h"] = step_sizes[0, comm_sampler.rank]
        flag_reduced = kernel.reduced()
        chain = MCMC(kernel)

    chain.parameters["number_of_samples"] = n_samples  # The total number of samples
    chain.parameters["print_progress"] = 20  # The number of time to print to screen
    chain.parameters["print_level"] = -1

    noise = dl.Vector(comm_mesh)
    model.prior.init_vector(noise, "noise")

    for ii in range(comm_sampler.rank):
        chain.consume_random()
        if m0 is None: hp.parRandom.normal(1., noise)

    if m0 is None:
        m0 = dl.Vector(comm_mesh)
        model.prior.init_vector(m0, 0)
        hp.parRandom.normal(1., noise)
        model.prior.sample(noise, m0)

    if flag_reduced:
        tracer = ReducedTracer(model.problem.Vh[PARAMETER], output_path=output_path)
        tracer.parameters["visual_frequency"] = n_samples // 5
        tracer.parameters["save_frequency"] = n_samples // 5
    else:
        tracer = FullTracer(model.problem.Vh[PARAMETER], output_path=output_path)
        tracer.parameters["visual_frequency"] = n_samples // 5
        tracer.parameters["save_frequency"] = n_samples // 5

    chain.run(m0, qoi=qoi, tracer=tracer)
    time1 = time.time()

    print("Process %d Finished sampling with step size tuning in %.2f minutes" % (comm_sampler.rank, (time1-time0)/60))
    sys.stdout.flush()

    with h5py.File(output_path + "mcmc_data.hdf5", "r") as f:
        n_samples = f.attrs["n_samples"]
        accept = f["accept"][:]
        samples = f["parameter"][:]
        jump = f["jump"][1:]

    if isinstance(kernel, list):
        accept_rate = np.mean(np.min(np.abs(accept[n_burn_in:]), axis=1))
    else:
        accept_rate = np.mean(accept[n_burn_in:])

    mean_square_jump = np.mean(jump[n_burn_in:]**2)
    ess = SingleChainESS(samples[n_burn_in:, :])

    accept_rate = comm_sampler.gather(accept_rate, root=0)
    mean_square_jump = comm_sampler.gather(mean_square_jump, root=0)
    ess_median = comm_sampler.gather(np.median(ess), root=0)
    if comm_sampler.rank == 0:
        print("accept_rate: ", accept_rate)
        print("mean_square_jump: ", mean_square_jump)
        print("ess_median: ", ess_median)
        print("step_sizes: ", step_sizes)
        idx = find_first_valley(accept_rate)
        print("truncation idx: ", idx)
        msj_idx = np.argmax(mean_square_jump[:idx])
        print("max mean square jump idx: ", msj_idx)
        ess_idx = np.argmax(ess_median[:idx])
        print("max ess idx: ", ess_idx)
        if msj_idx == ess_idx:
            return step_sizes[msj_idx, :]
        else:
            idx_out = (msj_idx + ess_idx) // 2
            return step_sizes[idx_out, :]