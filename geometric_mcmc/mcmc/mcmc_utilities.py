import numpy as np
import os
import dolfin as dl
import hippylib as hp
import time
import h5py
import warnings
import pickle
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
    if isinstance(arr, list):
        arr = np.array(arr)
    out = np.where(np.diff(arr) >= 0.0)[0]
    return out[0]+1 if out.size > 0 else arr.size

def step_size_tuning(comm_sampler, model, kernel, step_sizes, n_samples, output_path, 
                     m0 = None, n_burn_in = None, qoi = None, verbose = False, output_frequency=100):
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
    :param verbose: The verbosity flag
    :param output_frequency: The frequency to save the data
    :return: The tuned step size
    """
    comm_mesh = model.prior.R.mpi_comm() # Get the MPI communicator for the mesh

    # Check the step size input: must be non-decreasing order (if multilevel with level > 2, then non-decreasing in each level)
    if isinstance(step_sizes, list): # Convert list to numpy array
        step_sizes = np.array(step_sizes) 
    if step_sizes.ndim == 1: # Convert 1D array to 2D array
        step_sizes = np.expand_dims(step_sizes, axis=0)
    if isinstance(kernel, list) and step_sizes.shape[0] != len(kernel)-1: # Check the number of rows in step sizes
        raise ValueError("For multilevel delayed acceptance scheme, the number of rows in step sizes should be the same as the number of kernels minus one." +
                         "Got {0} step sizes and {1} kernels".format(step_sizes.shape[0], len(kernel)))
    assert comm_sampler.size >= step_sizes.shape[1], "The number of processors should be the same as the number of step sizes"
    assert np.all(np.diff(step_sizes, axis=1) >= 0), "The step sizes should be in non-decreasing order. For multilevel schemes with more than 2 leveles, each level should have non-decreasing step sizes."


    if comm_sampler.rank >= step_sizes.shape[1]:
        tuned_step_size = None
        accept_rate = 0.0
        mean_square_jump = 0.0
        ess_median = 0.0
    else:
        # Add additional folders in the output path and save mcmc data there
        if comm_mesh.rank == 0 and not os.path.exists(output_path + "size_%d/"%(comm_sampler.rank)): # Create the output path if it does not exist at the zero mesh rank
            os.makedirs(output_path + "size_%d/"%(comm_sampler.rank), exist_ok=True) 

        if n_burn_in is None: # Set the burn-in period if not given
            n_burn_in = n_samples // 2
            warnings.warn("Burn-in period is not set. Using half of the total number of samples as the burn-in period")
        
        if isinstance(kernel, list): # Check if the kernel is a list, i.e., multilevel delayed acceptance scheme
            for ii, ker in enumerate(kernel[:-1]):
                ker.parameters["h"] = step_sizes[ii, comm_sampler.rank] # Set the step size for each level
            flag_reduced = kernel[-1].reduced() # Check if the last kernel is a reduced kernel with numpy data structure
            chain = MultiLevelDelayedAcceptanceMCMC(kernel) # Create the multilevel delayed acceptance MCMC object
        else:
            kernel.parameters["h"] = step_sizes[0, comm_sampler.rank] # Set the step size for the kernel
            flag_reduced = kernel.reduced() # Check if the kernel is a reduced kernel with numpy data structure
            chain = MCMC(kernel) # Create the MCMC object

        chain.parameters["number_of_samples"] = n_samples  # The total number of samples
        chain.parameters["print_progress"] = 20  # The number of time to print to screen
        chain.parameters["print_level"] = 1 if verbose else -1 # The level to print to screen, -1 means no print

        noise = dl.Vector(comm_mesh) # Create a dolfin vector for the noise
        model.prior.init_vector(noise, "noise") # Initialize the noise vector

        for ii in range(comm_sampler.rank): # Consumes random samples for the given sampler rank
            chain.consume_random()
            if m0 is None: hp.parRandom.normal(1., noise)

        if m0 is None: # Generate a random prior sample if the initial parameter is not given
            m0 = dl.Vector(comm_mesh)
            model.prior.init_vector(m0, 0)
            hp.parRandom.normal(1., noise)
            model.prior.sample(noise, m0)

        if flag_reduced: # Check if the kernel is a reduced kernel with numpy data structure
            tracer = ReducedTracer(model.problem.Vh[PARAMETER], output_path=output_path + "size_%d/"%(comm_sampler.rank))
        else: # Create a full tracer object
            tracer = FullTracer(model.problem.Vh[PARAMETER], output_path=output_path + "size_%d/"%(comm_sampler.rank))
        tracer.parameters["visual_frequency"] = output_frequency
        tracer.parameters["save_frequency"] = output_frequency
        tracer.parameters["moving_average_window"] = min(100, n_samples//2)

        time0 = time.time() # Start the timer
        chain.run(m0, qoi=qoi, tracer=tracer) # Run the MCMC chain
        time1 = time.time() # Stop the timer

        if comm_mesh.rank==0 and verbose: print("Process %d Finished sampling with step size tuning in %.2f minutes" % (comm_sampler.rank, (time1-time0)/60)) # Print the time taken

        with h5py.File(output_path + "size_%d/"%(comm_sampler.rank) + "mcmc_data.hdf5", "r") as f: # Load the MCMC data
            n_samples = f.attrs["n_samples"]
            accept = f["accept"][:]
            samples = f["parameter"][:]
            jump = f["jump"][1:]

        # Compute the acceptance rate. Note that multilevel delayed acceptance scheme has multiple acceptance rates and need to be treated differently
        accept_rate = np.mean(np.min(np.abs(accept[n_burn_in:]), axis=1)) if isinstance(kernel, list) else np.mean(accept[n_burn_in:])
        
        # Compute the mean square jump and the effective sample size
        mean_square_jump = np.mean(jump[n_burn_in:]**2)
        ess = SingleChainESS(samples[n_burn_in:, :])

        # Gather the data from all processors
        accept_rate = np.array(comm_sampler.gather(accept_rate, root=0))
        mean_square_jump = np.array(comm_sampler.gather(mean_square_jump, root=0))
        ess_median = np.array(comm_sampler.gather(np.median(ess), root=0))

        # Compute the optimal step size on the root sampler processor
        if comm_sampler.rank == 0:
            # Finding the first entry with either an increase in acceptance rate or a decrease in mean square jump or a decrease in ESS
            idx = min([find_first_valley(accept_rate), find_first_valley(-mean_square_jump)])
            # Finding the optimal step size index
            ess_idx = np.argmax(ess_median[:idx])
            tuned_step_size = step_sizes[:, ess_idx] # The tuned step size
            
            tuning_results = {} # Save the tuning results in a dictionary
            tuning_results["step_sizes_candidates"] = step_sizes
            tuning_results["n_samples"] = n_samples
            tuning_results["n_burn_in"] = n_burn_in
            tuning_results["accept_rate"] = accept_rate[:step_sizes.shape[1]]
            tuning_results["mean_square_jump"] = mean_square_jump[:step_sizes.shape[1]]
            tuning_results["median_ESS_percentage"] = ess_median[:step_sizes.shape[1]]
            tuning_results["tuned_step_size"] = tuned_step_size
            tuning_results["index"] = ess_idx
            
            if (comm_mesh.rank + comm_sampler.rank) == 0: # Save the tuning results
                with open(output_path + "step_size_tuning_results.pkl", "wb") as f:
                    pickle.dump(tuning_results, f)
        else:
            tuned_step_size = None
        
        
    return comm_sampler.bcast(tuned_step_size, root=0) # Broadcast the tuned step size to all processors and return