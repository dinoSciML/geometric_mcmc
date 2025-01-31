import numpy as np
import dolfin as dl
import hippylib as hp
import geometric_mcmc as gmc
import os
from mpi4py import MPI
import pickle
import time

def data_generation_settings(settings = {}):
    settings["n_samples"] = 20000
    settings["n_input_bases"] = 200
    settings["n_output_bases"] = 25
    settings["n_dis_samples"] = 1000
    settings["output_path"] = "./data/"
    settings["verbose"] = True
    return settings

def generate_data_serial(model, settings):

    comm_mesh = model.prior.R.mpi_comm()
    if comm_mesh.size != MPI.COMM_WORLD.size: raise Exception("This function should be run in sample serial mode. The bases computation is not shared among MPI processes.")

    if not os.path.exists(settings["output_path"]):
        os.makedirs(settings["output_path"], exist_ok=True)

    assert settings["n_samples"] >= settings["n_dis_samples"], "Number of samples for derivative-informed subspace computation should be less than the total number of samples"
    

    pto_map_jac = gmc.PtOMapJacobian(model.problem, model.misfit.observable)
    x = model.generate_vector()
    noise = dl.Vector()
    model.prior.init_vector(noise, "noise")
    param_samples = []
    observable_samples = []
    jacobian_samples = []

    if comm_mesh.rank == 0 and settings["verbose"]: print("Generating data set")
    time_forward = 0
    time_jacobian = 0
    for ii in range(settings["n_dis_samples"]):
        hp.parRandom.normal(1.0, noise)
        model.prior.sample(noise, x[hp.PARAMETER])
        time0 = time.time()
        model.solveFwd(x[hp.STATE], x)
        time_forward += time.time() - time0
        param_samples.append(x[hp.PARAMETER].copy())
        observable_samples.append(model.misfit.observable.eval(x))
        time0 = time.time()
        pto_map_jac.setLinearizationPoint(x)
        jacobian_samples.append(pto_map_jac.generate_jacobian())
        pto_map_jac.eval(jacobian_samples[-1], "reverse")
        time_jacobian += time.time() - time0
        if comm_mesh.rank == 0 and settings["verbose"] and ((ii+1) % (settings["n_samples"]//20) == 0 or ii+1 == settings["n_dis_samples"]): 
            print("%d samples generated (%.2f %%)" % (ii+1, (ii+1)/settings["n_samples"]*100))
            print("Average time for forward solve: %.6f "%( time_forward/ii ))
            print("Average time for full Jacobian computation: %.6f "%( time_jacobian/ii ))

    if comm_mesh.rank == 0 and settings["verbose"]: print("Computing the derivative-informed subspace at %d samples" % settings["n_dis_samples"])
    time0 = time.time()

    input_res, output_res = gmc.compute_DIS_from_samples(model, jacobian_samples, settings["n_input_bases"], settings["n_output_bases"], oversampling=20)
    if comm_mesh.rank == 0 and settings["verbose"]: print("Time for computing the derivative-informed subspace: ", time.time()-time0)
    input_eigenvalues, input_decoder, input_encoder = input_res
    output_eigenvalues, output_decoder, output_encoder = output_res

    reduced_jacobian_samples = np.zeros((settings["n_samples"], settings["n_output_bases"], settings["n_input_bases"]))
    reduced_jacobian_samples[:settings["n_dis_samples"]] = np.stack([output_encoder@(jacobian.dot_mv(input_decoder)) for jacobian in jacobian_samples])

    if comm_mesh.rank == 0 and settings["verbose"]: print("Generating the rest of the samples")
    help_param = model.generate_vector(hp.PARAMETER)
    time_forward = 0
    time_jacobian = 0
    for ii in range(settings["n_samples"] - settings["n_dis_samples"]):
        hp.parRandom.normal(1.0, noise)
        model.prior.sample(noise, x[hp.PARAMETER])
        time0 = time.time()
        model.solveFwd(x[hp.STATE], x)
        time_forward += time.time() - time0
        param_samples.append(x[hp.PARAMETER].copy())
        observable_samples.append(model.misfit.observable.eval(x))
        time0 = time.time()
        pto_map_jac.setLinearizationPoint(x)
        for jj in range(settings["n_output_bases"]):
            help_param.zero()
            pto_map_jac.transpmult(output_encoder[jj, :], help_param)
            reduced_jacobian_samples[ii + settings["n_dis_samples"], jj, :] = input_decoder.dot_v(help_param)
        time_jacobian += time.time() - time0
        if comm_mesh.rank == 0 and settings["verbose"] and ((ii + settings["n_dis_samples"]+1) % (settings["n_samples"]//20) == 0): 
            print("%d samples generated (%.2f %%)" % (ii + settings["n_dis_samples"]+ 1, (ii+settings["n_dis_samples"]+1)/settings["n_samples"]*100))
            print("Average time for forward solve: %.6f "%( time_forward/ii ))
            print("Average time for reduced Jacobian computation: %.6f "%( time_jacobian/ii ))

    for ii in range(settings["n_samples"]):
        param_samples[ii] = gmc.get_global(comm_mesh, param_samples[ii], root=0)
    
    if model.problem.Vh[hp.PARAMETER].ufl_element().degree() == 1:
        to_vertex = gmc.get_vertex_order(model.problem.Vh[hp.PARAMETER], root=0)
    else:
        to_vertex = None

    if comm_mesh.rank == 0:
        data_file = settings
        data_file["parameter"] = np.stack(param_samples)
        data_file["to_vertex_order"] = to_vertex
        data_file["observable"] = np.vstack(observable_samples)
        data_file["reduced_jacobian"] = reduced_jacobian_samples
        data_file["input_eigenvalues"] = input_eigenvalues
        data_file["input_decoder"] = gmc.get_global_mv(comm_mesh, input_decoder).T
        data_file["input_encoder"] = gmc.get_global_mv(comm_mesh, input_encoder)
        data_file["output_eigenvalues"] = output_eigenvalues
        data_file["output_decoder"] = output_decoder
        data_file["output_encoder"] = output_encoder

        timestr = time.strftime("%Y-%m-%d")
        with open(settings["output_path"] + timestr + "_data_set.pkl", "wb") as f:
            pickle.dump(data_file, f)