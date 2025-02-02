import os
import time
import numpy as np
from ..utilities.io import save_mv_to_XDMF
from ..mcmc.map_utilities import compute_Hessian_decomposition_at_sample, compute_MAP
from ..model.dimension_reduction import compute_DIS, compute_KLE
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT

def dimension_reduction_comparison(model, input_rank, output_rank, output_path, n_samples):

    if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
    
    mpi_comm = model.prior.R.mpi_comm()
    Vh = model.problem.Vh

    assert input_rank <= Vh[PARAMETER].dim()
    assert output_rank <= model.misfit.observable.dim()

    if mpi_comm.rank == 0: print("Computing the Hessian decomposition at the MAP point")
    t0 = time.time()
    x_MAP = compute_MAP(model)
    map_eigenvalues, map_decoder, map_encoder = compute_Hessian_decomposition_at_sample(model, x_MAP, gauss_newton_approx = False, rank=input_rank)
    if mpi_comm.rank == 0: print("Time for computing the Hessian decomposition at the MAP point: ", time.time()-t0)
    save_mv_to_XDMF(Vh[PARAMETER], output_path + "map_decoder.xdmf", map_decoder)
    save_mv_to_XDMF(Vh[PARAMETER], output_path + "map_encoder.xdmf", map_encoder)
    np.save(output_path + "map_eigenvalues", map_eigenvalues)
    if mpi_comm.rank == 0:
        plt.figure(figsize=(5,4))
        plt.semilogy(map_eigenvalues)
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.grid(":")
        plt.savefig(output_path + "map_hessian_eigenvalues.pdf", bbox_inches="tight")
        plt.close()


    if mpi_comm.rank == 0: print("Computing the KLE expansion of the prior")
    t0 = time.time()
    kle_eigenvalues, kle_decoder, kle_encoder = compute_KLE(model.prior, rank=input_rank)
    if mpi_comm.rank == 0: print("Time for computing the KLE expansion of the prior: ", time.time()-t0)
    save_mv_to_XDMF(Vh[PARAMETER], output_path + "kle_encoder.xdmf", kle_encoder)
    save_mv_to_XDMF(Vh[PARAMETER], output_path + "kle_decoder.xdmf", kle_decoder)
    np.save(output_path + "kle_eigenvalues", kle_eigenvalues)
    if mpi_comm.rank == 0:
        plt.figure(figsize=(5,4))
        plt.semilogy(kle_eigenvalues)
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.grid(":")
        plt.savefig(output_path + "kle_eigenvalues.pdf", bbox_inches="tight")
        plt.close()


    if mpi_comm.rank == 0: print("Computing the derivative-informed subspace")
    input_res, output_res, observables = compute_DIS(model, n_samples, input_rank, output_rank, oversampling=20, mode="reverse", return_observables=True)
    if mpi_comm.rank == 0: print("Time for computing the derivative-informed subspace: ", time.time()-t0)
    dis_input_eigenvalues, dis_input_decoder, dis_input_encoder = input_res
    dis_output_eigenvalues, dis_output_decoder, dis_output_encoder = output_res
    save_mv_to_XDMF(Vh[PARAMETER], output_path + "dis_input_encoder.xdmf", dis_input_encoder)
    save_mv_to_XDMF(Vh[PARAMETER], output_path + "dis_input_decoder.xdmf", dis_input_decoder)
    np.save(output_path + "dis_input_eigenvalues", dis_input_eigenvalues)
    np.save(output_path + "dis_output_encoder", dis_output_encoder)
    np.save(output_path + "dis_input_decoder", dis_output_decoder)
    np.save(output_path + "dis_output_eigenvalues", dis_output_eigenvalues)
    if mpi_comm.rank == 0:
        plt.figure(figsize=(5,4))
        plt.semilogy(dis_input_eigenvalues)
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.grid(":")
        plt.savefig(output_path + "dis_input_eigenvalues.pdf", bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(5,4))
        plt.semilogy(dis_output_eigenvalues)
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.grid(":")
        plt.savefig(output_path + "dis_output_eigenvalues.pdf", bbox_inches="tight")
        plt.close()
    
    pod_matrix = np.einsum("ij, ik -> jk", observables, observables)
    pod_output_eigenvalues, pod_output_decoder = eigh(pod_matrix, b=model.misfit.noise_precision)
    pod_output_eigenvalues = pod_output_eigenvalues[::-1]
    pod_output_decoder = pod_output_decoder[:, ::-1]
    pod_output_encoder = pod_output_decoder.T@model.misfit.noise_precision

    np.save(output_path + "pod_output_encoder", pod_output_encoder)
    np.save(output_path + "pod_ioutput_decoder", pod_output_decoder)
    np.save(output_path + "pod_output_eigenvalues", pod_output_eigenvalues)
    if mpi_comm.rank == 0:
        plt.figure(figsize=(5,4))
        plt.semilogy(pod_output_eigenvalues)
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.grid(":")
        plt.savefig(output_path + "pod_output_eigenvalues.pdf", bbox_inches="tight")
        plt.close()



    