import numpy as np
import hippylib as hp
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
import os, gc
import matplotlib.pyplot as plt

class FullTracer(object):
    def __init__(self, Vh, parameter_file=None, state_file=None):
        self.parameters = {}
        self.parameters["output_frequency"] = 5
        self.parameters["output_path"] = "./mcmc_output/"
        self.data = {}
        self.data["qoi"] = []
        self.data["accept"] = []
        self.data["cost"] = []
        self.i = 0
        self.Vh = Vh
        self.parameter_file = parameter_file
        self.state_file = state_file

        if not os.path.exists(self.parameters["output_path"]):
            os.makedirs(self.parameters["output_path"], exist_ok=True)

    def append(self,current, q, accept):
        self.data["qoi"].append(q)
        self.data["accept"].append(accept)
        self.data["cost"].append(current.cost)

        if self.parameter_file is not None:
            self.parameter_file.write(hp.vector2Function(current.m, self.Vh[PARAMETER], name="parameter"), self.i)

        if self.state_file is not None:
            self.state_file.write(hp.vector2Function(current.u, self.Vh[STATE], name="state"), self.i)
        self.i+=1

        if self.i % self.parameters["output_frequency"] == 0:
            output_utility(self.data, self.parameters["output_path"])
    
def output_utility(data, output_path):

    # Plotting the acceptance ratio moving average
    accept = np.stack(data["accept"])
    np.save(output_path + "accept.npy", accept)
    n=100
    if accept.ndim == 1:
        accept = np.expand_dims(accept, axis=1)
    accept = np.min(np.abs(accept), axis=1)
    if accept.size > n:
        moving_average = np.convolve(accept, np.ones(n) / n, mode='valid')
        plt.figure(figsize=(5,4))
        plt.plot(np.arange(accept.size-n+1) + n, moving_average)
        plt.ylabel("Acceptance ratio")
        plt.xlabel("MCMC chain")
        plt.grid(":")
        plt.savefig(output_path + "acceptance_100_moving_average.pdf", bbox_inches="tight")
        plt.close()
    del accept
    gc.collect()

    # Plotting the misfit evolution
    cost = np.stack(data["cost"])
    np.save(output_path + "cost.npy", cost)
    plt.figure(figsize=(5,4))
    plt.plot(np.arange(cost.size) + 1, cost)
    plt.ylabel("Data misfit")
    plt.xlabel("MCMC chain")
    plt.grid(":")
    plt.savefig(output_path + "cost.pdf", bbox_inches="tight")
    plt.close()
    # Plotting the misfit evolution for the second half of the chain
    if cost.size > 10:
        plt.figure(figsize=(5,4))
        plt.plot(np.arange(cost.size//2+1, cost.size+1), cost[cost.size//2:])
        plt.ylabel("Data misfit")
        plt.xlabel("MCMC chain")
        plt.grid(":")
        plt.savefig(output_path + "cost_half.pdf", bbox_inches="tight")
        plt.close()
    del cost
    gc.collect()

    # Plotting the qoi evolution
    qoi = np.stack(data["qoi"])
    np.save(output_path + "qoi.npy", qoi)
    if qoi.ndim == 1: 
        qoi = np.expand_dims(qoi, axis=1)
    for ii in range(qoi.shape[1]):
        plt.figure(figsize=(5,4))
        plt.plot(qoi[:, ii])
        plt.ylabel("QoI value")
        plt.xlabel("MCMC chain")
        plt.grid(":")
        plt.savefig(output_path + "qoi_%d.pdf"%ii, bbox_inches="tight")
        plt.close()
        # Plotting the misfit evolution for the second half of the chain
        if qoi.shape[0] > 10:
            plt.figure(figsize=(5,4))
            plt.plot(np.arange(qoi.shape[0]//2+1, qoi.shape[0]+1), qoi[qoi.shape[0]//2:, ii])
            plt.ylabel("QoI value")
            plt.xlabel("MCMC chain")
            plt.grid(":")
            plt.savefig(output_path + "qoi_half_%d.pdf"%ii, bbox_inches="tight")
            plt.close()
        del qoi
        gc.collect()