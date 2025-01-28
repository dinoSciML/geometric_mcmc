import numpy as np
import dolfin as dl
import os, gc
import matplotlib.pyplot as plt
import h5py
from ..utilities.collective import get_global, get_vertex_order
from .sample_structure import SampleStruct

class Tracer(object):
    """
    This is an abstract class for tracing and saving the output of MCMC.
    """
    def append(self, current, q, accept):
        """
        Append the data to the tracer.
        :param current: The current sample data. Must be an instance of SampleDataSturct
        :param q: The QoI value
        :param accept: The acceptance value for Metropolis--Hastings
        """
        raise NotImplementedError("append method is not implemented.")
    
    def save(self):
        """
        Save the data to the file.
        """
        raise NotImplementedError("save method is not implemented.")

class Qoi(object):
    """
    This is a simplified Qoi object that only require a eval method. Used for tracing MCMC results.
    """
    def eval(self, x):
        """
        Evaluate the QoI
        :param x: The input data consists of a list of state and parameter
        Note: the input might consists of :code:`dolfin.Vector` for :code:`SampleStruct` or :code:`numpy.ndarray` for :code:`ReducedSampleStruct`
        """
        raise NotImplementedError("eval method is not implemented.")

class ObservableQoi(Qoi):
    def __init__(self, observable, index = None):
        self.observable = observable
        self.index = index
    def eval(self, x):
        if self.index is None:
            return self.observable.eval(x)
        else:
            return self.observable.eval(x)[self.index]

class NullTracer(Tracer):
    """
    This class is used to trace no outputs from the MCMC chain.
    """
    def __init__(self):
        pass
    def append(self, current, q, accept):
        pass
    def save(self):
        pass


class FullTracer(Tracer):
    """
    This class is used to trace all outputs from the MCMC chain.
    """
    def __init__(self, Vm, output_path: str, qoi_reference: np.ndarray=None) -> None:
        """
        Constructor for the MCMC tracer.
        :param Vm: The parameter function space
        :param output_path: The output path for the tracer. This must be set before the first append call.
        :param qoi_reference: The reference value for the QoI
        """
        self.parameters = {} # The parameters list for the tracer
        self.parameters["save_frequency"] = 100 # The frequency to save chain data to file
        self.parameters["visual_frequency"] = 100 # The frequency to visualize the data
        self.parameters["moving_average_window_size"] = 100 # The window size for the moving average of the acceptance ratio
        self.output_path = output_path
        self.data = {} # The data dictionary for the tracer
        self.data["qoi"] = []
        self.data["accept"] = []
        self.data["cost"] = []
        self.data["parameter"] = []
        self.data["jump"] = []
        self.qoi_reference = qoi_reference 
        self.i = 0 # The counter for the number of samples
        self._param_size = Vm.dim()
        self.mpi_comm = Vm.mesh().mpi_comm()

        self._help = dl.Function(Vm).vector()
        self._M = dl.assemble(dl.TestFunction(Vm)*dl.TrialFunction(Vm)*dl.dx)

        if self.mpi_comm.rank == 0 and not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        # Get the vertex order for the parameter only if the parameter uses nodal finite element
        if self.mpi_comm.size > 1 and Vm.ufl_element().degree() == 1:
            self.to_vertex, _ = get_vertex_order(Vm, root=0) # if mesh parallel, get the indices for image order of the parameter entires.
        else:
            self.to_vertex = None

    def append(self,current: SampleStruct, q: np.ndarray, accept: np.ndarray) -> None:
        """
        Append the data to the tracer.
        :param current: The current sample data. Must be an instance of SampleSturct
        :param q: The QoI value
        :param accept: The acceptance value for Metropolis--Hastings
        """
        # Append the data to data dictionary
        self.data["qoi"].append(q)
        self.data["accept"].append(accept)
        self.data["cost"].append(current.cost)
        self.data["parameter"].append(get_global(self.mpi_comm, current.m, root=0))
        
        # For the MCMC step, we need to get the size of the traced values and also initiate the file
        if self.i == 0:
            self._qoi_size = q.size if isinstance(q, np.ndarray) else 1
            self._accept_size = accept.size if isinstance(accept, np.ndarray) else 1
            if self.mpi_comm.rank == 0: self.initiate_file()
            self._help.zero()
            self._help.axpy(1.0, current.m)
            self.data["jump"].append(0)
        else:
            self._help.axpy(-1.0, current.m)
            self.data["jump"].append(np.sqrt(self._help.inner(self._M*self._help)))
            self._help.zero()
            self._help.axpy(1.0, current.m)

        self.i+=1 # Increment the counter

        # Save the data if the save frequency is reached
        if self.parameters["save_frequency"] is not None and self.i % self.parameters["save_frequency"] == 0 and self.mpi_comm.rank == 0:
            self.save()
        # Visualize the data if the visual frequency is reached
        if self.parameters["visual_frequency"] is not None and self.i % self.parameters["visual_frequency"] == 0 and self.mpi_comm.rank == 0:
            visual_utility(self.data, self.output_path, self.parameters["moving_average_window_size"], self.qoi_reference)

    def save(self):
        """
        Save the data to the file.
        """
        name_list = ['parameter', 'cost', 'qoi', 'accept', 'jump']
        data_size_list = [self._param_size, 1, self._qoi_size, self._accept_size, 1]

        if len(self.data["parameter"]) > 0: iterative_save_to_hdf5(self.output_path, name_list, data_size_list, self.data)
        self.data["parameter"] = [] # clean the parameter data


    def initiate_file(self):
        """
        Initiate the file for the tracer.
        """
        name_list = ['parameter', 'cost', 'qoi', 'accept', 'jump']
        data_size_list = [self._param_size, 1, self._qoi_size, self._accept_size, 1]
        dtype_list = [np.float64, np.float64, np.float64, np.int64, np.float64]
        attribute = {}
        attribute["n_samples"] = 0
        if self.to_vertex is not None: attribute["to_vertex"] = self.to_vertex
        initiate_hdf5(self.output_path, name_list, data_size_list, dtype_list, attribute)

class ReducedTracer(Tracer):
    def __init__(self, comm, output_path, qoi_reference=None):
        self.parameters = {}
        self.parameters["visual_frequency"] = 5
        self.parameters["save_frequency"] = 10
        self.parameters["moving_average_window_size"] = 100
        self.qoi_reference = qoi_reference
        self.output_path = output_path
        self.comm = comm
        self.data = {}
        self.data["qoi"] = []
        self.data["accept"] = []
        self.data["cost"] = []
        self.data["parameter"] = []
        self.data["jump"] = []
        self.i = 0

    def append(self, current, q, accept):
        self.data["qoi"].append(q)
        self.data["accept"].append(accept)
        self.data["cost"].append(current.cost)
        self.data["parameter"].append(current.m)
        self._help = None


        if self.i == 0:
            self._param_size = current.m.size
            self._qoi_size = q.size if isinstance(q, np.ndarray) else 1
            self._accept_size = accept.size if isinstance(accept, np.ndarray) else 1
            if self.mpi_comm.rank == 0: self.initiate_file()
        
        self.i+=1

        if self.parameter["visual_frequency"] is not None and self.i % self.parameters["visual_frequency"] == 0 and self.comm.rank == 0:
            visual_utility(self.data, self.output_path, self.parameters["moving_average_window_size"], self.qoi_reference)
        if self.parameters["save_frequency"] is not None and self.i % self.parameters["save_frequency"] == 0 and self.comm.rank == 0:
            self.save()

    def append(self,current: SampleStruct, q: np.ndarray, accept: np.ndarray) -> None:
        """
        Append the data to the tracer.
        :param current: The current sample data. Must be an instance of SampleSturct
        :param q: The QoI value
        :param accept: The acceptance value for Metropolis--Hastings
        """
        # Append the data to data dictionary
        self.data["qoi"].append(q)
        self.data["accept"].append(accept)
        self.data["cost"].append(current.cost)
        self.data["parameter"].append(current.m)
        self.data["jump"].append(np.sqrt(current.m.inner(current.m)))
        
        # For the MCMC step, we need to get the size of the traced values and also initiate the file
        if self.i == 0:
            self._qoi_size = q.size if isinstance(q, np.ndarray) else 1
            self._accept_size = accept.size if isinstance(accept, np.ndarray) else 1
            if self.mpi_comm.rank == 0: self.initiate_file()
            self._help = current.m
            self.data["jump"].append(0)
        else:
            self.data["jump"].append(np.linalg.norm(current.m - self._help))
            self._help = current.m

        self.i+=1 # Increment the counter

        # Save the data if the save frequency is reached
        if self.parameters["save_frequency"] is not None and self.i % self.parameters["save_frequency"] == 0 and self.mpi_comm.rank == 0:
            self.save()
        # Visualize the data if the visual frequency is reached
        if self.parameters["visual_frequency"] is not None and self.i % self.parameters["visual_frequency"] == 0 and self.mpi_comm.rank == 0:
            visual_utility(self.data, self.output_path, self.parameters["moving_average_window_size"], self.qoi_reference)

    def save(self):
        """
        Save the data to the file.
        """
        name_list = ['parameter', 'cost', 'qoi', 'accept', 'jump']
        data_size_list = [self._param_size, 1, self._qoi_size, self._accept_size, 1]
        if len(self.data["parameter"]) > 0: iterative_save_to_hdf5(self.output_path, name_list, data_size_list, self.data)
        self.data["parameter"] = [] # clean the parameter data
    
    def initiate_file(self):
        """
        Initiate the file for the tracer.
        """
        name_list = ['parameter', 'cost', 'qoi', 'accept', 'jump']
        data_size_list = [self._param_size, 1, self._qoi_size, self._accept_size, 1]
        dtype_list = [np.float64, np.float64, np.float64, np.int64, np.float64]
        attribute = {}
        attribute["n_samples"] = 0
        if self.to_vertex is not None: attribute["to_vertex"] = self.to_vertex
        initiate_hdf5(self.output_path, name_list, data_size_list, dtype_list, attribute)


def initiate_hdf5(output_path: str, name_list: list, data_size_list: list, dtype_list: list, attribute: dict = {}) -> None:
    """
    Initiate the file for the tracer.
    """
    with h5py.File(output_path + "mcmc_data.hdf5", "w") as f:
        for name, data_size, dtype in zip(name_list, data_size_list, dtype_list):
            f.create_dataset(name, (1, data_size), maxshape=(None, data_size), dtype=dtype)
        for key, value in attribute.items():
            f.attrs[key] = value

def iterative_save_to_hdf5(output_path: str, name_list: list, data_size_list: list, data: dict, attribute: dict = {}) -> None:
    """
    Save the data to the file. Note that we determine the saving batch size by the size of the parameter data, which is cleared after previous saving.
    """
    with h5py.File(output_path + "mcmc_data.hdf5", "a") as f:
        dset_list = [f[name] for name in name_list]
        start = f.attrs['n_samples']
        batch_size = len(data["parameter"])
        if start + batch_size > dset_list[0].shape[0]:
            for dset, data_size in zip(dset_list, data_size_list):
                dset.resize((start+batch_size, data_size))
        dset_list[0][start:(start+batch_size), :] = np.vstack(data["parameter"])
        for ii, name in enumerate(name_list[1:]):
            dset_list[ii+1][start:(start+batch_size), :] = np.vstack(data[name][start:])
        end = start + batch_size
        f.attrs['n_samples']=end
        for key, value in attribute.items():
            f.attrs[key] = value


def visual_utility(data, output_path, window_size, qoi_reference=None):

    # Plotting the acceptance ratio moving average
    accept = np.stack(data["accept"])
    if accept.ndim == 1: accept = np.expand_dims(accept, axis=1) 
    total_accept = np.min(np.abs(accept), axis=1)
    reject = np.cumsum(1-np.abs(accept), axis=0)
    total_reject = np.cumsum(np.abs(1- total_accept))
    if accept.shape[0] > window_size + 10:
        moving_average = np.convolve(total_accept, np.ones(window_size) / window_size, mode='valid')
        plt.figure(figsize=(5,4))
        plt.plot(np.arange(total_accept.size-window_size+1) + window_size, moving_average*100)
        plt.ylabel("Acceptance ratio (\%)")
        plt.xlabel("MCMC chain")
        plt.grid(":")
        plt.savefig(output_path + "acceptance_%d_moving_average.pdf"%(window_size), bbox_inches="tight")
        plt.close()
    start = find_first_one_in_array(total_reject)
    if accept.shape[1] > 1 and start >= 0:
        plt.figure(figsize=(5,4))
        for ii in range(accept.shape[1]):
            plt.plot(np.arange(start, accept.shape[0])+1, reject[start:, ii]/total_reject[start:]*100, label="Level %d"%(ii+1))
        plt.ylabel("Rejection contribution (\%)")
        plt.xlabel("MCMC chain")
        plt.grid(":")
        plt.legend(loc="best", frameon=False)
        plt.savefig(output_path + "reject_%d.pdf"%ii, bbox_inches="tight")
        plt.close()
    del accept, total_accept, reject, total_reject
    gc.collect()

    # Plotting the misfit evolution
    cost = np.stack(data["cost"])
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
    if qoi.ndim == 1: qoi = np.expand_dims(qoi, axis=1) 
        
    for ii in range(qoi.shape[1]):
        plt.figure(figsize=(5,4))
        plt.plot(qoi[:, ii])
        if qoi_reference is not None: plt.axhline(y=qoi_reference[ii], color='r', linestyle='--') 
        plt.ylabel("QoI value")
        plt.xlabel("MCMC chain")
        plt.grid(":")
        plt.savefig(output_path + "qoi_%d.pdf"%ii, bbox_inches="tight")
        plt.close()
        # Plotting the misfit evolution for the second half of the chain
        if qoi.shape[0] > 10:
            plt.figure(figsize=(5,4))
            plt.plot(np.arange(qoi.shape[0]//2+1, qoi.shape[0]+1), qoi[qoi.shape[0]//2:, ii])
            if qoi_reference is not None: plt.axhline(y=qoi_reference[ii], color='r', linestyle='--')
            plt.ylabel("QoI value")
            plt.xlabel("MCMC chain")
            plt.grid(":")
            plt.savefig(output_path + "qoi_half_%d.pdf"%ii, bbox_inches="tight")
            plt.close()
    del qoi
    gc.collect()


def find_first_one_in_array(array: np.ndarray) -> int:
    """
    Find the first one in an non-decreasing and non-negative array
    """
    # Use searchsorted to find the insertion point for 1
    index = np.searchsorted(array, 1, side='left')
    
    # Check if the value at the found index is actually 1
    if index < len(array) and array[index] == 1:
        return index
    return -1