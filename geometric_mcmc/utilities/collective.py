import numpy as np
import dolfin as dl
import hippylib as hp
from mpi4py import MPI

def set_global(comm: MPI.Intracomm, array: np.array, vector: dl.Vector) -> None:
    """
    Set the local part of a dolfin vector from a numpy array.
    :param comm: The MPI communicator
    :param array: The numpy array
    :param vector: The dolfin vector
    """
    if comm.size == 1:
        vector.set_local(array)
    else:
        start, end = vector.local_range()
        vector.set_local(array[start:end])
        vector.apply("")

def get_global(comm: MPI.Intracomm, vector: dl.Vector, root: int=None) -> np.ndarray:
    """
    Gather a dolfin vector as a global numpy array.
    :param comm: The MPI communicator
    :param vector: The dolfin vector
    :return: The numpy array
    """
    size = comm.size

    if size == 1:
        array = vector.get_local()
    elif root is None:
        start, end = vector.local_range()
        sizes = np.array(comm.allgather(end-start))
        displacements = np.array([sum(sizes[:i]) for i in range(size)])
        array = np.empty(vector.size())
        comm.Allgatherv(sendbuf=vector.get_local(), recvbuf=[array, (sizes, displacements)])
    else:
        array = comm.gather(vector.get_local(), root=root)
        if comm.rank == root:
            array = np.concatenate(array)
    return array

def set_global_mv(comm: MPI.Intracomm, array: np.ndarray, mv: hp.MultiVector) -> None:
    """
    Globally set a hp.Multivector from a numpy array.
    :param comm: The MPI communicator
    :param array: The numpy array
    :param mv: The hp.Multivector
    """
    if array.shape[0] != mv.nvec():
        array = array.T
    assert array.shape[0] == mv.nvec() and array.shape[1] == mv[0].size()
    for ii in range(mv.nvec()):
        set_global(comm, array[ii], mv[ii])

def get_global_mv(comm: MPI.Intracomm, mv: hp.MultiVector, root: int=None) -> np.ndarray:
    """
    Globally get a hp.Multivector as a numpy array.
    :param comm: The MPI communicator
    :param mv: The hp.Multivector
    :return: The numpy array
    """
    out = []
    for ii in range(mv.nvec()):
        out.append(get_global(comm, mv[ii], root=root))
    return np.stack(out)

def split_mpi_comm(comm: MPI.Intracomm, size_1: int, size_2: int) -> tuple[MPI.Intracomm, MPI.Intracomm]:
    """
    Split the MPI communicator :coder:`comm` into multiple MPI communicator according to the number of rank in the list :code:`partition`.
    :param comm: The MPI communicator
    :param size_1: The size of the first communicator
    :param size_2: The size of the second communicator
    return tuples of MPI communicators, with the first communicator having size :code:`size_1` and the second communicator having size :code:`size_2`
    """
    if not comm.size == size_1 * size_2:
        raise Exception("The size of the communicator is not divisible by the product of size_1 and size_2")
    key = comm.rank % size_1
    color = comm.rank // size_1
    comm_1 = comm.Split(color=color, key=key)
    comm_2 = comm.Split(color=key, key=color)
    return comm_1, comm_2

def get_vertex_order(Vh: dl.FunctionSpace, root: int=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the vertex order of a dolfin function space.
    :param Vh: The dolfin function space
    :return: The vertex order
    """
    comm = Vh.mesh().mpi_comm()
    assert Vh.num_sub_spaces() == 0, "Cannot handle mixed space."
    size = comm.size
    if size == 1:
        to_dof = dl.dof_to_vertex_map(Vh)
        to_vertex = dl.vertex_to_dof_map(Vh)
    else:
        to_vertex, to_dof = None, None
        start, end = Vh.dofmap().ownership_range()
        d2v = dl.dof_to_vertex_map(Vh)
        local_counts = end - start
        coord_dof = Vh.mesh().coordinates()[d2v[:local_counts]]
        if root is None:
            coord_all = comm.allgather(coord_dof)
        else:
            coord_all = comm.gather(coord_dof, root=root)
        if root is None or (comm.rank == root):    
            coord_all = np.concatenate(coord_all, axis=0)
            to_vertex = np.lexsort(tuple([coord_all[:, i] for i in range(coord_all.shape[1])]))
            to_dof = np.empty_like(to_vertex, dtype=int)
            to_dof[to_vertex] = np.arange(0, len(to_vertex), dtype=int)
    return to_vertex, to_dof