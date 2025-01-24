import numpy as np
from mpi4py import MPI

def set_global(comm, array, vector):
    """
    Set the local part of a dolfin vector from a numpy array.
    """
    if comm.size == 1:
        vector.set_local(array)
    else:
        start, end = vector.local_range()
        vector.set_local(array[start:end])
        vector.apply("")

def get_global(comm, vector):
    """
    Gather a dolfin vector as a global numpy array.
    """
    size = comm.size
    if size == 1:
        array = vector.get_local()
    else:
        start, end = vector.local_range()
        range = np.zeros((size, 2), dtype='i')
        comm.Allgather(np.array([[start, end]], dtype='i'), range)
        array = np.zeros(vector.size())
        comm.Allgatherv(sendbuf=vector.get_local(), recvbuf=[array, (range[:, 1]-range[:, 0], range[:, 0])])
    return array