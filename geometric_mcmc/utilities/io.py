import dolfin as dl
import hippylib as hp
import numpy as np
import warnings

def load_samples_from_XDMF(Vh: dl.FunctionSpace, file_name: str, n_samples: int, name: str ="") -> list[dl.Vector, ...]:
    """
    Load a samples from an XDMF file.
    :param Vh: The dolfin function space
    :param file_name: The XDMF file name
    :param n_samples: The number of samples to load
    :param name: The name to load
    :return: a list of :code:`dolfin.Vector`
    Note that over assigning :code:`n_samples` will raise a warning but will not raise an error.
    """
    function = dl.Function(Vh)
    samples = []
    file = dl.XDMFFile(Vh.mesh().mpi_comm(), file_name)
    file.parameters["functions_share_mesh"] = True
    file.parameters["rewrite_function_mesh"] = False
    for ii in range(n_samples):
        function.vector().zero()
        try:
            file.read_checkpoint(function, name, ii)
        except:
            warnings.warn("Could not read sample %d from file %s"%(ii,file_name))
            break
        samples.append(function.vector().copy())
    file.close()
    return samples

def save_samples_to_XDMF(Vh: dl.FunctionSpace, file_name: str, samples: list[dl.Vector], name: str ="") -> None:
    """
    Save samples to an XDMF file.
    :param Vh: :code:`dolfin.FunctionSpace`
    :param file_name: The XDMF file name. Must end with '.xdmf'
    :param samples: The samples consists of :code:`dolfin.Vector`
    :param name: The name to save
    """
    samples = []
    file = dl.XDMFFile(Vh.mesh().mpi_comm(), file_name)
    for ii, s in enumerate(samples):
        append = False if ii == 0 else True
        file.write_checkpoint(hp.vector2Function(s, Vh), name, ii, append=append)
    file.close()

def save_mv_to_XDMF(Vh: dl.FunctionSpace, file_name: str, mv: hp.MultiVector, name: str = "", normalize=False) -> None:
    """
    Save a hp.Multivector to an XDMF file.
    :param file_name: The XDMF file name
    :param mv: :code:`hp.Multivector`
    :param name: the name to save
    """
    file = dl.XDMFFile(Vh.mesh().mpi_comm(), file_name)
    for ii in range(mv.nvec()):
        append = False if ii == 0 else True
        if normalize: mv[ii] *= 1./mv[ii].norm("linf")
        file.write_checkpoint(hp.vector2Function(mv[ii], Vh), name, ii, append=append)
    file.close()

def load_mv_from_XDMF(Vh: dl.FunctionSpace, file_name: str, mv: hp.MultiVector, name: str ="") -> None:
    """
    Save a dolfin function to an XDMF file.
    :param file_name: The XDMF file name
    :param mv: The hp.Multivector
    :param name: The name of to load
    """
    file = dl.XDMFFile(Vh.mesh().mpi_comm(), file_name)
    function = dl.Function(Vh)
    file.parameters["functions_share_mesh"] = True
    file.parameters["rewrite_function_mesh"] = False
    for ii in range(mv.nvec()):
        mv[ii].zero()
        function.vector().zero()
        file.read_checkpoint(function, name, ii)
        mv[ii].axpy(1., function.vector())
    file.close()