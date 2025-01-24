import hippylib as hp
import numpy as np

def check_orthonormality(decoder: hp.MultiVector, encoder : hp.MultiVector) -> None:
    """
    Check the orthonormality of the decoder and encoder
    :param decoder: The decoder multivector
    :param encoder: The encoder multivector
    """
    error = np.linalg.norm(encoder.dot_mv(decoder) - np.eye(decoder.nvec))
    if error > 1.0e-5:
        raise Exception("The decoder and encoder are not orthonormal (error = %f)" % error)