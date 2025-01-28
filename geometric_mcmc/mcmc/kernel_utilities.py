
import numpy as np
import hippylib as hp
from scipy.linalg import eigh

def decomposeGaussNewtonHessian(J: hp.MultiVector, Rinv_J: hp.MultiVector, 
                     decoder: hp.MultiVector, encoder: hp.MultiVector, 
                     prior_precision: hp.Solver2Operator, noise_precision: np.ndarray) -> np.ndarray:
    """
    Compute the Gauss--Newton Hessian eigendecomposition given the Jacobian and prior-preconditioned Jacobian. 
    This method use the decomposition the inside-out Gauss-Newton Hessian to compute the decomposition of the Gauss-Newton Hessian.
    :param J: The Jacobian.
    :param Rinv_J: The prior preconditioned Jacobian.
    :param decoder: The decoder multivector to store the output.
    :param encoder: The encoder multivector to store the output.
    :param prior_precision: The prior precision operator (i.e., the Rinv operator in the hippylib convention),
    :param noise_precision: The noise precision matrix.
    :return: The eigenvalues.
    """
    hessin_rank = min(J.nvec(), J[0].size()) # Get the rank of the Hessian decomposition
    JCJt = J.dot_mv(Rinv_J) # Compute the J @ J adojint, a square matrix of size of the data
    sv_squared, lsv = eigh(JCJt) # Compute the eigenvalues and eigenvectors of the J @ J adojint
    sv = np.sqrt(sv_squared) # Compute the square root of the eigenvalues
    eigenvalues, rotation = eigh(np.diag(sv)@lsv.T@noise_precision@lsv@np.diag(sv)) # Compute the eigenvalues and eigenvectors of the Hessian
    weighting_matrix = rotation.T@np.diag(1./sv) @ lsv.T # Compute the weighting matrix
    reorder_indices = eigenvalues.argsort()[::-1] # Get the indices of the eigenvalues in descending order
    eigenvalues = eigenvalues[reorder_indices] # Reorder the eigenvalues
    for ii, idx in enumerate(reorder_indices[:hessin_rank]): # Loop over the eigenvectors
        encoder[ii].zero() # Zero the encoder multivector
        J.reduce(encoder[ii], weighting_matrix[idx]) # Compute the encoder multivector
    hp.MatMvMult(prior_precision, encoder, decoder) # Compute the decoder multivector
    return eigenvalues