
import numpy as np
import dolfin as dl
import hippylib as hp
from scipy.linalg import eigh

def decomposeGaussNewtonHessian(J: hp.MultiVector, Rinv_J: hp.MultiVector, 
                     decoder: hp.MultiVector, encoder: hp.MultiVector, 
                     prior: hp.prior, noise_precision: np.ndarray, oversampling=20) -> np.ndarray:
    """
    Compute the Gauss--Newton Hessian eigendecomposition given the Jacobian and prior-preconditioned Jacobian. 
    This method use the decomposition the inside-out Gauss-Newton Hessian to compute the decomposition of the Gauss-Newton Hessian.
    :param J: The Jacobian.
    :param Rinv_J: The prior preconditioned Jacobian.
    :param decoder: The decoder multivector to store the output.
    :param encoder: The encoder multivector to store the output.
    :param prior: The prior with R and Rsolver.
    :param noise_precision: The noise precision matrix.
    :return: The eigenvalues.
    """
    hessian_rank = min(J.nvec(), J[0].size()) # Get the rank of the Hessian decomposition
    assert encoder.nvec() == hessian_rank, "The encoder must have the same number of vectors as the rank of the Hessian decomposition."
    assert decoder.nvec() == hessian_rank, "The decoder must have the same number of vectors as the rank of the Hessian decomposition."
    JCJt = J.dot_mv(Rinv_J) # Compute the J @ J adojint, a square matrix of size of the data
    sv_squared, lsv = eigh(JCJt) # Compute the eigenvalues and eigenvectors of the J @ J adojint
    zero_entries = np.where(sv_squared < 1e-12)[0] # Get the indices of the zero eigenvalues
    if zero_entries.size == sv_squared.size - hessian_rank: # If the number of zero eigenvalues is equal to the number of data minus the rank of the Hessian decomposition
        sv_squared = sv_squared[::-1] # Reverse the eigenvalues
        lsv = lsv[:, ::-1] # Reverse the eigenvectors
        sv = np.sqrt(sv_squared[:hessian_rank]) # Compute the square root of the eigenvalues with truncation
        lsv = lsv[:, :hessian_rank] # Truncate the eigenvectors
        eigenvalues, rotation = eigh(np.diag(sv)@lsv.T@noise_precision@lsv@np.diag(sv)) # Compute the eigenvalues and eigenvectors of the Hessian
        weighting_matrix = rotation.T@np.diag(1./sv) @ lsv.T # Compute the weighting matrix
        reorder_indices = eigenvalues.argsort()[::-1] # Get the indices of the eigenvalues in descending order
        eigenvalues = eigenvalues[reorder_indices] # Reorder the eigenvalues
        for ii, idx in enumerate(reorder_indices): # Loop over the eigenvectors
            encoder[ii].zero() # Zero the encoder multivector
            J.reduce(encoder[ii], weighting_matrix[idx]) # Compute the encoder multivector
        Rinv_operator = hp.Solver2Operator(prior.Rsolver) # Create the prior preconditioned operator
        hp.MatMvMult(Rinv_operator, encoder, decoder) # Compute the decoder multivector
    else:
        operator = JTJOperator(J, noise_precision)
        Omega = hp.MultiVector(J[0], hessian_rank + oversampling)
        hp.parRandom.normal(1., Omega)
        eigenvalues, temp_mv = hp.doublePassG(operator, prior.R, prior.Rsolver, Omega, hessian_rank)
        decoder.swap(temp_mv)
        encoder = hp.MultiVector(J[0], hessian_rank)
        hp.MatMvMult(prior.R, decoder, encoder)
    return eigenvalues[:hessian_rank]

class JTJOperator():
    """
    A class for the Jacobian transpoe @ noise_precision @ Jacobian operator
    """
    def __init__(self, J, noise_precision):
        self.J = J
        self.noise_precision = noise_precision
    def mult(self, x, y):
        """
        The matrix-vector multiplication.
        """
        y.zero()
        self.J.reduce(y, self.noise_precision@self.J.dot_v(x))