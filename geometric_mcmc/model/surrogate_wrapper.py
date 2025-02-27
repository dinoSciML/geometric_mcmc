try:
    import torch
except:
    pass
import numpy as np
import warnings


class ReducedBasisSurrogate(object):

    def dim():
        """
        Return the dimension of the input and output vector.
        return: The input dimension (int) and output dimension (int).
        """
        raise NotImplementedError

    def eval(self, mr, derivative_order=0):
        """
        Evaluate the surrogate model and its derivative upto a given order.
        :param mr: The parameter numpy array.
        :param derivative_order: The derivative order.
        return: The surrogate model output (np.ndarray) and the derivatives of the surrogate model output (np.ndarray).
        For example...
        If :code:`mr` has shape (input_dim,) and derivative_order = 0, then the output is a single numpy array with shape (output_dim,) for surrogate evaluation.
        If :code:`mr` has shape (input_dim,) and derivative_order = 1, then the output is a tuple of two numpy arrays with shape (output_dim,) and (output_dim, input_dim) for surrogate evaluation and Jacobian.
        etc.
        """
        raise NotImplementedError("")
    
    def cost(self, ur, mr):
        """
        Compute the cost function value given the observable.
        :param ur: The observable numpy array.
        :param mr: The parameter numpy array.
        return: misfit + regularization, regularization, misfit value
        """
        raise NotImplementedError("")
    
    def misfit_vector(self, ur, weighted=False):
        """
        Compute the misfit vector.
        :param ur: The observable numpy array.
        :param weighted: Whether to apply the noise precision matrix.
        """
        raise NotImplementedError("")

def check_scaled_identity(mat):
    """
    Check if the matrix is the identity matrix.
    :param A: The matrix to check.
    """
    mat_test = np.eye(mat.shape[0])*mat[0, 0]
    return np.allclose(mat, mat_test)

try:
    class PyTorchSurrogateModel(ReducedBasisSurrogate):
        def __init__(self, neural_network: torch.nn.Module, input_dim: int = None, output_dim: int = None, 
                     data: np.ndarray = None, noise_precision: np.ndarray=None, output_decoder: np.ndarray = None, device: str = "cpu", dtype=torch.float32) -> None:
            """
            Constructor:
            :param neural_network: The PyTorch neural network model.
            :param input_dim: The input dimension of the model.
            :param output_dim: The output dimension of the model.
            :param data: The data (same length as neural network output) to compute the cost function.
            :param noise_precision: The noise precision matrix (same number of colimns and rows as neural network output).
            :param output_decoder: The output decoder matrix.
            :param device: The device to run the model.
            :param dtype: The data type.
            """
            self.neural_network = neural_network
            self.device = device
            self.dtype = dtype
            self.neural_network.to(self.device)
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.data = data
            self.noise_precision = noise_precision
            self.output_decoder = output_decoder
        
        def dim(self):
            return self.input_dim, self.output_dim
        
        def check_input(self):
            """
            This method checks if the data, noise precision, and output decoder are specified correctly.
            """
            assert self.data is not None, "Data must be specified"
            if self.data.size != self.output_dim:
                self.reduced_misfit = False
                if self.noise_precision is not None or self.decoder is not None:
                    raise Exception("The data is does not have the same size as surrogate output." + 
                                "The noise precision and output decoder must be specified when the data is not reduced.")
            elif self.noise_precision is None:
                self.reduced_misfit = True
                self.noise_precision = np.eye(self.output_dim)
                warnings.warn("The noise precision is not specified. Using the identity matrix.")
            elif not check_scaled_identity(self.noise_precision):
                raise Exception("The noise precision is not a scaled identity matrix. The misfit functional cannot be fully reduced. Please provide the full data and the output decoder")
            else:
                self.reduced_misfit = True

        def eval(self, mr, derivative_order= 0):

            if mr.ndim == 1: # If the input is a 1D array (single sample) then expand the dimension to 2D
                mr = np.expand_dims(mr, axis=0)
                expanded = True
            else: expanded = False

            mr = torch.from_numpy(mr).to(dtype=self.dtype, device=self.device) # Convert the numpy array to torch tensor

            if derivative_order > 0: # If we need to compute the derivatives
                mr = mr.requires_grad_(True) # Set the requires_grad flag to True
                output = [self.neural_network(mr)] # Compute the function evaluation
                if derivative_order >= 1: # If we need to compute the Jacobian
                    output.append(torch.vmap(torch.func.jacrev(self.neural_network))(mr)) # Compute the Jacobian
                if derivative_order >= 2: # If we need to compute the Hessian
                    output.append(torch.vmap(torch.func.hessian(self.neural_network))(mr))  # Compute the Hessian
                if derivative_order >= 3: # If we need to compute the third order derivative
                    output.append(torch.vmap(torch.func.jacrev(torch.func.hessian(self.neural_network)))(mr)) # Compute the third order derivative
                if derivative_order >= 4: 
                    raise ValueError("Higher order derivatives are not supported")
                
                if expanded: output = [y.squeeze() for y in output] # If the input is a 1D array (single sample) then squeeze the dimension and convert to numpy array

                return tuple([y.cpu().detach().numpy() for y in output]) # Return the output as tuple
            else:
                with torch.no_grad(): # If we do not need to compute the derivatives then use the no_grad context
                    ur = self.neural_network(mr) # Compute the function evaluation

                if expanded: ur = ur.squeeze(0) # If the input is a 1D array (single sample) then squeeze the dimension
            
                return ur.cpu().detach().numpy()
        
        def cost(self, ur, mr):
            self.check_input()
            temp = self.misfit_vector(ur)
            misfit = 0.5*np.inner(temp, self.noise_precision.dot(temp))
            regularization = 0.5*np.inner(mr, mr)
            return [misfit + regularization, regularization, misfit]
        
        def misfit_vector(self, ur, weighted=False):
            self.check_input()
            out = ur - self.data if self.reduced_misfit else self.decoder@ur - self.data
            if weighted:
                return self.noise_precision.dot(out)
            else:
                return out
except:
    pass
