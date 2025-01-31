import torch
import numpy as np

class FFN(torch.nn.Module):
    def __init__(self, width = [100, 100, 100], activation = torch.nn.GELU()):
        super(FFN, self).__init__()
        layers = []
        for ii in range(len(width) - 1):
            layers.append(torch.nn.Linear(width[ii], width[ii+1]))
            if ii < len(width) - 2:
                layers.append(activation)
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DIFFN(torch.nn.Module):
    def __init__(self, FFN, max_order=1):
        """
        Initialize the FFNWithDerivatives module.

        :param ffn: Feedforward network (instance of nn.Module)
        :param max_order: The maximum order of derivatives to compute (default is 1)
        """
        super(DIFFN, self).__init__()
        self.FFN = FFN  # The given feedforward network (FFN)
        self.max_order = max_order  # Maximum derivative order

    def forward(self, x):
        """
        Forward pass that computes the function evaluation and its derivatives up to the given order.

        :param x: Input tensor of shape (batch_size, input_dim)
        :return: A tuple (output, derivatives)
                 - output: The FFN evaluation at x
                 - derivatives: A list of tensors containing the derivatives (Jacobian, Hessian, etc.)
        """
        # Initial function evaluation (order 0 derivative)

        x = x.requires_grad_(True)
        output = [self.FFN(x)]
        if self.max_order >= 1:
            output.append(torch.vmap(torch.func.jacrev(self.FFN))(x))
        if self.max_order >= 2:
            output.append(torch.vmap(torch.func.hessian(self.FFN))(x))
        if self.max_order >= 3:
            output.append(torch.vmap(torch.func.jacrev(torch.func.hessian(self.FFN)))(x))
        return tuple(output)

def training(model, optimizer, loss_fn, data_loader, epochs, scheduler = None, device="cpu", verbose = False):
    train_losses, valid_losses = [], []
    order = model.max_order
    model.to(device)
    for epoch in range(epochs):
        running_loss = []
        for data in data_loader[0]:
            optimizer.zero_grad()
            x, *data_list = data
            x = x.to(device)
            data_list = [data.to(device) for data in data_list]
            *y_pred, = model(x)
            loss = 0.0
            for ii in range(order+1):
                loss += loss_fn(y_pred[ii], data_list[ii])/(order+1)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.detach().item())
        train_losses.append(np.mean(np.array(running_loss)))
        with torch.no_grad():
            for data in data_loader[1]:
                x, *data_list = data
                x = x.to(device)
                data_list = [data.to(device) for data in data_list]
                if len(data_list) == 1:
                    y_pred = model.FFN(x)
                    valid_loss = loss_fn(y_pred, data_list[0])
                else:
                    *y_pred, = model(x)
                    valid_loss = 0.0
                    for ii, y_ref in enumerate(data_list):
                        valid_loss += loss_fn(y_pred[ii], y_ref)/len(data_list)
            valid_losses.append(valid_loss.item())
        if verbose:
            print(f"Epoch: {epoch}, Training Loss: {train_losses[-1]}, Validation Loss: {valid_losses[-1]}")
        if not scheduler is None:
            scheduler.step()
    return train_losses, valid_losses
