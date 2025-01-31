

import torch
import numpy as np
import sys, os, argparse
sys.path.insert(0, '../learning_utilities/')
from surrogate_training import FFN, DIFFN, training
import matplotlib.pyplot as plt
import matplotlib
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass

DTYPE = torch.float32

import pickle
import time
import seaborn as sns

def data_processing(data_file, partition, with_jac, output_mean_shift = None):
    dataset = []
    start = 0
    for n_samples, load_jac in zip(partition, with_jac):
        end = start + n_samples
        x = np.einsum("jk, ik->ij", data_file["input_encoder"], data_file["parameter"][start:end])
        y_shifted = data_file["observable"][start:end] - output_mean_shift[np.newaxis, :]
        y = np.einsum("jk, ik->ij", data_file["output_encoder"], y_shifted)
        x = torch.from_numpy(x).to(dtype=DTYPE)
        y = torch.from_numpy(y).to(dtype=DTYPE)
        if load_jac:
            y_jac = data_file["reduced_jacobian"][start:end]
            y_jac = torch.from_numpy(y_jac).to(dtype=DTYPE)
            dataset.append(torch.utils.data.TensorDataset(x, y, y_jac))
        else:
            dataset.append(torch.utils.data.TensorDataset(x, y))
        start = end

    return tuple(dataset)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a feedforward network with derivatives')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use (cpu or cuda)')
    parser.add_argument('--data_file', type=str, default="./data/data.pkl", help='the data file')
    parser.add_argument('--output_path', type=str, default="./training_result/", help='Output path to save the results')
    parser.add_argument('--h1_training', type=int, default=1, help='whether to use derivative-informed training')
    parser.add_argument('--total_epochs', type=int, default=250, help='Maximum number of epochs')
    parser.add_argument('--n_train', type=int, default=2048, help='Number of training samples')
    parser.add_argument('--n_validation', type=int, default=256, help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=2048, help='Number of testing samples')
    parser.add_argument('--width', type=int, default=800, help='Width of the feedforward network')
    parser.add_argument('--depth', type=int, default=3, help='Width of the hidden layers')
    parser.add_argument('--verbose', type=int, default=1, help='Print training information')
    args = parser.parse_args()

    tracer = vars(args)

    torch.manual_seed(0)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    with open(tracer["data_file"], "rb") as f:
        data_file = pickle.load(f)

    width = [data_file["n_input_bases"]]
    for ii in range(args.depth):
        width.append(args.width)
    width.append(data_file["n_output_bases"])
    model = FFN(width = width)
    model.to(args.device)
    modelwithjac = DIFFN(model, max_order = tracer["h1_training"])

    output_mean_shift = np.mean(data_file["observable"][:tracer["n_train"]], axis=0)

    train_dataset, validation_dataset, test_dataset = data_processing(data_file, 
                                                        [tracer["n_train"], tracer["n_validation"], tracer["n_test"]], 
                                                        [tracer["h1_training"], 0, 1], output_mean_shift=output_mean_shift)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = min(32, tracer["n_train"]))
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = min(256, tracer["n_validation"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = tracer["total_epochs"], eta_min = 1e-6)

    def mean_relative_error(y_pred, y):
        dim = tuple([i for i in np.arange(1, len(y.size()))])
        return torch.mean(torch.norm(y_pred - y, dim=dim)**2/torch.norm(y, dim=dim)**2)
    
    time_start = time.time()
    train_losses, valid_losses  = training(modelwithjac, optimizer, mean_relative_error, 
                                           [train_loader, validation_loader], tracer["total_epochs"], 
                                           scheduler=scheduler, device=tracer["device"], verbose=tracer["verbose"])
    time_end = time.time()
    torch.save(model.state_dict(), args.output_path + "surrogate.pt")

    plt.figure(figsize=(5,4))
    plt.plot(train_losses, label="Training")
    plt.plot(valid_losses, label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(tracer["output_path"] + "training_loss.pdf", bbox_inches="tight")
    plt.close()

    def relative_error(y_pred, y):
        dim = tuple([i for i in np.arange(1, len(y.size()))])
        return torch.norm(y_pred - y, dim=dim)/torch.norm(y, dim=dim)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = min(256, tracer["n_test"]))
    y_pred_list = [[], []]
    error_list = [[], []]
    with torch.no_grad():
        for data in test_loader:
            x, *data_list = data
            x = x.to(tracer["device"])
            data_list = [data.to(tracer["device"]) for data in data_list]
            *y_pred, = modelwithjac(x)
            for ii in range(2):
                error_list[ii].append(relative_error(y_pred[ii], data_list[ii]).detach().cpu().numpy())
                y_pred_list[ii].append(y_pred[ii].detach().cpu().numpy())
    
    error_list = [np.concatenate(error_list[ii]) for ii in range(2)]
    y_pred_list = [np.concatenate(y_pred_list[ii]) for ii in range(2)]

    data_frame = {"Observable": error_list[0], "Reduced" + "\n" + "Jacobian": error_list[1]}
    plt.figure(figsize=(4,4))
    sns.boxplot(data_frame, log_scale=True)
    plt.ylabel("Relative test error")
    plt.grid(":")
    plt.savefig(tracer["output_path"] + "relative_error_distribution.pdf", bbox_inches="tight")
    plt.close()
    
    tracer["training_time"] = time_end - time_start
    tracer["input_encoder"] = data_file["input_encoder"]
    tracer["output_encoder"] = data_file["output_encoder"]
    tracer["input_decoder"] = data_file["input_decoder"]
    tracer["output_decoder"] = data_file["output_decoder"]
    tracer["parameter_samples"] = data_file["parameter"][tracer["n_train"] + tracer["n_validation"]:tracer["n_train"] + tracer["n_validation"] + tracer["n_test"]]
    tracer["observable_samples"] = data_file["observable"][tracer["n_train"] + tracer["n_validation"]:tracer["n_train"] + tracer["n_validation"] + tracer["n_test"]]
    tracer["observable_mean_shift"] = output_mean_shift
    tracer["input_samples"] = test_dataset[:][0].detach().cpu().numpy()
    tracer["output_samples"] = test_dataset[:][1].detach().cpu().numpy()
    tracer["reduced_jacobian"] = test_dataset[:][2].detach().cpu().numpy()
    tracer["output_prediction"] = y_pred_list[0]
    tracer["jacobian_prediction"] = y_pred_list[1]
    tracer["input-output_error"] = error_list[0]
    tracer["reduced_jacobian_error"] = error_list[1]

    with open(tracer["output_path"] + "surrogate_results.pkl", "wb") as f:
        pickle.dump(tracer, f)