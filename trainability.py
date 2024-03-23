from .linalg import eigenthings, cos_vec_vec
from .data import DataFactory
from .models import ModelFactory
from .models.utils import get_layer_idxs, \
    get_trainable_parameters, \
    set_trainable_parameters, \
    get_current_gradients
from .evaluate import compute_loss_acc
from torch.utils.data import DataLoader
import torch
import os
import torch.nn.functional as F
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='global seed')
    parser.add_argument('--is_double', action="store_true", default=False, help='use float64 as data type')
    parser.add_argument('--lr', type=float, default=0.1, help='base learning rate eta_0')
    parser.add_argument('--num_epochs', type=int, default=5000, help='number of epochs')  
    parser.add_argument('--log_every_k', type=int, default=100, help='log iterations')   
    parser.add_argument('--path', type=str, default='data/downloads', help='path to data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory')
    parser.add_argument('--scale', type=float, default=10, help='initialization scale (alpha)')
    parser.add_argument('--activation', type=str, default='ReLU', help='activation function (ReLU / Tanh / Sigmoid / Identity)')
    parser.add_argument('--network', type=str, default='LeNet300100', help='model architecture (see models module)')
    parser.add_argument('--dataset', type=str, default='FMNIST', help='dataset (see data module)')
    args = parser.parse_args()
    return args


def get_fileid(args):
    fileid = (f"{args.seed}_"
            f"{args.network}_"
            f"{args.dataset}_"
            f"{args.activation}_"
            f"{args.scale}_"
            f"{args.num_epochs}_"
            f"{args.lr}")
    return fileid


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'[device: {device} is ready]')
    return device


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    dtype = torch.float64 if args.is_double else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)
    fileid = get_fileid(args)
    data = DataFactory(
            dataset_name=args.dataset,
            path=args.path,
            to_transform=False)
    model = ModelFactory(
            model_name=args.network,
            in_shape=data.in_shape,
            activation=args.activation,
            num_classes=data.num_classes,
            dtype=dtype,
            device=device)
    metrics = {
        "train-accs": [],
        "test-accs": [],
        "g-norms": [],
        "w-norms": [],
        "angles-ww": [],
        "angles-wg": [],
        "zero-logits": [],
        "sharpness": []}
    num_layers = len(get_layer_idxs(model))
    init = get_trainable_parameters(model).clone().squeeze()
    set_trainable_parameters(model, (args.scale*init).to(dtype))
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=(1./(args.scale**(num_layers-2)))*args.lr)
    dataloader = DataLoader(data.datasets["train"],
            batch_size=len(data.datasets["train"]))
    iteration = 0
    while iteration < args.num_epochs:
        for X,y in dataloader:
            optimizer.zero_grad()
            model.zero_grad()
            y = y.to(model.device)
            outs = model(X)
            loss = F.cross_entropy(outs, y)
            loss.backward()
            g = get_current_gradients(model).detach().squeeze()
            optimizer.step()
            if iteration % args.log_every_k == 0:
                model.eval()
                y_pred = outs.detach().argmax(dim=-1)
                train_acc = torch.mean((y_pred==y).float()).item()
                metrics["train-accs"].append(train_acc)
                _, acc = compute_loss_acc(data.datasets["dev"], model)
                metrics["test-accs"].append(acc)
                print(f'[epoch: {iteration}][train/test acc: {train_acc:.4f}/{acc:.4f}]')
                L,_ = eigenthings(
                        model=model,
                        loss=F.cross_entropy,
                        dataset=data.datasets["train"],
                        num_things=1)
                metrics["sharpness"].append(max(L[0], 1e-15))
                w = get_trainable_parameters(model).detach().squeeze()
                metrics["g-norms"].append(g.norm().item())
                metrics["w-norms"].append(w.norm().item())
                metrics["angles-wg"].append(cos_vec_vec(w, g))
                metrics["angles-ww"].append(cos_vec_vec(w, init))
                zero_logits = torch.sum(outs==0)/outs.numel()
                metrics["zero-logits"].append(zero_logits.item())
                model.train()
            iteration+=1

    f = open(os.path.join(args.output_dir, fileid+'.json'), 'w')
    json.dump(metrics, f)
    f.close()




