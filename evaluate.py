from .models.model_base import ClassificationModelBase
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Categorical
from typing import Tuple
import torch.nn.functional as F
import torch
import numpy as np


def compute_loss_acc(
            dataset: Dataset,
            model: ClassificationModelBase) -> Tuple[float, float]:
    
    """ Computes loss and accuracy of the model on dataset.
    """
    model.eval()
    val_dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False)
    loss = 0.
    ys = []
    fs = []
    model.eval()
    with torch.no_grad():
        for X, y in val_dataloader:
            y = y.to(model.device)
            outs = model(X)
            fs.append(torch.argmax(outs, dim=-1).squeeze())
            ys.append(y)
            curr_loss = F.cross_entropy(outs, y)
            loss += len(y)*curr_loss.item()
    y_pred = torch.cat(fs)
    y_true = torch.cat(ys)
    acc = (y_true==y_pred).float().mean().item()
    loss = loss/len(dataset)
    return loss, acc


def get_entropy(
            model: ClassificationModelBase,
            dataset: Dataset) -> torch.Tensor:

    """ Computes prediction entropy for each sample in dataset.
    """
    model.eval()
    entropies = []
    inputs = []
    for X,_ in dataset:
        if X.shape[0]>1:
            X = X.unsqueeze(0)
        inputs.append(X)
    inputs = torch.vstack(inputs).to(model.device)
    outs = model(inputs)
    for out in outs:
        out = out.reshape(1,-1)
        if out.isnan().any() or out.isinf().any():
            probs = torch.zeros(out.numel())
            probs[0] += 1.
        else:
            probs = F.softmax(out, dim=-1).detach()
        ent = Categorical(probs).entropy().item()
        entropies.append(ent)
    entropies = torch.Tensor(entropies)
    return entropies


def ECE(
            model: ClassificationModelBase,
            dataset: Dataset) -> float:
    
    """ Computes Expected Calibration Error (ECE)
        of the model on dataset.
        See https://arxiv.org/pdf/1706.04599.pdf
    """
    y_corr_bin = {i:[] for i in range(10)} 
    confidence = {i:[] for i in range(10)} 
    dataloader = DataLoader(dataset, batch_size=1)
    for X,y in dataloader:
        X = X.to(model.device)
        y = y.to(model.device)
        with torch.no_grad():
            out = model(X)
            prob = torch.max(F.softmax(out, dim=-1)).squeeze()
        if prob.isnan().any() or prob.isinf().any():
            return float('inf')
        pred = torch.argmax(out.squeeze())
        key = int(10*prob.item()) if prob<1 else 9
        y_corr_bin[key].append(int(y==pred))
        confidence[key].append(prob.item())
    ece = 0
    for k,b in y_corr_bin.items():
        if len(b)>0:
            acc = np.mean(b)
            conf = np.mean(confidence[k])
            ece += abs(acc-conf)*len(b)
    return ece/len(dataset)