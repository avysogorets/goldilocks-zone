from torch.utils.data import Dataset
from models.model_base import ClassificationModelBase
from scipy.sparse.linalg import svds
import torch
import numpy as np


def NC1(
            model: ClassificationModelBase,
            dataset: Dataset,
            num_classes: int) -> float:
    
    """ Computes the NC1 metric of neural collapse.
    """
    representations = [[] for k in range(num_classes)]
    for i,module in enumerate(model.module_list):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            last_layer_idx = i
    extractor = torch.nn.Sequential(*model.module_list[:last_layer_idx])
    for X,y in dataset:
        with torch.no_grad():
            representation = extractor(X.to(model.dtype))
        if representation.isnan().any() or representation.isinf().any():
            return float('inf')
        representations[y].append(representation)
    for k in range(num_classes):
        representations[k] = torch.vstack(representations[k])
    mu_k = [torch.mean(representations[k], dim=0) for k in range(num_classes)]
    mu_g = torch.mean(torch.vstack(mu_k), dim=0)
    dim = representations[0].shape[1]
    SW = torch.zeros((dim, dim))
    for y in range(num_classes):
        for representation in representations[y]:
            vec = (representation-mu_k[y]).reshape(-1,1)
            SW += torch.matmul(vec, vec.T)
    SW /= len(dataset)
    SG = torch.zeros((dim, dim))
    for y in range(num_classes):
        vec = (mu_k[y]-mu_g).reshape(-1,1)
        SG += torch.matmul(vec, vec.T)
    SG = (SG/num_classes).numpy()
    eigvec, eigval, _ = svds(SG, k=num_classes-1)
    inv_Sb = eigvec@np.diag(eigval**(-1))@eigvec.T 
    nc1 = np.trace(SW.numpy()@inv_Sb)/num_classes
    return nc1

def NC2(
            model: ClassificationModelBase,
            dataset: Dataset,
            num_classes: int) -> float:
    
    """ Computes the NC2 metric of neural collapse.
    """
    representations = [[] for k in range(num_classes)]
    for i,module in enumerate(model.module_list):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            last_layer_idx = i
    extractor = torch.nn.Sequential(*model.module_list[:last_layer_idx])
    for X,y in dataset:
        with torch.no_grad():
            representation = extractor(X.to(model.dtype))
        if representation.isnan().any() or representation.isinf().any():
            return float('inf')
        representations[y].append(representation)
    for k in range(num_classes):
        representations[k] = torch.vstack(representations[k])
    mu_k = [torch.mean(representations[k], dim=0) for k in range(num_classes)]
    mu_g = torch.mean(torch.vstack(mu_k), dim=0)
    M = []
    for k in range(num_classes):
        vec = (mu_k[k]-mu_g).reshape(1,-1)
        vec = vec / vec.norm()
        M.append(vec)
    M = torch.vstack(M)
    A = torch.matmul(M,M.T)/torch.norm(torch.matmul(M,M.T), p='fro')
    one_k = torch.ones((num_classes, 1))
    I_k = torch.eye(num_classes)
    const_1 = 1./np.sqrt(num_classes-1)
    const_2 = 1./num_classes
    factor = torch.matmul(one_k, one_k.T)
    nc2 = torch.norm(A-const_1*(I_k-const_2*factor) , p='fro')
    return nc2.item()

def NC3(
            model: ClassificationModelBase,
            dataset: Dataset,
            num_classes: int) -> float:
    
    """ Computes the NC3 metric of neural collapse.
    """
    representations = [[] for k in range(num_classes)]
    for i,module in enumerate(model.module_list):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            last_layer_idx = i
    extractor = torch.nn.Sequential(*model.module_list[:last_layer_idx])
    for X,y in dataset:
        with torch.no_grad():
            representation = extractor(X.to(model.dtype))
        if representation.isnan().any() or representation.isinf().any():
            return float('inf')
        representations[y].append(representation)
    for k in range(num_classes):
        representations[k] = torch.vstack(representations[k])
    mu_k = [torch.mean(representations[k], dim=0) for k in range(num_classes)]
    mu_g = torch.mean(torch.vstack(mu_k), dim=0)
    M = []
    for k in range(num_classes):
        vec = (mu_k[k]-mu_g).reshape(1,-1)
        vec = vec / vec.norm()
        M.append(vec)
    M = torch.vstack(M)
    C = model.module_list[last_layer_idx].weight.data
    AM = torch.matmul(C, M.T)/torch.norm(torch.matmul(C, M.T), p='fro')
    one_k = torch.ones((num_classes, 1))
    I_k = torch.eye(num_classes)
    const_1 = 1./np.sqrt(num_classes-1)
    const_2 = 1./num_classes
    factor = torch.matmul(one_k, one_k.T)
    nc2 = torch.norm(AM-const_1*(I_k-const_2*factor) , p='fro')
    return nc2.item()

def NC4(
            model: ClassificationModelBase,
            dataset: Dataset,
            num_classes: int) -> float:
    
    """ Computes the NC4 metric of neural collapse.
    """
    representations = [[] for k in range(num_classes)]
    for i,module in enumerate(model.module_list):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            last_layer_idx = i
    extractor = torch.nn.Sequential(*model.module_list[:last_layer_idx])
    for X,y in dataset:
        with torch.no_grad():
            representation = extractor(X.to(model.dtype))
        if representation.isnan().any() or representation.isinf().any():
            return float('inf')
        representations[y].append(representation)
    for y in range(num_classes):
        representations[y] = torch.vstack(representations[y])
    mu_k = [torch.mean(representations[y], dim=0) for y in range(num_classes)]
    nn_score = 0
    for y in range(num_classes):
        for representation in representations[y]:
            true_class_norm = (representation-mu_k[y]).norm()
            nn_correct = True
            for c in range(num_classes):
                 if (representation-mu_k[c]).norm()<true_class_norm:
                    nn_correct = False
            nn_score += int(nn_correct)
    return 1-nn_score/len(dataset)