from .models.model_base import ClassificationModelBase
from .models.utils import get_current_gradients
from .models.utils import get_trainable_parameters, \
    get_trainable_parameters_in_layer, \
    set_trainable_parameters, \
    set_trainable_parameters_in_layer, \
    flip_parameters_to_tensors
from typing import Union, Tuple
from torch.utils.data import Dataset, DataLoader
from hessian_eigenthings import compute_hessian_eigenthings
from copy import deepcopy
import numpy as np
import torch
import math
import torch.nn.functional as F


def cos_vec_vec(
            vec_1: torch.Tensor,
            vec_2: torch.Tensor) -> float:
    
    """ Computes cosine similarity between vec_1 and vec_2.
    """
    vec_1 = vec_1.cpu()
    vec_2 = vec_2.cpu()
    if vec_1.norm() > 0 and vec_2.norm() > 0:
        cos = np.dot(vec_1,vec_2)/(vec_1.norm()*vec_2.norm())
        return cos.item()
    else:
        return float('inf')


def get_random_ortho_matrix(
            D: int,
            d: int,
            device: torch.device) -> torch.Tensor:
    
    """ Computes a random (D, d) orthonormal matrix.
        Adapted from https://github.com/jeffiar/cs229-final-project
    """
    M = torch.zeros(D, d, device=device)
    for i in range(d):
        col = torch.zeros(D)
        prob = 1 / math.sqrt(D)
        col[torch.rand(D) < prob] = 1
        col[torch.rand(D) < 0.5] *= -1
        col /= col.norm()
        M[:,i] = col
    return M


def sparse_vector(
            D: int,
            n: int) -> torch.Tensor:
    
    """ Computes a D dimensional sparse vector with n non-zero entries.
        Adapted from https://github.com/jeffiar/cs229-final-project
    """
    vec = torch.zeros(D).float()
    idxs = np.random.choice(range(D), size=n, replace=False)
    signs = np.random.choice([-1.,1.],size=n)
    cnsts = torch.from_numpy(signs*math.sqrt(n)*np.ones(n)).float()
    vec[idxs] = cnsts
    return vec


def goldilocks(
            model: ClassificationModelBase,
            dataset: Dataset,
            dim: int,
            device: torch.device,
            layer: Union[int, None] = None) -> Tuple[float, float]:
    if layer is None:
        w = get_trainable_parameters(model).to(device)
    else:
        w = get_trainable_parameters_in_layer(model, layer).to(device)
    w = w.squeeze()
    ns = np.random.choice(range(w.numel()), size=dim)
    M = torch.vstack([sparse_vector(w.numel(), n) for n in ns])
    R = get_random_ortho_matrix(len(w), dim, device).to(model.dtype)
    L,V = eigenvvs(
            model=model,
            dataset=dataset,
            top_k=-1,
            is_subspace=True,
            R=R,
            layer=layer)
    L = L.to(model.dtype)
    frac_pos = sum(L>0).item()/L.numel()
    trace = torch.sum(L).item()
    norm = torch.linalg.norm(L).item()
    if norm == 0:
        trace_norm = float('inf')
    else:
        trace_norm = trace / norm
    return frac_pos, trace_norm


def eigenvvs(
            model: ClassificationModelBase,
            dataset: Dataset,
            top_k: int,
            is_subspace: bool = False,
            R: Union[torch.Tensor, None] = None,
            layer: Union[int, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """ Computes top_k eigenvalues and eigenvectors of the model
        on train_dataset (top_k = -1 means all).
         - is_subspace: use a subspace specified by R
         - R (P, d): column vectors spanning the subsapce
         - layer: compute the Hessian for a specified tensor layer.
    """
    H = hessian(model, dataset, is_subspace, R, layer)
    H = H.squeeze()
    L, V = torch.linalg.eig(H)
    V = V.T
    L = torch.real(L)
    V = torch.real(V)
    o = sorted(range(len(L)), key=lambda i: L[i], reverse=True)
    if top_k>0:
        V_k = V[o][:top_k]
        L_k = L[o][:top_k]
    else:
        V_k = V[o]
        L_k = L[o]
    return L_k, V_k
                
                
def hessian(
            model: ClassificationModelBase,
            dataset: Dataset,
            is_subspace: bool = False,
            R: Union[torch.Tensor, None] = None,
            layer: Union[int, None] = None) -> torch.Tensor:
    
    """ Computes the full Hessian of the model on the dataset.
         - is_subspace: use a subspace specified by R
         - R (P, d): column vectors spanning the subspace
         - layer: compute the Hessian for a specified tensor layer.
    
        Warning: this function invokes flip_parameters_to_tensors,
        which removes all parameters from the model; hence, it can
        be called only once. After invoking this function, it is
        impossible to retrieve gradients with respect to model's
        parameters as they are no longer leaf tensors. Likewise,
        no training of the model can be arranged thereafter.
    """
    model.zero_grad()
    model.eval()
    if not is_subspace:
        if layer is None:
            w = get_trainable_parameters(model)
        else:
            w = get_trainable_parameters_in_layer(model, layer)
        w.requires_grad_(True)
        R = torch.eye(w.numel()).to(model.device)
        d = torch.zeros_like(w).to(model.device)
    else:
        if layer is None:
            d = get_trainable_parameters(model)
        else:
            d = get_trainable_parameters_in_layer(model, layer)
        w = torch.zeros(R.shape[1]).requires_grad_(True)
    w = w.reshape(-1,1).to(model.dtype).to(model.device)
    model_copy = deepcopy(model)
    flip_parameters_to_tensors(model_copy)
    dataloader = DataLoader(dataset, batch_size=128)
    def func(w):
        W = torch.mm(R, w)
        W = W + d.reshape(W.shape)
        if layer is None:
            set_trainable_parameters(model_copy, W)
        else:
            set_trainable_parameters_in_layer(model_copy, W, layer)
        y_preds = []
        y_trues = []
        model_copy.eval()
        for X,y in dataloader:
            X = X.to(model_copy.device)
            y = y.to(model_copy.device)
            y_preds.append(model_copy(X))
            y_trues.append(y)
        y_preds = torch.vstack(y_preds)
        y_trues = torch.cat(y_trues)
        return F.cross_entropy(y_preds, y_trues)
    H = torch.autograd.functional.hessian(func, w).squeeze()
    del model_copy
    return H
    

def eigenthings(
            model: ClassificationModelBase,
            loss: callable,
            dataset: Dataset,
            num_things: int) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """ Computes num_things eigenvalues and eigenvectors.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
    use_gpu = True if model.device.type == 'cuda' else False
    vals, vects = compute_hessian_eigenthings(
            model,
            dataloader,
            loss,
            num_things,
            use_gpu=use_gpu)
    return vals, vects


def _get_G_term(
            J: torch.Tensor,
            p: torch.Tensor) -> torch.Tensor:
    
    """ Computes the G-term from the Gauss-Newton decomposition.
        - J (K, d): Jacobian
        - p (K): softmax output
    """
    num_p = J.shape[1]
    A = p.reshape(1,-1) @ J
    Jp = J*p.repeat(num_p, 1).T
    G = (J.T @ Jp) - (A.T @ A)
    return G
          

def get_G_term(
            J: torch.Tensor,
            p: torch.Tensor) -> torch.Tensor:
    
    """ Computes the G-term from the Gauss-Newton decomposition.
        - J (S, K, d): Jacobian
        - p (S, K): softmax output
    """
    S = J.shape[0]
    Gs = torch.vstack([_get_G_term(J[s], p[s]).unsqueeze(0) for s in range(S)])
    return Gs.mean(dim=0)


def get_Jacobian(
            model: ClassificationModelBase,
            dataset: Dataset,
            K: int,
            R: Union[torch.Tensor, None] = None) -> torch.Tensor:
    
    """ Computes Jacobian of the model on dataset.
        - K: number of classes;
        - R (P, d): subspace (if any).
        Returns:
        - J (S, K, d): Jacobian with respect to samples in dataset.
    """
    w = get_trainable_parameters(model)
    d = w.numel() if R is None else R.shape[1]
    J = torch.zeros(len(dataset), K, d).to(model.dtype)
    for s,(X,_) in enumerate(dataset):
        for k in range(K):
            model.zero_grad()
            if X.shape[0]>1:
                logit = model(X.unsqueeze(0)).squeeze()[k]
            else:
                logit = model(X).squeeze()[k]
            logit.backward()
            g = get_current_gradients(model).detach().squeeze()
            if R is not None:
                g = g.reshape(1,-1)@R
            J[s][k] = g.reshape(-1)
    model.zero_grad()
    return J


def Gamma(
            P: torch.Tensor) -> float:
    
    """ Computes Gamma(P) for a matrix of softmax outputs.
        - P (S, K): matrix of softmax outputs. 
    """
    M = sum((torch.diag(p.reshape(-1))-p.reshape(-1,1)@p.reshape(1,-1)) for p in P)
    M = M/P.shape[0]
    trace = torch.trace(M)
    norm = torch.linalg.norm(M, 'fro')
    if trace == 0 or norm == 0:
        return 0
    else:
        return trace/norm
    

def EG_curvature(
            var_E: float,
            var_C: float,
            d: int,
            P: torch.Tensor):
    
    """ Computes positive curvature of the expected G_term per Eq. 9.
        - var_E: estimated variance of logit gradients
        - var_C: estimated variance of logit gradient means
        - d: dimension of the subspace
        - P (S, K): matrix of softmax outputs. 
    """
    gamma = Gamma(P)
    if gamma >= 0 and gamma < 1:
        return 0
    numer = np.sqrt(d)*(var_E+var_C)
    denom = np.sqrt(var_E**2 + 2*var_E*var_C + d*(var_C**2)/(gamma**2))
    res = numer / denom
    return res


def hessian_vector_product(
            model: ClassificationModelBase,
            loss: torch.Tensor,
            vec: torch.Tensor) -> torch.Tensor:
    
    """ Computes hessian-vector product H @ vec wrt to 
        model parameters given a scalar loss tensor.
    """
    grad = torch.autograd.grad(
            loss, 
            model.parameters(), 
            create_graph=True, 
            allow_unused=True)
    grad = torch.cat([g.reshape(-1) for g in grad if g is not None])
    g = grad @ vec
    hvp = torch.autograd.grad(g, model.parameters(), allow_unused=True)
    hvp = torch.cat([g.reshape(-1) for g in hvp if g is not None])
    return hvp.detach()


def hutch_tr_H(
            model: ClassificationModelBase,
            train_X: torch.Tensor,
            train_y: torch.LongTensor,
            maxiter: int = 100) -> float:
    
    """ Computes trace of the model Hessian computed on data
        (train_X, train_y) using Hutchinson's stochastic
        approximation method using maxiter iterations.
    """
    w = get_trainable_parameters(model)
    traces = []
    for _ in range(maxiter):
        model.zero_grad()
        vec = torch.randint_like(w, high=2).squeeze()
        vec[vec == 0] = -1
        loss = F.cross_entropy(model(train_X), train_y)
        t = vec@hessian_vector_product(model, loss, vec)
        traces.append(t.item())
    return np.mean(traces).item()
  

def hutch_fr_H(
            model: ClassificationModelBase,
            train_X: torch.Tensor,
            train_y: torch.LongTensor,
            maxiter: int = 100) -> float:
    
    """ Computes Frobenius norm of the model Hessian computed
        on data (train_X, train_y) using Hutchinson's stochastic
        approximation method using maxiter iterations.
    """
    w = get_trainable_parameters(model)
    frobs = []
    for _ in range(maxiter):
        model.zero_grad()
        vec = torch.randint_like(w, high=2).squeeze()
        vec[vec == 0] = -1
        loss = F.cross_entropy(model(train_X), train_y)
        hvp = hessian_vector_product(model, loss, vec).squeeze()
        frobs.append(hvp@hvp)
    return np.sqrt(np.mean(frobs)).item()

