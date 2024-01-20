from linalg import cos_vec_vec
from tqdm.auto import tqdm
from typing import Iterable
import torch


def random_prior(
            K: int) -> torch.Tensor:
    """ Draws a vector form a probability simplex on K
        vertices randomly at uniform.
    """
    xs = torch.rand(K-1)
    x_0 = torch.zeros(1)
    x_1 = torch.ones(1)
    xs = torch.cat([xs, x_0, x_1])
    xs,_ = torch.sort(xs)
    return xs[1:] - xs[:-1]


def get_prior(
            ys: Iterable[int],
            K: int) -> torch.Tensor:
    """ Computes an empirical multinoulli distribution over [K] in ys.
    """
    prior = torch.zeros(K)
    for y in ys:
        prior[y] += 1
    return prior/len(ys)


def WCA(
            gs: torch.Tensor,
            ys: Iterable[int]) -> float:
    
    """ Computes within class alignment of logit gradients.
        Inspired by Fort & Ganguli (2019): https://arxiv.org/abs/1910.05929 
        - gs (K, S, P): a matrix of logit gradients
        - ys: (S): a vector of labels s.t. gs[k][s] corr. to ys[s] for all s, k.
    """
    num_classes = gs.shape[0]
    class_idxs = [[] for k in range(num_classes)]
    for i,y in enumerate(ys):
        class_idxs[y].append(i)
    tot_angle = 0
    for k in tqdm(range(num_classes)):
        coeff = len(class_idxs[k])*(len(class_idxs[k])-1)
        for idx_1 in class_idxs[k]:
            for idx_2 in class_idxs[k]: 
                if idx_1 != idx_2:
                    angle = cos_vec_vec(gs[k][idx_1], gs[k][idx_2])
                    tot_angle += angle/coeff
    return tot_angle/num_classes


def CCA(
            gs: torch.Tensor) -> float:
    
    """ Computes cross class alignment of logit gradients.
        Inspired by Fort & Ganguli (2019): https://arxiv.org/abs/1910.05929 
        - gs (K, S, P): a matrix of logit gradients
        - ys: (S): a vector of labels s.t. gs[k][s] corr. to ys[s] for all s, k.
    """
    num_classes = gs.shape[0]
    tot_angle = 0
    coeff = num_classes*gs.shape[1]*(gs.shape[1]-1)
    for k in tqdm(range(num_classes)):
        for idx_1 in range(gs.shape[1]):
            for idx_2 in range(gs.shape[1]):
                if idx_1 != idx_2:
                    angle = cos_vec_vec(gs[k][idx_1], gs[k][idx_2])
                    tot_angle += angle/coeff              
    return tot_angle