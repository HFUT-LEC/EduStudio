import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/ethanfetaya/NRI/blob/master/utils.py
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10, dim=-1):
    """
    NOTE: Stolen from https://github.com/ethanfetaya/NRI/blob/master/utils.py
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dim=dim)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    """
    NOTE: Stolen from https://github.com/ethanfetaya/NRI/blob/master/utils.py
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps, dim=dim)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def kl_categorical(preds, log_prior, concept_num, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (concept_num * preds.size(0))


def kl_categorical_uniform(preds, concept_num, num_edge_types, add_const=False, eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (concept_num * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    # pred: [concept_num, embedding_dim]
    # target: [concept_num, embedding_dim]
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.mean()


class VAELoss(nn.Module):

    def __init__(self, concept_num, edge_type_num=2, prior=False, var=5e-5):
        super(VAELoss, self).__init__()
        self.concept_num = concept_num
        self.edge_type_num = edge_type_num
        self.prior = prior
        self.var = var

    def forward(self, ec_list, rec_list, z_prob_list, log_prior=None):
        time_stamp_num = len(ec_list)
        loss = 0
        for time_idx in range(time_stamp_num):
            output = rec_list[time_idx]
            target = ec_list[time_idx]
            prob = z_prob_list[time_idx]
            loss_nll = nll_gaussian(output, target, self.var)
            if self.prior:
                assert log_prior is not None
                loss_kl = kl_categorical(prob, log_prior, self.concept_num)
            else:
                loss_kl = kl_categorical_uniform(prob, self.concept_num, self.edge_type_num)
            if time_idx == 0:
                loss = loss_nll + loss_kl
            else:
                loss = loss + loss_nll + loss_kl
        return loss / time_stamp_num
