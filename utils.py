import torch

from const import START_TAG, END_TAG

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ind, device, tag=False):
    idxs = [to_ind[w] for w in seq]
    if tag:
        idxs = [to_ind[START_TAG]] + idxs + [to_ind[END_TAG]]
    return torch.tensor(idxs, dtype=torch.long).to(device)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))