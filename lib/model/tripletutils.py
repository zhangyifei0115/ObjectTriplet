import  matplotlib.pyplot as plt
import torch
from torch.autograd import Variable,Function
# from torch.nn.modules.distance import PairwiseDistance

class PairwiseDistance(Function):

    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)


class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, anchor, positive, negative):
        # print(self.margin)

        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min = 0.0)
        loss = torch.mean(hinge_dist)

        return loss

