
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborRoutingAgg(nn.Module):
    def __init__(self, hidden_size, routing_iter, device):
        super(NeighborRoutingAgg, self).__init__()
        self.dim = hidden_size
        self.routing_iter = routing_iter
        self.device = device

        self._cache_zero = torch.zeros(1, 1).to(self.device)

    def forward(self, x, x_nb):
        '''
            x： n \times d
            x_nb: n \times m
        '''
        n, m, d = x.size(0), x_nb.size(1), self.dim
        x = F.normalize(x, dim=1)

        z = x[x_nb - 1].view(n, m, d)
        u = None
        for clus_iter in range(self.routing_iter):
            if u is None:
                # p = torch.randn(n, m).to(self.device)
                p = self._cache_zero.expand(n * m, 1).view(n, m)
            else:
                p = torch.sum(z * u.view(n, 1, d), dim=2)
            p = torch.softmax(p, dim=1)
            u = torch.sum(z * p.view(n, m, 1), dim=1)
            u += x.view(n, d)
            if clus_iter < self.routing_iter - 1:
                squash = torch.norm(u, dim=1) ** 2 / (torch.norm(u, dim=1) ** 2 + 1)
                u = squash.unsqueeze(1) * F.normalize(u, dim=1)
                # u = F.normalize(u, dim=1)
        return u.view(n, d)
