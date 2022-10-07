import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BYOL(torch.nn.modules.loss._Loss):
    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(BYOL, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):

        # L2 normalize
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)

        # positive logits: Nxk, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        l_neg = torch.mm(emb_anchor, emb_positive.t())

        loss = - l_pos.sum()
        return loss

class SimCLR(torch.nn.modules.loss._Loss):
    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(SimCLR, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):
        
        # positive logits: Nx1
        emb_anchor = torch.mm(torch.diag(torch.sum(torch.pow(emb_anchor, 2), axis=1) ** (-0.5)), emb_anchor)
        emb_positive = torch.mm(torch.diag(torch.sum(torch.pow(emb_positive, 2), axis=1) ** (-0.5)), emb_positive)

        N = emb_anchor.shape[0]
        emb_total = torch.cat([emb_anchor, emb_positive], dim=0)

        logits = torch.mm(emb_total, emb_total.t())
        logits[torch.arange(2*N), torch.arange(2*N)] = -1e10

        # apply temperature
        logits /= self.T

        labels = torch.LongTensor(torch.cat([torch.arange(N, 2*N), torch.arange(N)])).to(self.device)

        # loss
        loss = F.cross_entropy(logits, labels)
        l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        return loss