import numpy as np

import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm


class Flows(nn.Module): 
    def __init__(self, state, dims_in=128):
        super(Flows, self).__init__()
        self.dims_in = dims_in 
        self.dims_hid = state['flows_hidden'] 
        self.num_steps = state['flows_steps']
        self.num_layers = state['flows_layers']
        self.dropout = state['flows_dropout']
        self.clamp = state['flows_clamp'] 
        self.model = state['flows_model']
        self.verbose = state['verbose']
        self.net = self.build()

    def build(self):
        # coupling network
        def subnet_fc(dims_in, dims_out):
            layers = [nn.Linear(dims_in, self.dims_hid), nn.Dropout(self.dropout), nn.ReLU()]
            for _ in range(self.num_layers):
                layers.extend([nn.Linear(self.dims_hid, self.dims_hid), nn.Dropout(self.dropout), nn.ReLU()])
            layers.append(nn.Linear(self.dims_hid, dims_out))
            return nn.Sequential(*layers)
        # coupling block
        coupling_block = {
            'glow': Fm.GLOWCouplingBlock,
            'gin': Fm.GINCouplingBlock,
            'realnvp': Fm.RNVPCouplingBlock,
            'nice': Fm.NICECouplingBlock
        }
        # coupling params
        coupling_params = {'subnet_constructor': subnet_fc}
        if self.model != 'nice':
            coupling_params['clamp'] = self.clamp
        # build the flow
        nodes = [Ff.InputNode(self.dims_in, name='input')]
        for k in range(self.num_steps):
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}, name=f'actnorm_{k}'))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': (k // 2) * self.num_steps + (k % 2)}, name=f'permute_{k}'))
            nodes.append(Ff.Node(nodes[-1], coupling_block[self.model], coupling_params, name=f'coupling_{k}'))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        # build the reversible model
        return Ff.ReversibleGraphNet(nodes, verbose=self.verbose)
        
    def forward(self, x):
        return self.net(x)


class LogLikelihood(nn.Module):
    def __init__(self, state):
        super(LogLikelihood, self).__init__()
        self.q = state['q_nll']
        self.calc_loss = {
            0.0: self.min_loss,
            1.0: self.mean_loss
        }.get(self.q, self.quantile_loss)

    def min_loss(self, ll):
        return -torch.min(ll)

    def mean_loss(self, ll):
        return -torch.mean(ll)

    def quantile_loss(self, ll):
        return -torch.quantile(ll, self.q)

    def forward(self, z, log_jac_det):
        prior = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior.view(z.size(0), -1).sum(-1) 
        ll = prior_ll + log_jac_det

        nll = self.calc_loss(ll)
        
        return ll, nll

