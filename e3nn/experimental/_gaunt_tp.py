# Code borrowed from Ameya Daigavane

from e3nn import o3

import torch
from torch import nn

class GauntTensorProductFixedParity(nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        lmax: int,
        num_channels: int,
        res_beta: int = None,
        res_alpha: int = None,
    ):
        super(GauntTensorProductFixedParity, self).__init__()
        
        self.linear_in1 = o3.Linear(irreps_in=irreps_in1,
                                    irreps_out=o3.s2_irreps(lmax, p_val=self.p_val1, p_arg=-1))
        self.linear_in2 = o3.Linear(irreps_in=irreps_in2,
                                    irreps_out=o3.s2_irreps(lmax, p_val=self.p_val2, p_arg=-1))
        self.register_buffer("to_s2grid1", o3.ToS2Grid(lmax=lmax,
                                                       res=(res_beta, res_alpha),
                                                       ))
        self.register_buffer("from_s2grid", o3.FromS2Grid(lmax=lmax, res=(res_beta, res_alpha)))

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        return self.from_s2grid(self.to_s2grid(input1) * self.to_s2grid(input2))