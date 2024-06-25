# flake8: noqa

from e3nn.util.datatypes import Path, Chunk
from e3nn import o3
from e3nn.o3.experimental._full_tp import _prepare_inputs

import torch
from torch import nn
import numpy as np

def transpose(x):
    "Yikes !"
    return torch.permute(x, list(range(x.ndim))[::-1])

class FullTensorProductSparse(nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        *,
        leading_shape = (),
        filter_ir_out: o3.Irreps = None,
        irrep_normalization: str = "component",
        regroup_output: bool = True,
    ):
        """Tensor Product adapted from https://github.com/e3nn/e3nn-jax/blob/cf37f3e95264b34587b3a202ea4c3eb82597307e/e3nn_jax/_src/tensor_products.py#L40-L135"""
        super(FullTensorProductSparse, self).__init__()

        if regroup_output:
            irreps_in1 = o3.Irreps(irreps_in1).regroup()
            irreps_in2 = o3.Irreps(irreps_in2).regroup()

        paths = {}
        m3s = {}
        m1m2s = {}
        irreps_out = []
        for (mul_1, ir_1), slice_1 in zip(irreps_in1, irreps_in1.slices()):
            for (mul_2, ir_2), slice_2 in zip(irreps_in2, irreps_in2.slices()):
                for ir_out in ir_1 * ir_2:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue
                    l1, l2, l3 = ir_1.l, ir_2.l, ir_out.l
                    cg = o3.wigner_3j(l1, l2, l3)
                    chunk = torch.empty((2 * l3 + 1, mul_1, mul_2) + leading_shape)
                    self.register_buffer(f"chunk_{l1}_{l2}_{l3}", chunk)
                    for m3 in range(-l3, l3 + 1):
                        if (l1, l2, l3) in m3s:
                            m3s[(l1, l2, l3)].append(m3)
                        else:
                            m3s[(l1, l2, l3)] = [m3]
                        for m1 in range(-l1, l1 + 1):
                            for m2 in set([m3 - m1, m3 + m1, -m3 + m1, -m3 - m1]):
                                if (m2 < -l2) or (m2 > l2):
                                    continue
                                if (l1, l2, l3, m3) in m1m2s:
                                    m1m2s[(l1, l2, l3, m3)].append((m1, m2))
                                else:
                                    m1m2s[(l1, l2, l3, m3)] = [(m1, m2)]
                                cg_coeff = cg[l1 + m1, l2 + m2, l3 + m3]
                                if irrep_normalization == "component":
                                    cg_coeff *= np.sqrt(ir_out.dim)
                                elif irrep_normalization == "norm":
                                    cg_coeff *= np.sqrt(ir_1.dim * ir_2.dim)
                                else:
                                    raise ValueError(f"irrep_normalization={irrep_normalization} not supported")
                                self.register_buffer(f"cg_{l1}_{m1}_{l2}_{m2}_{l3}_{m3}", cg_coeff)

                    paths[(l1, l2, l3)] = Path(
                        Chunk(mul_1, ir_1.dim, slice_1), Chunk(mul_2, ir_2.dim, slice_2), Chunk(mul_1 * mul_2, ir_out.dim)
                    )
                    irreps_out.append((mul_1 * mul_2, ir_out))
        self.paths = paths
        self.m3s = m3s
        self.m1m2s = m1m2s
        irreps_out = o3.Irreps(irreps_out)
        self.irreps_out,_, self.inv = irreps_out.sort()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
    ) -> torch.Tensor:
        input1, input2, leading_shape = _prepare_inputs(input1, input2)
        chunks = []
        for (l1, l2, l3), (
            (mul_1, input_dim1, slice_1),
            (mul_2, input_dim2, slice_2),
            (output_mul, output_dim, _),
        ) in self.paths.items():
            x1 = input1[..., slice_1].reshape(leading_shape + (mul_1, input_dim1))
            x1_t = transpose(x1)
            x2 = input2[..., slice_2].reshape(leading_shape + (mul_2, input_dim2))
            x2_t = transpose(x2)

            chunk = getattr(self, f"chunk_{l1}_{l2}_{l3}")
            for m3 in self.m3s[(l1, l2, l3)]:
                sum = 0
                for m1, m2 in list(self.m1m2s[(l1, l2, l3, m3)]):
                    cg_coeff = getattr(self, f"cg_{l1}_{m1}_{l2}_{m2}_{l3}_{m3}")
                    path = torch.einsum("...u, ...v -> uv... ", x1_t[l1 + m1, ...], x2_t[l2 + m2, ...])
                    path *= cg_coeff
                    sum += path
                chunk[l3 + m3, ...] = sum
            chunk = transpose(chunk)
            chunk = torch.reshape(chunk, leading_shape + (output_mul * output_dim, ))
            chunks.append(chunk)
        
        return torch.cat([chunks[i] for i in self.inv], dim=-1)