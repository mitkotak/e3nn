from e3nn.util.datatypes import Path, Chunk
from e3nn import o2

import torch
from torch import nn
import numpy as np


def _prepare_inputs(input1, input2):
    dtype = torch.promote_types(input1.dtype, input2.dtype)

    input1 = input1.to(dtype=dtype)
    input2 = input2.to(dtype=dtype)

    leading_shape = torch.broadcast_shapes(input1.shape[:-1], input2.shape[:-1])
    input1 = input1.broadcast_to(leading_shape + (-1,))
    input2 = input2.broadcast_to(leading_shape + (-1,))
    return input1, input2, leading_shape


class FullTensorProduct(nn.Module):
    def __init__(
        self,
        irreps_in1: o2.Irreps,
        irreps_in2: o2.Irreps,
        *,
        filter_ir_out: o2.Irreps = None,
        irrep_normalization: str = "component",
        regroup_output: bool = True,
    ):
        """Tensor Product adapted from https://github.com/e3nn/e3nn-jax/blob/cf37f3e95264b34587b3a202ea4c3eb82597307e/e3nn_jax/_src/tensor_products.py#L40-L135"""
        super(FullTensorProduct, self).__init__()

        # Need to do fancy string checking for o3/o2
        # Stop mixing between o3/o2 irreps. Throw an assert
        # We will assume o3 unless specified

        if regroup_output:
            irreps_in1 = o2.Irreps(irreps_in1).regroup()
            irreps_in2 = o2.Irreps(irreps_in2).regroup()

        paths = {}
        irreps_out = []
        for (mul_1, ir_1), slice_1 in zip(irreps_in1, irreps_in1.slices()):
            for (mul_2, ir_2), slice_2 in zip(irreps_in2, irreps_in2.slices()):
                for ir_out in ir_1 * ir_2:  # Might not be implemented correctly
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue
                    cg = o2.decomp_3m(ir_1.m, ir_2.m, ir_out.m, ir_1.p, ir_2.p, ir_out.p)
                    if irrep_normalization == "component":
                        cg *= np.sqrt(ir_out.dim)
                    elif irrep_normalization == "norm":
                        cg *= np.sqrt(ir_1.dim * ir_2.dim)
                    else:
                        raise ValueError(f"irrep_normalization={irrep_normalization} not supported")
                    self.register_buffer(f"cg_{ir_1.m}_{ir_1.p}_{ir_2.m}_{ir_2.p}_{ir_out.m}_{ir_out.p}", cg)
                    paths[(ir_1.m, ir_1.p, ir_2.m, ir_2.p, ir_out.m, ir_out.p)] = Path(
                        Chunk(mul_1, ir_1.dim, slice_1), Chunk(mul_2, ir_2.dim, slice_2), Chunk(mul_1 * mul_2, ir_out.dim)
                    )
                    irreps_out.append((mul_1 * mul_2, ir_out))
        self.paths = paths
        irreps_out = o2.Irreps(irreps_out)
        print(irreps_out)
        self.irreps_out, _, self.inv = irreps_out.sort()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
    ) -> torch.Tensor:
        input1, input2, leading_shape = _prepare_inputs(input1, input2)
        chunks = []
        for (m1, p1, m2, p2, m3, p3), (
            (mul_1, input_dim1, slice_1),
            (mul_2, input_dim2, slice_2),
            (output_mul, output_dim, _),
        ) in self.paths.items():
            x1 = input1[..., slice_1].reshape(
                leading_shape
                + (
                    mul_1,
                    input_dim1,
                )
            )
            x2 = input2[..., slice_2].reshape(
                leading_shape
                + (
                    mul_2,
                    input_dim2,
                )
            )
            cg = getattr(self, f"cg_{m1}_{p1}_{m2}_{p2}_{m3}_{p3}")
            chunk = torch.einsum("...ui, ...vj, ijk -> ...uvk", x1, x2, cg)
            chunk = torch.reshape(chunk, chunk.shape[:-3] + (output_mul * output_dim,))
            chunks.append(chunk)

        return torch.cat([chunks[i] for i in self.inv], dim=-1)
