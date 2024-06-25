from e3nn.util.datatypes import Path, Chunk
from e3nn import o3, o2

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


class BaseFullTensorProduct(nn.Module):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        *,
        filter_ir_out=None,
        irrep_normalization="component",
        regroup_output=True,
    ):
        """Tensor Product adapted from https://github.com/e3nn/e3nn-jax/blob/cf37f3e95264b34587b3a202ea4c3eb82597307e/e3nn_jax/_src/tensor_products.py#L40-L135"""
        super(BaseFullTensorProduct, self).__init__()
        self.irrep_normalization = irrep_normalization
        self.regroup_output = regroup_output
        self.filter_ir_out = filter_ir_out
        self.setup_irreps(irreps_in1, irreps_in2)
        self.compute_paths_and_cg()

    def setup_irreps(self, irreps_in1, irreps_in2):
        if self.regroup_output:
            self.irreps_in1 = self.Irreps(irreps_in1).regroup()
            self.irreps_in2 = self.Irreps(irreps_in2).regroup()
        else:
            self.irreps_in1 = self.Irreps(irreps_in1)
            self.irreps_in2 = self.Irreps(irreps_in2)

    def compute_paths_and_cg(self):
        paths = {}
        irreps_out = []
        for (mul_1, ir_1), slice_1 in zip(self.irreps_in1, self.irreps_in1.slices()):
            for (mul_2, ir_2), slice_2 in zip(self.irreps_in2, self.irreps_in2.slices()):
                for ir_out in ir_1 * ir_2:
                    if self.filter_ir_out is not None and ir_out not in self.filter_ir_out:
                        continue
                    cg = self.compute_cg(ir_1, ir_2, ir_out)
                    if self.irrep_normalization == "component":
                        cg *= np.sqrt(ir_out.dim)
                    elif self.irrep_normalization == "norm":
                        cg *= np.sqrt(ir_1.dim * ir_2.dim)
                    else:
                        raise ValueError(f"irrep_normalization={self.irrep_normalization} not supported")
                    self.register_cg_buffer(ir_1, ir_2, ir_out, cg)
                    paths[self.get_path_key(ir_1, ir_2, ir_out)] = Path(
                        Chunk(mul_1, ir_1.dim, slice_1), Chunk(mul_2, ir_2.dim, slice_2), Chunk(mul_1 * mul_2, ir_out.dim)
                    )
                    irreps_out.append((mul_1 * mul_2, ir_out))
        self.paths = paths
        irreps_out = self.Irreps(irreps_out)
        self.irreps_out, _, self.inv = irreps_out.sort()

    def compute_cg(self, ir_1, ir_2, ir_out):
        raise NotImplementedError("Subclasses must implement this method")

    def register_cg_buffer(self, ir_1, ir_2, ir_out, cg):
        raise NotImplementedError("Subclasses must implement this method")

    def get_path_key(self, ir_1, ir_2, ir_out):
        raise NotImplementedError("Subclasses must implement this method")

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1, input2, leading_shape = _prepare_inputs(input1, input2)
        chunks = []
        for path_key, (
            (mul_1, input_dim1, slice_1),
            (mul_2, input_dim2, slice_2),
            (output_mul, output_dim, _),
        ) in self.paths.items():
            x1 = input1[..., slice_1].reshape(leading_shape + (mul_1, input_dim1))
            x2 = input2[..., slice_2].reshape(leading_shape + (mul_2, input_dim2))
            cg = self.get_cg_buffer(path_key)
            chunk = torch.einsum("...ui, ...vj, ijk -> ...uvk", x1, x2, cg)
            chunk = torch.reshape(chunk, chunk.shape[:-3] + (output_mul * output_dim,))
            chunks.append(chunk)

        return torch.cat([chunks[i] for i in self.inv], dim=-1)

    def get_cg_buffer(self, path_key):
        raise NotImplementedError("Subclasses must implement this method")


class FullTensorProductO2(BaseFullTensorProduct):
    from e3nn import o2

    Irreps = o2.Irreps

    def compute_cg(self, ir_1, ir_2, ir_out):
        return o2.decomp_3m(ir_1.m, ir_2.m, ir_out.m, ir_1.p, ir_2.p, ir_out.p)

    def register_cg_buffer(self, ir_1, ir_2, ir_out, cg):
        self.register_buffer(f"cg_{ir_1.m}_{ir_1.p}_{ir_2.m}_{ir_2.p}_{ir_out.m}_{ir_out.p}", cg)

    def get_path_key(self, ir_1, ir_2, ir_out):
        return (ir_1.m, ir_1.p, ir_2.m, ir_2.p, ir_out.m, ir_out.p)

    def get_cg_buffer(self, path_key):
        m1, p1, m2, p2, m3, p3 = path_key
        return getattr(self, f"cg_{m1}_{p1}_{m2}_{p2}_{m3}_{p3}")


class FullTensorProductO3(BaseFullTensorProduct):
    from e3nn import o3

    Irreps = o3.Irreps

    def compute_cg(self, ir_1, ir_2, ir_out):
        return o3.wigner_3j(ir_1.l, ir_2.l, ir_out.l)

    def register_cg_buffer(self, ir_1, ir_2, ir_out, cg):
        self.register_buffer(f"cg_{ir_1.l}_{ir_2.l}_{ir_out.l}", cg)

    def get_path_key(self, ir_1, ir_2, ir_out):
        return (ir_1.l, ir_2.l, ir_out.l)

    def get_cg_buffer(self, path_key):
        l1, l2, l3 = path_key
        return getattr(self, f"cg_{l1}_{l2}_{l3}")
