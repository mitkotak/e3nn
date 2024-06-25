import torch
from e3nn import o3
import pytest


@pytest.mark.parametrize("irreps_in1", ["0e", "0e + 1e"])
@pytest.mark.parametrize("irreps_in2", ["2x0e", "2x0e + 3x1e"])
def test_sparsefulltp(irreps_in1, irreps_in2):
    x = o3.Irreps(irreps_in1).randn(-1)
    y = o3.Irreps(irreps_in2).randn(-1)

    tp = o3.FullTensorProduct(irreps_in1, irreps_in2)
    tp_pt2 = torch.compile(o3.experimental.FullTensorProductSparse(irreps_in1, irreps_in2), fullgraph=True, disable=True)
    torch.testing.assert_close(tp_pt2(x, y), tp(x, y))
