import torch
from e3nn import o2
import pytest


@pytest.mark.parametrize("irreps_in1", ["0e + 1", "0e"])
@pytest.mark.parametrize("irreps_in2", ["0e", "0e + 1"])
def test_fulltp(irreps_in1, irreps_in2):
    x = o2.Irreps(irreps_in1).randn(10, -1)
    y = o2.Irreps(irreps_in2).randn(10, -1)

    tp_pt2 = torch.compile(o2.experimental.FullTensorProductv2(irreps_in1, irreps_in2), fullgraph=True)
    tp = o2.FullTensorProduct(irreps_in1, irreps_in2)

    torch.testing.assert_close(tp_pt2(x, y), tp(x, y))