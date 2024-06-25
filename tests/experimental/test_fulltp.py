import torch
from e3nn import o3, o2, experimental
import pytest


@pytest.mark.parametrize("irreps_in1", ["0e", "0e + 1e"])
@pytest.mark.parametrize("irreps_in2", ["2x0e", "2x0e + 3x1e"])
def test_fulltpo3(irreps_in1, irreps_in2):
    x = o3.Irreps(irreps_in1).randn(10, -1)
    y = o3.Irreps(irreps_in2).randn(10, -1)

    tp_pt2 = torch.compile(experimental.FullTensorProductO3(irreps_in1, irreps_in2), fullgraph=True)
    tp = o3.FullTensorProduct(irreps_in1, irreps_in2)

    torch.testing.assert_close(tp_pt2(x, y), tp(x, y))


@pytest.mark.parametrize("irreps_in1", ["0e", "0e + 1"])
@pytest.mark.parametrize("irreps_in2", ["2x0e", "2x0e + 3x1"])
def test_fulltpo2(irreps_in1, irreps_in2):
    x = o2.Irreps(irreps_in1).randn(10, -1)
    y = o2.Irreps(irreps_in2).randn(10, -1)

    tp_pt2 = torch.compile(experimental.FullTensorProductO2(irreps_in1, irreps_in2), fullgraph=True)
    tp = o2.FullTensorProduct(irreps_in1, irreps_in2)

    torch.testing.assert_close(tp_pt2(x, y), tp(x, y))
