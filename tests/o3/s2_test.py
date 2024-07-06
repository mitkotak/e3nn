import torch

import pytest
from e3nn.o3 import ToS2Grid, FromS2Grid, Irreps
from e3nn.util.test import assert_equivariant


@pytest.mark.parametrize("res_a", [11, 12, 13, 14, 15, 16, None])
@pytest.mark.parametrize("res_b", [12, 14, 16, None])
@pytest.mark.parametrize("irreps", ["0e + 1o", "1o + 2e", Irreps.spherical_harmonics(4)])
def test_inverse1(irreps, res_b, res_a) -> None:
    m = FromS2Grid(coeffs=irreps, res=(res_b, res_a))
    k = ToS2Grid(coeffs=irreps, res=(res_b, res_a))

    res_b, res_a = m.res_beta, m.res_alpha
    x = torch.randn(res_b, res_a)
    x = k(m(x))  # remove high frequencies

    y = k(m(x))
    torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("res_a", [11, 12, 13, 14, 15, 16, None])
@pytest.mark.parametrize("res_b", [12, 14, 16, None])
@pytest.mark.parametrize("irreps", ["0e + 1o", "1o + 2e", Irreps.spherical_harmonics(4)])
def test_inverse2(irreps, res_b, res_a) -> None:
    m = FromS2Grid(coeffs=irreps, res=(res_b, res_a))
    k = ToS2Grid(coeffs=irreps, res=(res_b, res_a))
    lmax = m.lmax

    x = torch.randn((lmax + 1) ** 2)

    y = m(k(x))
    torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("res_a", [100, 101])
@pytest.mark.parametrize("res_b", [98, 100])
@pytest.mark.parametrize("lmax", [1, 5])
def test_equivariance(lmax, res_b, res_a) -> None:
    m = FromS2Grid(coeffs=Irreps.spherical_harmonics(lmax), res=(res_b, res_a))
    k = ToS2Grid(coeffs=Irreps.spherical_harmonics(lmax), res=(res_b, res_a))

    def f(x):
        y = k(x)
        y = y.exp()
        return m(y)

    f.irreps_in = f.irreps_out = Irreps.spherical_harmonics(lmax)

    assert_equivariant(f)
