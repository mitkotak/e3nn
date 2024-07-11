import torch

from e3nn.util._scatter import scatter_sum, scatter_mean, scatter_max


def test_scatter_sum():
    i = torch.Tensor([0, 2, 2, 0])
    x = torch.Tensor([1.0, 2.0, 3.0, -10.0])
    torch.testing.assert_allclose(
        scatter_sum(x, dst=i, output_size=3),
        torch.Tensor([-9.0, 0.0, 5.0]),
    )

    torch.testing.assert_allclose(  # map_back
        scatter_sum(x, dst=i, map_back=True),
        torch.Tensor([-9.0, 5.0, 5.0, -9.0]),
    )


    x = torch.Tensor([1.0, 2.0, 1.0, 0.5, 0.5, 0.7, 0.2, 0.1])
    nel = torch.Tensor([3, 2, 3])
    torch.testing.assert_allclose(  # nel
        scatter_sum(x, nel=nel),
        torch.Tensor([4.0, 1.0, 1.0]),
    )

    torch.testing.assert_allclose(  # nel + map_back
        scatter_sum(x, nel=nel, map_back=True),
        torch.Tensor([4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )

    i = torch.Tensor([[0, 2], [2, 0]])
    x = torch.Tensor([[[1.0, 0.0], [2.0, 1.0]], [[3.0, 0.0], [-10.0, -1.0]]])
    torch.testing.assert_allclose(
        scatter_sum(x, dst=i, output_size=3),
        torch.Tensor([[-9.0, -1.0], [0.0, 0.0], [5.0, 1.0]]),
    )


def test_scatter_mean():
    x = torch.Tensor([[2.0, 3.0], [0.0, 3.0], [-10.0, 42.0]])
    dst = torch.Tensor([[0, 2], [2, 2], [0, 1]])

    torch.testing.assert_allclose(  # dst
        scatter_mean(x, dst=dst, output_size=3),
        torch.Tensor([-4.0, 42.0, 2.0]),
    )

    torch.testing.assert_allclose(  # map_back
        scatter_mean(x, dst=dst, map_back=True),
        torch.Tensor([[-4.0, 2.0], [2.0, 2.0], [-4.0, 42.0]]),
    )

    x = torch.Tensor([10.0, 1.0, 2.0, 3.0])
    nel = torch.Tensor([1, 0, 3])
    torch.testing.assert_allclose(  # nel
        scatter_mean(x, nel=nel),
        torch.Tensor([10.0, 0.0, 2.0]),
    )

    torch.testing.assert_allclose(  # nel + map_back
        scatter_mean(x, nel=nel, map_back=True),
        torch.Tensor([10.0, 2.0, 2.0, 2.0]),
    )

def test_scatter_max():
    i = torch.Tensor([0, 2, 2, 0])
    x = torch.Tensor([1.0, 2.0, 3.0, -10.0])
    torch.testing.assert_allclose(
        scatter_max(x, dst=i, output_size=3),
        torch.Tensor([1.0,    torch.inf, 3.0]),
    )

    torch.testing.assert_allclose(  # map_back
        scatter_max(x, dst=i, map_back=True),
        torch.Tensor([1.0, 3.0, 3.0, 1.0]),
    )

    torch.testing.assert_allclose(  # nel
        scatter_max(
            torch.Tensor([-1.0, -2.0, -1.0, 0.5, 0.5, 0.7, 0.2, 0.1]),
            nel=torch.Tensor([3, 2, 3]),
        ),
        torch.Tensor([-1.0, 0.5, 0.7]),
    )