# Torch copy of https://github.com/e3nn/e3nn-jax/blob/c1a1adda485b8de756df56c656ce1d0cece73b64/e3nn_jax/_src/scatter.py
import warnings
from typing import Optional, Union

import torch
import torch.utils._pytree as pytree

def _max(x: Union[int, torch.Tensor], y: Union[int, torch.Tensor]):
    if isinstance(x, int) and isinstance(y, int):
        return max(x, y)
    elif isinstance(x, torch.Tensor):
        return torch.maximum(x, torch.Tensor([y]))
    elif isinstance(y, torch.Tensor):
        return torch.maximum(torch.Tensor([x]), y)
    else:
        return torch.maximum(x, y)

def _distinct_but_small(x: torch.Tensor) -> torch.Tensor:
    """Maps the input to the integers 0, 1, 2, ..., n-1, where n is the number of distinct elements in x.

    Args:
        x (`torch.Tensor`): array of integers

    Returns:
        `torch.Tensor`: array of integers of same size
    """
    shape = x.shape
    x = torch.ravel(x)
    unique = torch.unique(x, size=x.shape[0])  # Pigeonhole principle
    x = torch.Tensorcan(
        lambda _, i: (None, torch.where(i == unique, size=1)[0][0]), None, x
    )[1]
    return torch.reshape(x, shape)


def scatter_sum(
    data: torch.Tensor,
    *,
    dst: Optional[torch.Tensor] = None,
    nel: Optional[torch.Tensor] = None,
    output_size: Optional[int] = None,
    map_back: bool = False,
) -> torch.Tensor:
    r"""Scatter sum of data.

    Performs either of the following two operations::
        output[dst[i]] += data[i]

    or::

        output[i] = sum(data[sum(nel[:i]):sum(nel[:i+1])])

    Args:
        data (`torch.Tensor` or `IrrepsArray`): array of shape ``(n1,..nd, ...)``
        dst (optional, `torch.Tensor`): array of shape ``(n1,..nd)``. If not specified, ``nel`` must be specified.
        nel (optional, `torch.Tensor`): array of shape ``(output_size,)``. If not specified, ``dst`` must be specified.
        output_size (optional, int): size of output array.
            If not specified, ``nel`` must be specified or ``map_back`` must be ``True``.
        map_back (bool): whether to map back to the input position

    Returns:
        `torch.Tensor` or `IrrepsArray`: output array of shape ``(output_size, ...)``
    """
    return _scatter_op(
        "sum",
        0.0,
        data,
        dst=dst,
        nel=nel,
        output_size=output_size,
        map_back=map_back,
    )


def scatter_mean(
    data: torch.Tensor,
    *,
    dst: Optional[torch.Tensor] = None,
    nel: Optional[torch.Tensor] = None,
    output_size: Optional[int] = None,
    map_back: bool = False,
) -> torch.Tensor:
    r"""Scatter mean of data.

    Performs either of the following two operations::

        n[dst[i]] += 1
        output[dst[i]] += data[i] / n[i]

    or::

        output[i] = sum(data[sum(nel[:i]):sum(nel[:i+1])]) / nel[i]

    Args:
        data (`torch.Tensor` or `IrrepsArray`): array of shape ``(n1,..nd, ...)``
        dst (optional, `torch.Tensor`): array of shape ``(n1,..nd)``. If not specified, ``nel`` must be specified.
        nel (optional, `torch.Tensor`): array of shape ``(output_size,)``. If not specified, ``dst`` must be specified.
        output_size (optional, int): size of output array.
            If not specified, ``nel`` must be specified or ``map_back`` must be ``True``.
        map_back (bool): whether to map back to the input position

    Returns:
        `torch.Tensor` or `IrrepsArray`: output array of shape ``(output_size, ...)``
    """
    if map_back and nel is not None:
        assert dst is None
        assert output_size is None

        total = _scatter_op(
            "sum",
            0.0,
            data,
            nel=nel,
            map_back=False,
        )
        den = _max(1, den)

        for _ in range(total.ndim - nel.ndim):
            den = den[..., None]

        output = total / den.astype(total.dtype)
        output = torch.Tensormap(
            lambda x: torch.repeat(x, nel, axis=0, total_repeat_length=data.shape[0]),
            output,
        )
        return output

    total = _scatter_op(
        "sum",
        0.0,
        data,
        dst=dst,
        nel=nel,
        output_size=output_size,
        map_back=map_back,
    )

    if dst is not None or map_back:
        if dst is not None:
            ones = torch.ones(data.shape[: dst.ndim], dtype=torch.int32)
        if nel is not None:
            ones = torch.ones(data.shape[:1], dtype=torch.int32)

        nel = _scatter_op(
            "sum",
            0.0,
            ones,
            dst=dst,
            nel=nel,
            output_size=output_size,
            map_back=map_back,
        )

    nel = _max(1, nel)

    for _ in range(total.ndim - nel.ndim):
        nel = nel[..., None]

    return total / nel.to(total.dtype)


def scatter_max(
    data: torch.Tensor,
    *,
    dst: Optional[torch.Tensor] = None,
    nel: Optional[torch.Tensor] = None,
    initial: float =  torch.inf,
    output_size: Optional[int] = None,
    map_back: bool = False,
) -> torch.Tensor:
    r"""Scatter max of data.

    Performs either of the following two operations::

        output[i] = max(initial, *(x for j, x in zip(dst, data) if j == i))

    or::

        output[i] = max(initial, *data[sum(nel[:i]):sum(nel[:i+1])])

    Args:
        data (`torch.Tensor` or `IrrepsArray`): array of shape ``(n, ...)``
        dst (optional, `torch.Tensor`): array of shape ``(n,)``. If not specified, ``nel`` must be specified.
        nel (optional, `torch.Tensor`): array of shape ``(output_size,)``. If not specified, ``dst`` must be specified.
        initial (float): initial value to compare to
        output_size (optional, int): size of output array. If not specified, ``nel`` must be specified
            or ``map_back`` must be ``True``.
        map_back (bool): whether to map back to the input position

    Returns:
        `torch.Tensor` or `IrrepsArray`: output array of shape ``(output_size, ...)``
    """
    return _scatter_op(
        "max",
        initial,
        data,
        dst=dst,
        nel=nel,
        output_size=output_size,
        map_back=map_back,
    )


def _scatter_op(
    op: str,
    initial: float,
    data: torch.Tensor,
    *,
    dst: Optional[torch.Tensor] = None,
    nel: Optional[torch.Tensor] = None,
    output_size: Optional[int] = None,
    map_back: bool = False,
) -> torch.Tensor:
    if dst is None and nel is None:
        raise ValueError("Either dst or nel must be specified")
    if dst is not None and nel is not None:
        raise ValueError("Only one of dst or nel must be specified")

    if nel is not None:
        if output_size is not None:
            raise ValueError("output_size must not be specified if nel is specified")
        output_size = nel.shape[0]
        num_elements = data.shape[0]
        dst = torch.repeat(torch.arange(output_size), nel, total_repeat_length=num_elements)
        if map_back:
            output_size = None

    if not (dst.shape == data.shape[: dst.ndim]):
        raise ValueError(
            (
                f"trying to do e3nn.scatter_{op} with dst.shape={dst.shape} and data.shape={data.shape}"
                f" but dst.shape must be equal to data.shape[: dst.ndim]"
            )
        )

    if output_size is None and map_back is False:
        raise ValueError("output_size must be specified if map_back is False")
    if output_size is not None and map_back is True:
        raise ValueError("output_size must not be specified if map_back is True")

    if output_size is None and map_back is True:
        output_size = dst.size
        dst = _distinct_but_small(dst)

    def _op(x, dst):
        z = initial * torch.ones((output_size,) + x.shape[dst.ndim :], dtype=x.dtype)
        dst = dst.to(torch.int) # Explicitly cast before indexing
        if op == "sum":
            z[(dst,)] += x
        elif op == "max":
            z[(dst,)] = max(z[(dst,), x])
        return z
    output = pytree.tree_map(_op, data, dst)

    if map_back:
        output = output[(dst,)]

    return output


def index_add(
    indices: torch.Tensor = None,
    input: torch.Tensor = None,
    *,
    n_elements: torch.Tensor = None,
    out_dim: int = None,
    map_back: bool = False,
) -> torch.Tensor:
    warnings.warn(
        "e3nn.index_add is deprecated, use e3nn.scatter_sum instead", DeprecationWarning
    )
    return scatter_sum(
        input, dst=indices, nel=n_elements, output_size=out_dim, map_back=map_back
    )