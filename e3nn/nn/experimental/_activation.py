from typing import Callable, List, Optional
import torch
from torch import nn
import torch.nn.functional as F

from e3nn import o3
import numpy as np

def soft_odd(x):
    """Smooth odd function that can be used as activation function for odd scalars.

    .. math::

        x (1 - e^{-x^2})

    Note:
        Odd scalars (l=0 and p=-1) has to be activated by functions with well defined parity:

        * even (:math:`f(-x)=f(x)`)
        * odd (:math:`f(-x)=-f(x)`).
    """
    return (1 - torch.exp(-(x**2))) * x

def normalspace(n: int) -> torch.Tensor:
    r"""Sequence of normally distributed numbers :math:`x_i` for :math:`i=1, \ldots, n` such that

    .. math::

        \int_{-\infty}^{x_i} \phi(x) dx = \frac{i}{n+1}

    where :math:`\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}` is the normal distribution.

    Args:
        n (int): Number of points

    Returns:
        jax.Array: Sequence of normally distributed numbers

    Examples:
        >>> normalspace(5)
        Array([-0.96742135, -0.4307273 ,  0.        ,  0.43072742,  0.96742165],      dtype=float32)
    """
    return np.sqrt(2) * torch.erfinv(torch.linspace(-1.0, 1.0, n + 2)[1:-1])


def normalize_function(phi: Callable[[float], float]) -> Callable[[float], float]:
    r"""Normalize a function, :math:`\psi(x)=\phi(x)/c` where :math:`c` is the normalization constant such that

    .. math::

        \int_{-\infty}^{\infty} \psi(x)^2 \frac{e^{-x^2/2}}{\sqrt{2\pi}} dx = 1
    """
    
    x = normalspace(1_000_001)
    c = torch.mean(phi(x) ** 2) ** 0.5
    c = c.item()

    if np.allclose(c, 1.0):
        return phi
    else:

        def rho(x):
            return phi(x) / c

        return rho

def parity_function(phi: Callable[[float], float]) -> int:
        x = torch.linspace(0.0, 10.0, 256)

        a1, a2 = phi(x), phi(-x)
        if torch.max(torch.abs(a1 - a2)) < 1e-5:
            return 1
        elif torch.max(torch.abs(a1 + a2)) < 1e-5:
            return -1
        else:
            return 0

def is_zero_in_zero(phi: Callable[[float], float]) -> bool:
    return torch.allclose(phi(torch.Tensor([0.0])), 0.0)
        
class ScalarActivation(nn.Module):
    
    def __init__(self, 
                 irreps_in: o3.Irreps,
                 acts: List[Optional[Callable[[float], float]]] = None,
                 *,
                 even_act: Callable[[float], float] = F.gelu,
                 odd_act: Callable[[float], float] = soft_odd,
                 normalize_act: bool = True):
        r"""Apply activation functions to the scalars of `o3.Irreps`.
        The activation functions are by default normalized.

        Args:
            input (IrrepsArray): input irreps
            acts (optional, list of functions): list of activation functions, one for each chunk of the input
            even_act (Callable[[float], float]): Activation function for even scalars. Default: :func:`F.gelu`.
            odd_act (Callable[[float], float]): Activation function for odd scalars. Default: :math:`(1 - \exp(-x^2)) x`.
            normalize_act (bool): if True, normalize the activation functions using `normalize_function`

        Note:
            The parity of the output depends on the parity of the activation function.
        """

        super(ScalarActivation, self).__init__()

        if acts is None:
            acts = [
                {1: even_act, -1: odd_act}[ir.p] if ir.l == 0 else None
                for _, ir in irreps_in
            ]

        assert len(irreps_in) == len(acts), (irreps_in, acts)
        irreps_out = []
        paths = {}

        for (mul, (l_in, p_in)), slice_x, act in zip(irreps_in, irreps_in.slices(), acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(
                        f"Activation: cannot apply an activation function to a non-scalar input. {irreps_in} {acts}"
                    )

                if normalize_act:
                    act = normalize_function(act)

                p_out = parity_function(act) if p_in == -1 else p_in
                if p_out == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
                    )

                irreps_out.append((mul, (0, p_out)))
            else:
                irreps_out.append((mul, (l_in, p_in)))
                
            paths[l_in] = (slice_x, act)

        self._same_acts = False
        # for performance, if all the activation functions are the same, we can apply it to the contiguous array as well:
        if acts and acts.count(acts[0]) == len(acts):
            if acts[0] is None:
                self.act = None
            else:
                act = acts[0]
                if normalize_act:
                    self.act = normalize_function(act)
 
        irreps_out = o3.Irreps(irreps_out)
        self.irreps_out, _, self.inv = irreps_out.sort()
        self.paths = paths

    def forward(self, input: torch.Tensor):
        
        if self._same_acts:
            if self.act is None:
                return input
            else:
                return self.act(input)
    
        chunks = []
        for (slice_x, act) in self.paths.values():
            if act is None:
                chunks.append(input[..., slice_x])
            else:
                chunks.append(act(input[..., slice_x]))

        return torch.cat([chunks[i] for i in self.inv], dim=-1)

class NormActivation(nn.Module):
    def __init__(self,
                 irreps_in: o3.Irreps,
                 acts: List[Optional[Callable[[float], float]]],
                 *,
                 normalization: str = "component",):
        
        super(NormActivation, self).__init__()
        
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        paths = {}
        for (mul, (l_in, p_in)), slice_x, act in zip(irreps_in, irreps_in.slices(), acts):
            if act is None:
                continue
            paths[l_in] = (slice_x, act)
  
    def forward(self, input: torch.Tensor):
        chunks = []
        for slice_x, act in self.paths.values():
            x = input[..., slice_x]
            n2 = torch.sum(x**2, axis=-1, keepdims=True)
            if self.normalization == "component":
                n2 = n2 / x.shape[-1]
            n = torch.where(n2 > 0.0, torch.sqrt(torch.where(n2 > 0.0, n2, 1.0)), 1.0)
            x = x * act(n)

            chunks.append(x)
        return torch.cat(chunks, dim=-1)
