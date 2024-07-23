from e3nn import o3
from torch import nn
from typing import Callable
from ._activation import soft_odd
import torch.nn.functional as F

class Gate(nn.Module):
    '''
    The input is split into scalars that are activated separately, scalars that are used as gates, and non-scalars that are
    multiplied by the gates.
    
    List of assumptions:
    
    - The gate scalars are on the right side of the scalars.

    irreps_in (e3nn.o3.Irreps): Input irreps
    even_act (Callable[[float], float]): Activation function for even scalars. Default: :func:`jax.nn.gelu`.
    odd_act (Callable[[float], float]): Activation function for odd scalars. Default: :math:`(1 - \exp(-x^2)) x`.
    even_gate_act (Callable[[float], float]): Activation function for even gate scalars. Default: :func:`jax.nn.sigmoid`.
    odd_gate_act (Callable[[float], float]): Activation function for odd gate scalars. Default: :func:`jax.nn.tanh`.
    '''
    

    def __init__(self,
                 irreps_in: o3.Irreps,
                 even_act: Callable[[float], float] = F.gelu,
                 odd_act: Callable[[float], float] = soft_odd,
                 even_gate_act: Callable[[float], float] = F.sigmoid,
                 odd_gate_act: Callable[[float], float] = F.tanh):
        
        
        