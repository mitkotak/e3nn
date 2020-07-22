# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch
import torch_geometric as tg


class Convolution(tg.nn.MessagePassing):
    def __init__(self, kernel, kernel_groups=1):
        super(Convolution, self).__init__(aggr='add', flow='target_to_source')
        self.kernels = [kernel for i in range(kernel_groups)]

    def forward(self, features, edge_index, edge_r, size=None, n_norm=1):
        """
        :param features: Tensor of shape [n_target, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_messages]
                           edge_index[0] = sources (convolution centers)
                           edge_index[1] = targets (neighbors)
        :param edge_r: Tensor of shape [num_messages, 3]
                       edge_r = position_target - position_source
        :param size: (n_source, n_target) or None
        :param n_norm: typical number of targets per source

        :return: Tensor of shape [n_source, dim(Rs_out)]
        """
        k = torch.stack([ker(edge_r) for ker in self.kernels], dim=1)
        k.div_(n_norm ** 0.5)
        return self.propagate(edge_index, size=size, x=features, k=k)

    def message(self, x_j, k):
        N = x_j.shape[0]
        groups = len(self.kernels)
        cout, cin = k.shape[-2:]
        x_j = x_j.view(N, groups, cin)  # Rs_tp1
        if k.shape[0] == 0:  # https://github.com/pytorch/pytorch/issues/37628
            return torch.zeros(0, groups * cout)
        return torch.einsum('ekij,ekj->eki', k, x_j).reshape(N, groups * cout)
