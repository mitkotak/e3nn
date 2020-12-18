# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable, abstract-method
import math
import collections
import networkx as nx
import torch
import torch_geometric as tg
from torch_geometric.nn import nearest
from torch_scatter import scatter_mean, scatter_std, scatter_add
from torch_cluster import fps
import torch_sparse

from e3nn import rsh, rs
from e3nn.tensor_product import WeightedTensorProduct, GroupedWeightedTensorProduct
from e3nn.linear import Linear
from e3nn.tensor import SphericalTensor


class Convolution(tg.nn.MessagePassing):
    def __init__(self, kernel):
        super(Convolution, self).__init__(aggr='add', flow='target_to_source')
        self.kernel = kernel

    def forward(self, features, edge_index, edge_r, size=None, n_norm=1, groups=1):
        """
        :param features: Tensor of shape [n_target, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_messages]
                           edge_index[0] = sources (convolution centers)
                           edge_index[1] = targets (neighbors)
        :param edge_r: Tensor of shape [num_messages, 3]
                       edge_r = position_target - position_source
        :param size: (n_target, n_source) or None
        :param n_norm: typical number of targets per source

        :return: Tensor of shape [n_source, dim(Rs_out)]
        """
        k = self.kernel(edge_r)
        k.div_(n_norm ** 0.5)
        return self.propagate(edge_index, size=size, x=features, k=k, groups=groups)

    def message(self, x_j, k, groups):
        N = x_j.shape[0]
        cout, cin = k.shape[-2:]
        x_j = x_j.view(N, groups, cin)  # Rs_tp1
        if k.shape[0] == 0:  # https://github.com/pytorch/pytorch/issues/37628
            return torch.zeros(0, groups * cout)
        if k.dim() == 4 and k.shape[1] == groups:  # kernel has group dimension
            return torch.einsum('egij,egj->egi', k, x_j).reshape(N, groups * cout)
        return torch.einsum('eij,egj->egi', k, x_j).reshape(N, groups * cout)


class WTPConv(tg.nn.MessagePassing):
    def __init__(self, Rs_in, Rs_out, Rs_sh, RadialModel, normalization='component'):
        """
        :param Rs_in:  input representation
        :param lmax:   spherical harmonic representation
        :param Rs_out: output representation
        :param RadialModel: model constructor
        """
        super().__init__(aggr='add', flow='target_to_source')
        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        self.tp = WeightedTensorProduct(Rs_in, Rs_sh, Rs_out, normalization, own_weight=False)
        self.rm = RadialModel(self.tp.nweight)
        self.Rs_sh = Rs_sh
        self.normalization = normalization

    def forward(self, features, edge_index, edge_r, sh=None, size=None, n_norm=1):
        """
        :param features: Tensor of shape [n_target, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_messages]
                           edge_index[0] = sources (convolution centers)
                           edge_index[1] = targets (neighbors)
        :param edge_r: Tensor of shape [num_messages, 3]
                       edge_r = position_target - position_source
        :param sh: Tensor of shape [num_messages, dim(Rs_sh)]
        :param size: (n_target, n_source) or None
        :param n_norm: typical number of targets per source

        :return: Tensor of shape [n_source, dim(Rs_out)]
        """
        if sh is None:
            sh = rsh.spherical_harmonics_xyz(self.Rs_sh, edge_r, self.normalization)  # [num_messages, dim(Rs_sh)]
        sh = sh / n_norm**0.5

        w = self.rm(edge_r.norm(dim=1))  # [num_messages, nweight]

        return self.propagate(edge_index, size=size, x=features, sh=sh, w=w)

    def message(self, x_j, sh, w):
        """
        :param x_j: [num_messages, dim(Rs_in)]
        :param sh:  [num_messages, dim(Rs_sh)]
        :param w:   [num_messages, nweight]
        """
        return self.tp(x_j, sh, w)


class WTPConv2(tg.nn.MessagePassing):
    r"""
    WTPConv with self interaction and grouping

    This class assumes that the input and output atom positions are the same
    """
    def __init__(self, Rs_in, Rs_out, Rs_sh, RadialModel, groups=math.inf, normalization='component'):
        super().__init__(aggr='add', flow='target_to_source')
        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        self.lin1 = Linear(Rs_in, Rs_out, allow_unused_inputs=True, allow_zero_outputs=True)
        self.tp = GroupedWeightedTensorProduct(Rs_in, Rs_sh, Rs_out, groups=groups, normalization=normalization, own_weight=False)
        self.rm = RadialModel(self.tp.nweight)
        self.lin2 = Linear(Rs_out, Rs_out)
        self.Rs_sh = Rs_sh
        self.normalization = normalization

    def forward(self, features, edge_index, edge_r, sh=None, size=None, n_norm=1):
        # features = [num_atoms, dim(Rs_in)]
        if sh is None:
            sh = rsh.spherical_harmonics_xyz(self.Rs_sh, edge_r, self.normalization)  # [num_messages, dim(Rs_sh)]
        sh = sh / n_norm**0.5

        w = self.rm(edge_r.norm(dim=1))  # [num_messages, nweight]

        self_interation = self.lin1(features)
        features = self.propagate(edge_index, size=size, x=features, sh=sh, w=w)
        features = self.lin2(features)
        has_self_interaction = torch.cat([
            features.new_ones(mul * (2 * l + 1)) if any(l_in == l and p_in == p for _, l_in, p_in in self.Rs_in) else features.new_zeros(mul * (2 * l + 1))
            for mul, l, p in self.Rs_out
        ])
        return 0.5**0.5 * self_interation + (1 + (0.5**0.5 - 1) * has_self_interaction) * features

    def message(self, x_j, sh, w):
        return self.tp(x_j, sh, w)


class Pooling(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, bloom_lmax, bloom_conv_module, bloom_module, cluster_module, gather_conv_module):
        """[summary]

        Args:
            bloom_conv_module ([type]): This module produces
            cluster_module ([type]): [description]
            gather_conv_module ([type]): [description]
        """
        super().__init__()
        self.Rs_bloom = [(1, L, (-1)**L) for L in range(bloom_lmax + 1)]
        self.Rs_inter = self.Rs_bloom + Rs_out
        self.layers = torch.nn.ModuleDict()
        self.layers['conv'] = bloom_conv_module(Rs_in, self.Rs_inter)
        self.layers['bloom'] = bloom_module
        self.layers['cluster']= cluster_module(Rs_out, Rs_out)
        self.layers['gather'] = gather_conv_module(Rs_out, Rs_out)

    @classmethod
    def new_edge_index(self, N, edge_index, bloom_batch, cluster):
        """[summary]

        Args:
            N (int): number of original nodes
            edge_index (torch.LongTensor of shape [2, num_edges]): initial edge_index
            bloom_batch (torch.LongTensor of shape [num_bloom_nodes]): mapping of bloomed nodes to original nodes
            cluster (torch.LongTensor of shape [num_bloom_nodes]): mapping of bloomed nodes to new nodes

        Returns:
            [type]: [description]
        """
        B, C = len(bloom_batch), max(cluster + 1)
        bloom_index = torch.stack([bloom_batch, torch.arange(len(bloom_batch))], dim=0)
        cluster_index = torch.stack([torch.arange(len(bloom_batch)), cluster], dim=0)
        E, F, G = edge_index.shape[-1], bloom_index.shape[-1],  cluster_index.shape[-1]
        convert_edge_index, vals = torch_sparse.spspmm(
            edge_index, torch.ones(E),
            bloom_index, torch.ones(F),
            N, N, B
        )
        convert_edge_index, vals = torch_sparse.spspmm(
            convert_edge_index, torch.ones(len(vals)),
            cluster_index, torch.ones(G),
            N, B, C
        )
        new_edge_index, vals = torch_sparse.spspmm(
            convert_edge_index[[1, 0], :], torch.ones(len(vals)),
            convert_edge_index, torch.ones(len(vals)),
            C, N, C
        )
        return new_edge_index

    def forward(self, x, pos, edge_index, edge_attr, batch=None, n_norm=1):
        N = pos.shape[0]
        out = self.layers['conv'](x, edge_index, edge_attr, batch=batch, n_norm=n_norm)
        sph = out[..., :rs.dim(self.Rs_bloom)]
        x = out[..., rs.dim(self.Rs_bloom):]
        bloom_pos, bloom_batch = self.layers['bloom'](sph, pos)
        clusters = self.layers['cluster'](bloom_pos, batch[bloom_batch])
        new_pos = scatter_mean(pos[bloom_batch], clusters, dim=0)
        gather_edge_index = x.new([clusters, bloom_batch]).long()  # [target, source]
        gather_edge_attr = pos[bloom_batch] - new_pos[clusters]
        x = self.layers['gather'](x, gather_edge_index, gather_edge_attr)
        new_edge_index = self.new_edge_index(N, edge_index, bloom_batch, cluster)
        new_edge_attr = new_pos(new_edge_index[1]) - new_pos(new_edge_index[0])
        return x, new_pos, new_edge_index, new_edge_attr


class Unpooling(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, bloom_conv_module, peak_module, gather_conv_module):
        """[summary]

        Args:
            bloom_conv_module ([type]): [description]
            peak_module ([type]): [description]
            gather_conv_module ([type]): [description]
        """
        super().__init__()
        self.bloom_conv_module = bloom_conv_module
        self.peak_module = peak_module
        self.gather_conv_module = gather_conv_module 

    def forward(self):
        pass


class Bloom(torch.nn.Module):
    def __init__(self, res=200):
        super().__init__()
        self.res = res

    def forward(self, signal, x, pos, min_radius, percentage=False, absolute_min=0.01):
        all_peaks = []
        new_indices = []
        self.used_radius = None
        for i, sig in enumerate(signal):
            peaks, radii = SphericalTensor(sig).find_peaks(res=self.res)
            if percentage:
                self.used_radius = max((min_radius * torch.max(radii)), absolute_min)
                keep_indices = (radii > max((min_radius * torch.max(radii)), absolute_min))
            else:
                self.used_radius = min_radius
                keep_indices = (radii > min_radius)
            if len(keep_indices) == 0:
                all_peaks.append(signal.new_zeros(1, 3))
                new_indices.append(signal.new_tensor([i]).long())
            else:
                all_peaks.append(peaks[keep_indices] *
                                 radii[keep_indices].unsqueeze(-1))
                new_indices.append(signal.new_tensor([i] * len(keep_indices)).long())
        all_peaks = torch.stack(all_peaks, dim=0)
        new_indices = torch.stack(new_indices, dim=0)
        return all_peaks + pos[new_indices], new_indices


class KMeans(torch.nn.Module):
    def __init__(self, tol=0.001, max_iter=300, score_norm=1):
        super().__init__()
        self.tol = tol
        self.max_iter = max_iter
        self.score_norm = score_norm

    def score(self, pos, batch, centroids, classification):
        scores = (pos - centroids[classification]).norm(self.score_norm, -1)
        return scatter_add(scores, batch, dim=0)

    def update_centroids(self, pos, batch, centroids, centroids_batch):
        N = pos.shape[0]
        M = centroids.shape[0]
        classification = nearest(pos, centroids, batch, centroids_batch)
        update = scatter_mean(pos, classification, dim=0, dim_size=M)
        mask = scatter_mean(torch.ones(N), classification, dim=0, dim_size=M).unsqueeze(-1)
        new_centroids = update * mask + centroids * (1 - mask)
        return new_centroids, classification

    def forward(self, pos, batch, start_pos=None, start_batch=None, fps_ratio=0.5):
        if start_pos is None:
            start_pos = pos
            start_batch = batch
        fps_indices = fps(start_pos, start_batch, fps_ratio)
        centroids = start_pos[fps_indices]
        centroids_batch = start_batch[fps_indices]
        for _ in range(self.max_iter):
            new_centroids, classification = self.update_centroids(pos, batch, centroids, centroids_batch)
            if ((centroids - new_centroids).norm(2, -1) < self.tol).all():
                return classification, new_centroids, centroids_batch
            centroids = new_centroids
        return classification, centroids, centroids_batch


class SymmetricKMeans(KMeans):
    def __init__(self, tol=0.001, max_iter=300, rand_iter=10, score_norm=1):
        super().__init__(tol, max_iter, score_norm=score_norm)
        self.rand_iter = rand_iter
        self.score_norm = score_norm

    def forward(self, pos, batch, start_pos=None, start_batch=None, fps_ratio=0.5):
        N = pos.shape[0]
        # Use giant batch for iterations of KMeans
        big_pos = torch.cat(self.rand_iter * [pos], dim=0)
        big_batch = (batch.unsqueeze(0).repeat(self.rand_iter, 1) + torch.arange(self.rand_iter).unsqueeze(-1) * (batch.max() + 1)).reshape(-1)
        if start_batch is not None:
            big_start_pos = torch.cat(self.rand_iter * [start_pos], dim=0)
            big_start_batch = (start_batch.unsqueeze(0).repeat(self.rand_iter, 1) + torch.arange(self.rand_iter).unsqueeze(-1) * (start_batch.max() + 1)).reshape(-1)
        else:
            big_start_pos = None
            big_start_batch = None
        classification, centroids, _ = super().forward(big_pos, big_batch, start_pos=big_start_pos, start_batch=big_start_batch, fps_ratio=fps_ratio)
        scores = self.score(big_pos, big_batch, centroids, classification)
        scores = scores.reshape(self.rand_iter, -1).sum(1)
        # sorts = torch.argsort(scores, dim=0)
        cluster_dicts = []
        for index, score in zip(classification.reshape(self.rand_iter, -1), scores):
            d = collections.defaultdict(list)
            if score <= min(scores):  # This is technically too restrictive 
                for i, c in enumerate(index):
                    d[c.item()].append(i)
                cluster_dicts.append(d)

        G = nx.Graph()
        N = pos.shape[0]
        [G.add_node(i) for i in range(N)]
        for cluster_dict in cluster_dicts:
            for cluster in cluster_dict:
                values = cluster_dict[cluster]
                for i in range(len(values) - 1):
                    G.add_edge(values[i], values[i + 1])

        node_groups = [list(G.subgraph(c).nodes) for c in nx.connected_components(G)]
        labels = pos.new_zeros(pos.shape[0])
        for i in range(len(node_groups)):
            labels[node_groups[i]] = i
        return labels.to(torch.int64)
