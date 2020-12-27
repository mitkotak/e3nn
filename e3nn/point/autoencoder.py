# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable, abstract-method
import math
import collections
import networkx as nx
import torch
import torch_geometric as tg
from torch_geometric.nn import nearest, radius_graph
from torch_scatter import scatter_mean, scatter_std, scatter_add, scatter_max
from torch_cluster import fps
import torch_sparse

from e3nn import rs
from e3nn.tensor import SphericalTensor


class Autoencoder(torch.nn.Module):
    def __init__(self, pool, unpool, n_layers, bloom_lmax):
        super().__init__()
        self.pool = pool
        self.unpool = unpool
        self.n_layers = n_layers
        self.bloom_lmax = bloom_lmax
        # min_radius should be included here instead of in forward

    def encode(self, x, pos, edge_index, edge_attr, batch, n_norm):
        self.xs = [x]
        self.positions = [pos]
        self.map_indices = []
        self.edge_indices = [edge_index]
        self.edge_attrs = [edge_attr]
        self.batches = [batch]
        self.sphs = []
        for _ in range(self.n_layers):
            x, pos, edge_index, edge_attr, batch = self.pool(
                x, pos, edge_index, edge_attr, batch=batch, n_norm=n_norm
            )
            self.map_indices.append(self.pool.map_index)  # used for unpooling
            self.map_attrs.append(self.pool.map_attr)
            self.sphs.append(self.pool.sph)  # used for loss on bloom for clustering
            self.xs.append(x)
            self.positions.append(self.pos)
            self.edge_indices.append(edge_index)
            self.edge_attrs.append(edge_attr)
            self.batches.append(batch)
        return x, pos, edge_index, edge_attr, batch

    def decode(self, x, pos, edge_index, edge_attr, batch, n_norm):
        for _ in range(self.n_layers):
            x, pos, edge_index, edge_attr, batch = self.unpool(
                x, pos, edge_index, edge_attr, batch=batch, n_norm=n_norm
            )
            self.xs.append(x)
            self.positions.append(pos)
            self.edge_indices.append(edge_index)
            self.batches.append(batch)
        return x, pos, edge_index, edge_attr, batch

    def encode_decode(self, x, pos, edge_index, edge_attr, batch, n_norm):
        # Likely will want to save various properties of intermediate representations
        x, pos, edge_index, edge_attr, batch = self.encode(
            x, pos, edge_index, edge_attr, batch=batch, n_norm=n_norm
        )
        x, pos, edge_index, edge_attr, batch = self.decode(
            x, pos, edge_index, edge_attr, batch=batch, n_norm=n_norm
        )
        return x, pos, edge_index, edge_attr, batch

    @classmethod
    def signals_and_centers(cls, map_index, map_attr, centers_in_sph, lmax, min_radius=0.1):
        """Calculate signals and centers for given clustering

        Args:
            map_index (torch.LongTensor of shape [2, num_map]): Index of edges of [cluster, node].
            map_attr (torch.Torch of shape [num_map, 3]): 3D Cartesian relative distance vector between cluster and node.
            centers_in_sph (bool): Include center in spherical signal
            lmax (int): lmax for SphericalTensor
            min_radius (float, optional): min radius for SphericalTensor vectors.

        Returns:
            torch.Tensor of shape [nodes, (lmax + 1) ** 2]
            torch.Tensor of shape [nodes, 4]
        """
        clusters = map_index[0]
        vecs = -map_attr[[1, 0]]
        greater = vecs.norm(2, -1) > min_radius
        lesser = vecs.norm(2, -1) <= min_radius
        signals = []
        centers = []
        C = clusters.max() + 1
        for i in range(C):
            vec_indices = (clusters == i & greater).nonzero().reshape(-1)
            center_indices = (clusters == i & lesser).nonzero().reshape(-1)
            if vec_indices.shape[0] == 0 and center_indices.shape[0] == 0:
                print("Hmm something's wrong. This cluster has no edges")
            if vec_indices.shape[0] > 0:
                signal = SphericalTensor.from_geometry(vecs[vec_indices], lmax).signal
            else:
                signal = clusters.new_zeros((lmax + 1) ** 2)
            if center_indices > 0:
                if center_indices.shape[0] > 1:
                    print("Warning - more than one node in min_radius.")
                center_index = vecs[center_indices].norm(2, -1).argmax()
                center_pos = vecs[center_index][1, 2, 0]
                L1 = map_attr.new(center_pos)  # Permute x, y, z to y, z, x
                L0 = map_attr.new([1.])
                if centers_in_sph and vec_indices.shape[0] > 0:
                    signal[1:3] = center_pos
                centers.append(torch.cat([L0, L1], dim=0))
            else:
                centers.append(map_attr.new_zeros(4))
            signals.append(signal)
        return torch.stack(signals, dim=0), torch.stack(centers, dim=0)

    @classmethod
    def loss(cls, xs, sphs, unpool_xs, unpool_sphs, unpool_centers, map_indices, map_attrs, lmax, min_radius=0.1):
        match_loss = cls.loss_match_x(xs, unpool_xs)
        dist_loss = cls.loss_distribution_x(xs)
        cluster_loss = cls.loss_cluster_pool(sphs, map_indices, map_attrs, lmax, min_radius=min_radius)
        unpool_loss = cls.loss_unpool_bloom(unpool_sphs, unpool_centers, map_indices, map_attrs, lmax, min_radius=min_radius)
        return match_loss, dist_loss, cluster_loss, unpool_loss

    @classmethod
    def loss_match_x(cls, xs, unpool_xs):
        return ((xs[:-1]-unpool_xs.flip(0) ** 2).sum(0)

    @classmethod
    def loss_distribution_x(cls, xs):
        pass

    @classmethod
    def loss_cluster_pool(cls, sphs, map_indices, map_attrs, lmax, min_radius=0.1):
        cluster_sphs = []
        for map_index, map_attr in map_indices, map_attrs:
            cluster_sph, _ = cls.signals_and_centers(map_index, map_attr, True, lmax, min_radius=min_radius)
            cluster_sphs.append(cluster_sph)
        return ((cluster_sphs - sphs) ** 2).sum(0)

    @classmethod
    def loss_unpool_bloom(cls, unpool_sphs, unpool_centers, map_indices, map_attrs, lmax, min_radius=0.1):
        cluster_sphs, cluster_centers = [], []
        for map_index, map_attr in map_indices, map_attrs:
            cluster_sph, cluster_center = cls.signals_and_centers(map_index, map_attr, False, lmax, min_radius=min_radius)
            cluster_sphs.append(cluster_sph), cluster_centers.append(cluster_center)
        return (cluster_sph - unpool_sphs.flip(0)) ** 2 + (cluster_centers - unpool_centers.flip(0)) ** 2

    def decode_forward(self, n_norm, min_radius):
        """To be used for teacher forcing training"""
        for i in range(self.n_layers):
            i += 1
            x, pos, edge_index, edge_attr, batch, new_pos, new_pos, map_index, map_attr = (
                self.xs[-i], self.positions[-i], self.edge_indices[-i], self.edge_attrs[-i], self.batch[-i],
                self.positions[-(i+1)], self.map_indices[-i], self.map_attr[-i],
            )
            # get predictions
            x, pos, edge_index, edge_attr, batch = self.unpool.forward(
                x, pos, edge_index, edge_attr, batch,
                new_pos, map_index, map_attr,
                n_norm=n_norm, min_radius=min_radius
            )
            self.unpool_xs.append(x)
            self.unpool_edge_indices.append(edge_index)
            self.unpool_edge_attrs.append(edge_attr)
            self.unpool_sph.append(self.unpool.sph)
            self.unpool_centers.append(self.unpool.centers)    

    def forward(self, x, pos, edge_index, edge_attr, batch, n_norm=1, min_radius=0.1):
        x, pos, edge_index, edge_attr, batch = self.encode(
            x, pos, edge_index, edge_attr, batch=batch, n_norm=n_norm, min_radius=min_radius
        )
        return self.decode_forward(min_radius)

class Pooling(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, bloom_lmax, bloom_conv_module, bloom_module, cluster_module, gather_conv_module):
        """Pool to smaller graph with new features using learnable clustering.

        Args:
            Rs_in: Representation list of input.
            Rs_out: Representation list of output.
            bloom_lmax (int): Max L of SphericalTensor used by bloom module.
            bloom_conv_module (torch.nn.Module): Module to apply convolutions to produce SphericalTensors and features.
            bloom_module (torch.nn.Module): Module used to produce new points from SphericalTensors from bloom_conv_module.
            cluster_module (torch.nn.Module): Module use to cluster new points from bloom_module.
            gather_conv_module (torch.nn.Module): Single convolution used to gather features to new points.
        """
        super().__init__()
        self.Rs_bloom = [(1, L, (-1)**L) for L in range(bloom_lmax + 1)]
        self.Rs_inter = self.Rs_bloom + Rs_out
        self.layers = torch.nn.ModuleDict()
        self.layers['conv'] = bloom_conv_module(Rs_in=Rs_in, Rs_out=self.Rs_inter)
        self.layers['bloom'] = bloom_module
        self.layers['cluster']= cluster_module
        self.layers['gather'] = gather_conv_module(Rs_in=Rs_out, Rs_out=Rs_out)

    @classmethod
    def get_new_edge_index(cls, N, edge_index, map_batch, cluster):
        """Get new edge_index for pooled geometry based on original edge_index

        Args:
            N (int): number of original nodes
            edge_index (torch.LongTensor of shape [2, num_edges]): original edge_index
            bloom_batch (torch.LongTensor of shape [num_bloom_nodes]): mapping of bloomed nodes to original nodes
            cluster (torch.LongTensor of shape [num_bloom_nodes]): mapping of bloomed nodes to new nodes

        Returns:
            torch.LongTensor of shape [2, num_new_edges]: new edge_index
        """
        B, C = len(map_batch), max(cluster + 1)
        map_batch_index = torch.stack([map_batch, torch.arange(len(map_batch))], dim=0)
        cluster_index = torch.stack([torch.arange(len(map_batch)), cluster], dim=0)
        E, F, G = edge_index.shape[-1], map_batch.shape[-1], cluster_index.shape[-1]
        convert_edge_index, vals = torch_sparse.spspmm(
            edge_index, edge_index.new_ones(E, dtype=torch.float32),
            map_batch_index, edge_index.new_ones(F, dtype=torch.float32),
            N, N, B, coalesced=True
        )
        convert_edge_index, vals = torch_sparse.spspmm(
            convert_edge_index, edge_index.new_ones(len(vals), dtype=torch.float32),
            cluster_index, edge_index.new_ones(G, dtype=torch.float32),
            N, B, C, coalesced=True
        )
        new_edge_index, vals = torch_sparse.spspmm(
            convert_edge_index[[1, 0], :], edge_index.new_ones(len(vals), dtype=torch.float32),
            convert_edge_index, edge_index.new_ones(len(vals), dtype=torch.float32),
            C, N, C, coalesced=True
        )
        return new_edge_index

    def new_points(self, x, pos, edge_index, edge_attr, batch, n_norm=1, min_radius=0.1):
        """Generate pooled points from conv, bloom, and cluster modules.

        Args:
            x (torch.Tensor of shape [num_nodes, num_features]): features on nodes.
            pos (torch.Tensor of shape [num_pos, 3]): 3D Cartesian positions of nodes.
            edge_index (torch.LongTensor of shape [2, num_edges]): Node edges of [center, neighbor].
            edge_attr (torch.Tensor of shape [num_edges, 3]): 3D Cartesian relative distance vectors between center and neighbor.
            batch (torch.LongTensor): Example index per node.
            n_norm (int, optional): Number of average neighbors to normalize convolution. Defaults to 1.
            min_radius (float, optional): Minimum radius for bloom module. Defaults to 0.1.

        Returns:
            torch.Tensor of shape [num_pos, num_features]: new features
            torch.Tensor of shape [num_new_pos, 3]: 3D Cartesian positions of new nodes.
            torch.LongTensor of shape [2, num_cluster_edges]: Edges between new_pos (clusters) and pos
            torch.LongTensor of shape [num_bloom]: clusters each bloom_batch belongs to.
            torch.Tensor of shape [num_bloom, 3]: 3D Cartesian relative distance between pos and new_pos if pos in new_pos cluster.
        """
        out = self.layers['conv'](x, edge_index, edge_attr, n_norm=n_norm)
        self.sph, x = out[..., :rs.dim(self.Rs_bloom)], out[..., rs.dim(self.Rs_bloom):]
        # bloom_batch has pos index that bloomed point comes from
        bloom_pos, bloom_batch = self.layers['bloom'](self.sph, pos, min_radius)
        # Cluster bloom_pos based on centroids sampled from pos
        clusters = self.layers['cluster'](bloom_pos, batch[bloom_batch], start_pos=pos, start_batch=batch)
        map_index = torch.stack([clusters, bloom_batch], dim=0)  # [cluster, pos]
        map_index = torch_sparse.coalesce(  # Remove duplicate edges
            map_index, pos.new_ones(map_index.shape[-1]),
            max(clusters) + 1, max(batch) + 1)[0]
        # new points are the mean of pos in cluster, a single pos can be in several clusters
        new_pos = scatter_mean(pos[map_index[1]], map_index[0], dim=0)
        # relative distance vector between new_pos and pos of cluster
        map_attr = pos[map_index[1]] - new_pos[map_index[0]]
        return x, new_pos, map_index, map_attr

    def new_points_features(self, x, new_pos, map_index, map_attr, edge_index, batch, n_norm):
        """From new positions and cluster_index, create new features.

        Args:
            x (torch.Tensor of shape [num_old_nodes, num_features]): features of old nodes from bloom.
            new_pos (torch.Tensor of shape [num_new_nodes, 3]): 3D Cartesian position of new nodes.
            map_index (torch.LongTensor of shape [2, num_bloom]): Index that ties new to old nodes.
            map_attr (torch.LongTensor of shape [num_bloom, 3]): 3D Cartesian relative distance vector between new and old nodes.
            edge_index (torch.LongTensor of shape [2, num_edges]): Edge index of original graph.
            batch (torch.LongTensor of shape [num_old_nodes]): Index of example per old node.
            n_norm (int): Number of average neighbors to normalize convolution. Defaults to 1.

        Returns:
            torch.Tensor of shape [num_new_nodes, features]: Features of new nodes.
            torch.Tensor of shape [num_new_nodes, 3]: 3D Cartesian positions of new nodes.
            torch.LongTensor of shape [2, num_edges]: Edges of new graph
            torch.Tensor of shape [num_edges, 3]: 3D Cartesian relative distance vectors of edges.
            torch.LongTensor of shape [num_new_nodes]: Index of example per new node.
        """
        N, C = batch.shape[0], (map_index[0].max() + 1)
        x = self.layers['gather'](x, map_index, map_attr, n_norm=n_norm, size=(N, C))
        new_edge_index = self.get_new_edge_index(N, edge_index, batch[map_index[1]], map_index[0])
        new_edge_attr = new_pos[new_edge_index[1]] - new_pos[new_edge_index[0]]
        new_batch = scatter_max(batch[map_index[1]], map_index[0])[0]  # Just grab indices
        return x, new_pos, new_edge_index, new_edge_attr, new_batch

    def forward(self, x, pos, edge_index, edge_attr, min_radius=0.1, batch=None, n_norm=1):
        x, new_pos, self.map_index, self.map_attr = self.new_points(
            x, pos, edge_index, edge_attr, batch, n_norm=1, min_radius=min_radius
        )
        return self.new_points_features(
            x, new_pos, self.map_index, self.map_attr, edge_index, batch, n_norm
        )


class Unpooling(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, bloom_lmax, bloom_conv_module, bloom_module, gather_conv_module):
        """Unpool to larger graph with new features.

        Args:
            Rs_in: Representation list of input.
            Rs_out: Representation list of output.
            bloom_lmax (int): Max L of SphericalTensor used by bloom module.
            bloom_conv_module (torch.nn.Module): Module to apply convolutions to produce SphericalTensors and features.
            bloom_module (torch.nn.Module): Module used to produce new points from SphericalTensors from bloom_conv_module.
            gather_conv_module (torch.nn.Module): Single convolution used to gather features to new points.
        """
        super().__init__()
        self.Rs_bloom = [(1, L, (-1)**L) for L in range(bloom_lmax + 1)]
        self.Rs_centers = [(1, 0, 1), (1, 1, -1)]
        self.Rs_inter = self.Rs_bloom + self.Rs_centers + Rs_out
        self.layers = torch.nn.ModuleDict()
        self.layers['conv'] = bloom_conv_module(Rs_in=Rs_in, Rs_out=self.Rs_inter)
        self.layers['bloom'] = bloom_module
        self.layers['gather'] = gather_conv_module(Rs_in=Rs_out, Rs_out=Rs_out)

    @classmethod
    def merge_clusters(cls, pos, r, batch):
        # edges to merge
        edge_index = radius_graph(pos, r, batch=batch, loop=False)
        G = nx.Graph()
        G.add_nodes_from(range(pos.shape[0]))

        for i, (u, v) in enumerate(edge_index.T.tolist()):
            if v > u:
                continue
            G.add_edge(int(u), int(v))
        pos_map = []
        node_groups = [list(G.subgraph(c).nodes) for c in nx.connected_components(G)]
        for i, g in enumerate(node_groups):
            pos_map_index = torch.stack([
                torch.LongTensor([i] * len(g)),
                torch.LongTensor(g)
            ], dim=0)
            pos_map.append(pos_map_index)
        return torch.cat(pos_map, dim=-1)  # [2, N_old]

    def new_points(self, x, pos, edge_index, edge_attr, batch, n_norm=1, min_radius=0.1):
        out = self.layers['conv'](x, edge_index, edge_attr, n_norm=n_norm)
        self.sph, self.centers, x = (
            out[..., :rs.dim(self.Rs_bloom)],
            out[..., rs.dim(self.Rs_bloom):rs.dim(self.Rs_bloom) + rs.dim(self.Rs_centers)],
            out[..., rs.dim(self.Rs_bloom) + rs.dim(self.Rs_centers):]  # get x to pass on to gather
        )
        center_keepindices = (self.centers[:, 0] > 0.5).nonzero().reshape(-1)
        center_keepdisplace = self.centers[:, 1:][center_keepindices]
        center_pos = pos[center_keepindices] + center_keepdisplace[:, [2, 0, 1]]  # add displacement in yzx and change xyz
        # Get new points
        bloom_pos, bloom_batch = self.layers['bloom'](self.sph, pos, min_radius)
        pos_map = self.merge_clusters(bloom_pos, min_radius, batch[bloom_batch])  # Merge points
        new_pos = scatter_mean(bloom_pos, pos_map[0], dim=0)
        C = new_pos.shape[0]
        new_pos = torch.cat([new_pos, center_pos], dim=0)
        # Combine new positions and centers
        center_indices = torch.arange(center_pos.shape[0]) + C
        map_index = torch.stack([pos_map[0], bloom_batch], dim=0)  # [target, source]
        center_map_index = torch.stack([center_indices, center_keepindices], dim=0)
        map_index = torch.cat([map_index, center_map_index], dim=-1)
        map_attr = pos[bloom_batch] - new_pos[pos_map[0]]
        map_attr = torch.cat([map_attr, -center_keepdisplace], dim=0)  # Cat and invert center displacements
        return x, new_pos, map_index, map_attr

    def new_points_features(self, x, new_pos, map_index, map_attr, batch, n_norm=1):
        N, C = batch.shape[0], (map_index[0].max() + 1)
        x = self.layers['gather'](x, map_index, map_attr, n_norm=n_norm, size=(N, C))
        # Use bloom max diameter per example to construct radius graph
        map_max = scatter_max(map_attr.norm(2, -1), batch[map_index[1]], dim=0)[0]
        num_batch = scatter_add(x.new_ones(batch.shape), batch, dim=0).long()
        # Create new_edge_index based on max of bloom
        ## Might break in case of structure that is in final form
        new_edge_index = []
        for i, m in enumerate(map_max):
            n_start = int(num_batch[:i].sum())
            n_end = int(num_batch[:i + 1].sum())
            rad_graph = radius_graph(new_pos[n_start: n_end], m, loop=False)
            new_edge_index.append(rad_graph + n_start)
        new_edge_index = torch.cat(new_edge_index, dim=-1)
        new_edge_attr = new_pos[new_edge_index[1]] - new_pos[new_edge_index[0]]
        new_batch = scatter_max(batch[map_index[1]], map_index[0])[0]  # Just grab indices
        return x, new_pos, new_edge_index, new_edge_attr, new_batch

    def unpool(self, x, pos, edge_index, edge_attr, batch, n_norm=1, min_radius=0.1):
        x, new_pos, gather_edge_index, gather_edge_attr = self.new_points(
            x, pos, edge_index, edge_attr, batch, n_norm=n_norm, min_radius=min_radius
        )
        return self.new_points_features(
            x, new_pos, gather_edge_index, gather_edge_attr, batch, n_norm=n_norm)

    def forward(self, x, pos, edge_index, edge_attr, batch, new_pos, map_index, map_attr, n_norm=1, min_radius=0.1):
        """To be used with teacher forcing and a symmetric pooling layer"""
        x, _, _, _, _ = self.new_points(
            x, pos, edge_index, edge_attr, batch, n_norm=n_norm, min_radius=min_radius
        )
        return self.new_points_features(
            x, new_pos, map_index, map_attr, batch, n_norm=n_norm
        )


class Bloom(torch.nn.Module):
    def __init__(self, res=200):
        """Module for generating new point sets from peaks of spherical harmonic projections.

        Args:
            res (int, optional): Resolution of grid to use for peak finding. Defaults to 200.
        """
        super().__init__()
        self.res = res

    def forward(self, signal, pos, min_radius, percentage=False, absolute_min=0.01, use_L1=True):
        """Get peaks of signal

        Args:
            signal (torch.Tensor of shape [num_nodes, (L_{max}+1)^2]): Spherical harmonic projection per node.
            pos (torch.Tensor of shape [num_nodes, 3]): The Cartesian position per node.
            min_radius (float): Minimum radius of peaks to add point.
            percentage (bool, optional): Instead of min_radius, use percentage radius of max peak. Defaults to False.
            absolute_min (float, optional): If percentage, what absolute min radius to use for peaks. Defaults to 0.01.
            use_L1 (bool, optional): If no peaks, displace center by L=1 component. Defaults to True.

        Returns:
            torch.Tensor of shape [num_new_nodes]: 3D Cartesian position per new node.
            torch.LongTensor of shape [num_new_nodes: Index of old node per new node.
        """
        all_peaks = []
        new_indices = []
        signal = signal.detach()
        self.used_radius = None
        for i, sig in enumerate(signal):
            if sig.abs().max(0)[0] > 0.:
                peaks, radii = SphericalTensor(sig).find_peaks(res=self.res)
            else:
                peaks, radii = sig.new_zeros(0, 3), sig.new_zeros(0)
            if percentage:
                self.used_radius = max((min_radius * torch.max(radii)), absolute_min)
                keep_indices = (radii > max((min_radius * torch.max(radii)), absolute_min))
            else:
                self.used_radius = min_radius
                keep_indices = (radii > min_radius).nonzero().reshape(-1)
            if keep_indices.shape[0] == 0:
                if use_L1:
                    all_peaks.append(sig[1:1 + 3].unsqueeze(0))
                else:
                    all_peaks.append(signal.new_zeros(1, 3))
                new_indices.append(signal.new_tensor([i]).long())
            else:
                all_peaks.append(peaks[keep_indices] *
                                 radii[keep_indices].unsqueeze(-1))
                new_indices.append(signal.new_tensor([i] * len(keep_indices)).long())
        all_peaks = torch.cat(all_peaks, dim=0)
        new_indices = torch.cat(new_indices, dim=0)
        return all_peaks + pos[new_indices], new_indices


class KMeans(torch.nn.Module):
    def __init__(self, tol=0.001, max_iter=300, score_norm=1):
        """KMeans clustering.

        Args:
            tol (float, optional): Tolerance for KMeans clustering convergance. Defaults to 0.001.
            max_iter (int, optional): Number of KMeans interactions per rand_iter. Defaults to 300.
            score_norm (int, optional): Norm to use for scoring clusters. Defaults to 1.
        """
        super().__init__()
        self.tol = tol
        self.max_iter = max_iter
        self.score_norm = score_norm

    def score(self, pos, batch, centroids, classification):
        """Score clustering

        Args:
            pos (torch.Tensor of shape [num_nodes, 3]): 3D Cartesian positions of nodes.
            batch (torch.LongTensor of shape [num_nodes]): Index of example per node.
            centroids (torch.Tensor of shape [num_centroids, 3]): 3D Cartesian positions of centroids.
            classification (torch.LongTensor of shape [num_nodes]): Index of cluster per node.

        Returns:
            torch.Tensor of shape [num_examples]: Score per example.
        """
        scores = (pos - centroids[classification]).norm(self.score_norm, -1)
        return scatter_add(scores, batch, dim=0)

    def update_centroids(self, pos, batch, centroids, centroids_batch):
        """Update centroids of KMeans clusters

        Args:
            pos (torch.Tensor of shape [num_nodes, 3]): 3D Cartesian positions of nodes.
            batch (torch.LongTensor of shape [num_nodes]): Index of example per node.
            centroids (torch.Tensor of shape [num_centroids, 3]): 3D Cartesian positions of centroids.
            centroids_batch (torch.LongTensor of shape [num_centroids]): Index of example per centroid.

        Returns:
            torch.Tensor of shape [num_centroids, 3]: 3D Cartesian positions of updated centroids.
            torch.LongTensor of shape [num_nodes]: Index of cluster per node.
        """
        N = pos.shape[0]
        M = centroids.shape[0]
        classification = nearest(pos, centroids, batch, centroids_batch)
        update = scatter_mean(pos, classification, dim=0, dim_size=M)
        mask = scatter_mean(torch.ones(N), classification, dim=0, dim_size=M).unsqueeze(-1)
        new_centroids = update * mask + centroids * (1 - mask)
        return new_centroids, classification

    def forward(self, pos, batch, start_pos=None, start_batch=None, fps_ratio=0.5):
        """[summary]

        Args:
            pos (torch.Tensor of shape [total_nodes, 3]): 3D Cartesian positions of nodes.
            batch (torch.LongTensor of shape [total_nodes]): Index of example per node.
            start_pos (torch.Tensor of shape [total_nodes, 3], optional): 3D Cartesian positions of nodes used for sampling initial centroids. Defaults to None.
            start_batch (torch.LongTensor of shape [total_nodes], optional): Index of example per node for nodes used for sampling inital centroids. Defaults to None.
            fps_ratio (float, optional): Ratio of points to choose for "Farthest Point Sampling". Defaults to 0.5.

        Returns:
            torch.LongTensor of shape [num_nodes]: Index of cluster per node.
            torch.Tensor of shape [num_centroids, 3]: Centroids from KMeans clustering.
            torch.LongTensor of shape [num_centroids]: Index of example per centroids.
        """
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
    "Symmetric KMeans clustering algorithm"
    def __init__(self, tol=0.001, max_iter=300, rand_iter=10, score_norm=1):
        """Symmetric KMeans clustering algorithm

        Args:
            tol (float, optional): Tolerance for KMeans clustering convergance. Defaults to 0.001.
            max_iter (int, optional): Number of KMeans interactions per rand_iter. Defaults to 300.
            rand_iter (int, optional): Number of random iterations to perform KMeans clustering. Defaults to 10.
            score_norm (int, optional): Norm to use for scoring clusters. Defaults to 1.
        """
        super().__init__(tol, max_iter, score_norm=score_norm)
        self.rand_iter = rand_iter

    def cluster_edge_index_by_score(self, scores, classification, num_centroids, batch):
        """Get the edge index for disconnected subgraphs of nodes in the union of best performing clusters.

        Args:
            scores (torch.Tensor of shape [rand_iter, batch_size]): The score of the Kmeans clustering per example per random iteration.
            classification (torch.LongTensor of shape [rand_iter, batch_nodes]): Centroids labels for each point in all batches per random iteration.
            num_centroids (int): The number of centroids used for KMeans.
            batch ([torch.LongTensor of shape [batch_nodes]): The index of example per node.

        Returns:
            torch.LongTensor: Edge index of graph of best scoring disconnected subgraphs of clusters.
        """
        N = batch.shape[0]
        batch_index = torch.stack([batch, torch.arange(N)], dim=0)
        min_scores = scores.min(dim=0)[0]
        best_scores = (scores <= min_scores).nonzero()
        # Sizes of our sparse tensor
        R, B, C = self.rand_iter, max(batch) + 1, num_centroids
        # This is doing the equivalent of
        # torch.einsum('rbn)
        best_classifications = torch_sparse.spspmm(
            best_scores.T, scores.new_ones(best_scores.shape[0]),
            batch_index, scores.new_ones(N),
            R, B, N, coalesced=True
        )[0]  # Only keep edge_index of sparse tensor and not values
        best_clusters = classification[best_classifications[0], best_classifications[1]]
        cluster_index = torch.stack([best_clusters, best_classifications[1]], dim=0)
        cluster_edge_index = torch_sparse.spspmm(
            cluster_index[[1, 0]], scores.new_ones(cluster_index.shape[-1]),
            cluster_index, scores.new_ones(cluster_index.shape[-1]),
            N, C, N, coalesced=True
        )[0]  # Only keep edge_index of sparse tensor and not values
        return cluster_edge_index

    def forward(self, pos, batch, start_pos=None, start_batch=None, fps_ratio=0.5):
        """[summary]

        Args:
            pos (torch.Tensor of shape [total_nodes, 3]): 3D Cartesian positions of nodes.
            batch (torch.LongTensor of shape [total_nodes]): Index of example per node.
            start_pos (torch.Tensor of shape [total_nodes, 3], optional): 3D Cartesian positions of nodes used for sampling initial centroids. Defaults to None.
            start_batch (torch.LongTensor of shape [total_nodes], optional): Index of example per node for nodes used for sampling inital centroids. Defaults to None.
            fps_ratio (float, optional): Ratio of points to choose for "Farthest Point Sampling". Defaults to 0.5.

        Returns:
            cluster_batch: The index of cluster per node in pos.
        """
        N = pos.shape[0]
        # Use giant batch for iterations of KMeans
        big_pos = torch.cat(self.rand_iter * [pos], dim=0)
        big_batch = (batch[None].repeat(self.rand_iter, 1) + torch.arange(self.rand_iter)[..., None] * (batch.max() + 1)).reshape(-1)
        if start_batch is not None:
            big_start_pos = torch.cat(self.rand_iter * [start_pos], dim=0)
            big_start_batch = (start_batch[None].repeat(self.rand_iter, 1) + torch.arange(self.rand_iter)[..., None] * (start_batch.max() + 1)).reshape(-1)
        else:
            big_start_pos = None
            big_start_batch = None
        classification, centroids, _ = super().forward(
            big_pos, big_batch, start_pos=big_start_pos, start_batch=big_start_batch, fps_ratio=fps_ratio)
        scores = self.score(big_pos, big_batch, centroids, classification)
        scores = scores.reshape(self.rand_iter, -1)
        classification = classification.reshape(self.rand_iter, -1)
        num_centroids = centroids.shape[0]
        cluster_edge_index = self.cluster_edge_index_by_score(
            scores, classification, num_centroids, batch)  # Don't need to reshape centroids

        # Get connected components
        G = nx.Graph()
        N = int(pos.shape[0])
        [G.add_node(i) for i in range(N)]
        for i, j in cluster_edge_index.T:
            if i < j:  # NetworkX graphs only need single edge, leave out self edges
                G.add_edge(int(i), int(j))
        node_groups = [list(G.subgraph(c).nodes) for c in nx.connected_components(G)]
        labels = pos.new_zeros(pos.shape[0])
        for i in range(len(node_groups)):
            labels[node_groups[i]] = i
        return labels.to(torch.int64)
