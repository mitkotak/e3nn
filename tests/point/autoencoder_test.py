# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools
from functools import partial

import pytest
import torch

from e3nn import o3, rs
from e3nn.kernel import Kernel, GroupKernel
from e3nn.point.message_passing import Convolution
from e3nn.radial import ConstantRadialModel
from e3nn.tensor import SphericalTensor
from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.radial import GaussianRadialModel
from e3nn.kernel import Kernel
from e3nn.networks import GatedConvParityNetwork
from e3nn.point.data_helpers import DataNeighbors
from e3nn.point.autoencoder import KMeans, SymmetricKMeans, Bloom, Unpooling, Pooling
from functools import partial
torch.set_default_dtype(torch.float64)

def test_KMeans():
    pos = torch.tensor([[0., 0., 0.], [0.2, 0., 0.], [1.0, 0., 0.], [1.2, 0., 0.]])
    centers = torch.tensor([[0.1, 0., 0.], [1.1, 0., 0.]])
    pos = torch.cat([pos, pos], dim=0)
    batch = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])
    kmeans = KMeans()
    classification, centroids, centroids_batch = kmeans.forward(pos, batch, fps_ratio=0.5)
    # Check centroids_batch
    assert torch.allclose(centroids_batch, torch.LongTensor([0, 0, 1, 1]))
    # Check groupings
    assert torch.allclose(classification[torch.LongTensor([0, 2, 4, 6])], classification[torch.LongTensor([1, 3, 5, 7])])
    # Check cluster centers
    assert torch.allclose(torch.sort(centroids, dim=0)[0], torch.tensor([[0.1, 0., 0.], [0.1, 0., 0.], [1.1, 0., 0.], [1.1, 0., 0.]]))


def test_SymmetricKMeans():
    kmeans = SymmetricKMeans(rand_iter=10)

    batch = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])

    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])  # Square
    pos = torch.cat([pos, pos], dim=0)
    labels = kmeans.forward(pos, batch)
    truth = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).to(torch.int64)
    print(labels, truth)
    assert torch.allclose(labels, truth)

    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [2., 1., 0.]])  # Zig
    pos = torch.cat([pos, pos], dim=0)
    labels = kmeans.forward(pos, batch)
    truth = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).to(torch.int64)
    print(labels, truth)
    assert torch.allclose(labels, truth)


def test_SymmetryKMeans_cluster_edge_index_by_score():
    kmeans = SymmetricKMeans(rand_iter=2)
    batch = torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
    classification = torch.LongTensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [3, 3, 4, 4, 5, 5, 6, 6, 7],
        [8, 8, 8, 9, 9, 9, 10, 11, 12]
    ])
    scores = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.5],
        [0.0, 1.0]
    ])
    num_centroids = 13

    test_cluster_edge_index = torch.LongTensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8, 6, 7, 8]
    ])

    cluster_edge_index = kmeans.cluster_edge_index_by_score(
        scores, classification, num_centroids, batch)
    assert torch.allclose(cluster_edge_index, test_cluster_edge_index)


def test_Pooling_get_new_edge_index():
    N, B, C = 3, 6, 2
    edge_index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    bloom_batch = torch.LongTensor([0, 0, 1, 1, 2, 2])
    cluster = torch.LongTensor([0, 0, 0, 1, 1, 1])
    assert(B == len(bloom_batch))
    assert(C == max(cluster + 1))

    new_edge_index = Pooling.get_new_edge_index(N, edge_index, bloom_batch, cluster)
    assert torch.allclose(new_edge_index, torch.LongTensor([[0, 0, 1, 1], [0, 1, 0, 1]]))

    N, B, C = 4, 6, 2
    edge_index = torch.LongTensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
                                   [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]])
    bloom_batch = torch.LongTensor([0, 1, 1, 2, 2, 3])
    cluster = torch.LongTensor([0, 0, 1, 0, 1, 1])
    assert(B == len(bloom_batch))
    assert(C == max(cluster + 1))

    new_edge_index = Pooling.get_new_edge_index(N, edge_index, bloom_batch, cluster)
    assert torch.allclose(new_edge_index, torch.LongTensor([[0, 0, 1, 1], [0, 1, 0, 1]]))


def test_Bloom_no_peaks():
    bloom = Bloom()
    N = 2
    signal = torch.zeros(N, 25)
    signal[0, 1:4] = torch.tensor([0.01, 0.02, 0.03])
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    min_radius = 0.1
    bloom_pos, bloom_batch = bloom(signal, pos, min_radius, use_L1=True)
    assert torch.allclose(bloom_pos, torch.Tensor([[0.0100, 0.0200, 0.0300], [1.0000, 0.0000, 0.0000]]))
    assert torch.allclose(bloom_batch, torch.LongTensor([0, 1]))


def test_Bloom_tetra():
    import numpy as np
    # From @bdice
    def sort_rounded_xyz_array(arr, decimals=4):
        """The order of points returned is not always well-defined, such as in
        Voronoi or UnitCell creation. Instead of testing a fixed array, arrays must
        be sorted by their rounded representations in order to compare their
        values.
        """
        arr = np.asarray(arr)
        arr = arr.round(decimals)
        indices = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
        return arr[indices]

    tetra = torch.tensor([[0., 0., 0.], [1., 1., 0], [1., 0., 1.], [0., 1., 1]])
    tetra -= tetra.mean(dim=0, keepdims=True)
    tetra_sph = SphericalTensor.from_geometry(tetra, 5).signal

    bloom = Bloom(res=300)
    signal = tetra_sph.unsqueeze(0)
    pos = torch.tensor([[0., 0., 0.]])
    min_radius = 0.1

    bloom_pos, _ = bloom(signal, pos, min_radius, use_L1=True)
    assert np.allclose(sort_rounded_xyz_array(bloom_pos), sort_rounded_xyz_array(tetra), rtol=1e-2)


def test_Pooling():
    bloom_lmax = 4
    rmax = 3.
    number_of_basis = 5
    conv_kwargs = dict(
        mul=4, number_of_basis=number_of_basis, lmax=bloom_lmax, max_radius=rmax, convolution=Convolution, layers=3,
    )
    Rs_in = [(1, 0, 1)]
    Rs_out = [(4, 0, 1), (4, 0, -1), (4, 1, 1), (4, 1, -1), (4, 2, 1), (4, 2, -1)]

    RadialModel = partial(GaussianRadialModel, max_radius=rmax, min_radius=0.,
                          number_of_basis=number_of_basis, h=100,
                          L=bloom_lmax, act=swish)

    K = partial(Kernel, RadialModel=RadialModel, selection_rule=partial(
        o3.selection_rule_in_out_sh, lmax=bloom_lmax))

    single_conv = lambda Rs_in, Rs_out: Convolution(K(Rs_in, Rs_out))

    conv = partial(GatedConvParityNetwork, **conv_kwargs)
    bloom = Bloom(res=300)
    cluster = SymmetricKMeans(rand_iter=20)

    pool = Pooling(Rs_in, Rs_out, bloom_lmax, conv, bloom, cluster, single_conv)

    shape = torch.tensor([(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]).double()  # zigzag
    x = torch.ones(4, 1).double()
    data = DataNeighbors(x, shape, rmax, self_interaction=False)

    # x, new_pos, new_edge_index, new_edge_attr, new_batch
    _ = pool.forward(
        data.x, data.pos, data.edge_index,
        data.edge_attr, batch=torch.zeros(4).long(),
        n_norm=5  # Bloom is sensitive to this
    )


def test_Unpooling_merge_clusters():
    rmin = 0.1
    pos = torch.Tensor([
        [0., 0., 0.],
        [1., 0., 0.],
        [1.05, 0., 0.]
    ])
    N = pos.shape[0]
    batch = torch.zeros(N).long()
    pos_map = Unpooling.merge_clusters(pos, r=rmin, batch=batch)
    assert torch.allclose(
        pos_map,
        torch.LongTensor([[0, 1, 1], [0, 1, 2]]))


def test_Unpooling():
    bloom_lmax = 4
    rmax = 3.
    number_of_basis = 5
    conv_kwargs = dict(
        mul=4, number_of_basis=number_of_basis, lmax=bloom_lmax, max_radius=rmax, convolution=Convolution, layers=3,
    )
    Rs_in = [(1, 0, 1)]
    Rs_out = [(4, 0, 1), (4, 0, -1), (4, 1, 1), (4, 1, -1), (4, 2, 1), (4, 2, -1)]

    RadialModel = partial(GaussianRadialModel, max_radius=rmax, min_radius=0.,
                          number_of_basis=number_of_basis, h=100,
                          L=bloom_lmax, act=swish)

    K = partial(Kernel, RadialModel=RadialModel, selection_rule=partial(
        o3.selection_rule_in_out_sh, lmax=bloom_lmax))

    single_conv = lambda Rs_in, Rs_out: Convolution(K(Rs_in, Rs_out))

    conv = partial(GatedConvParityNetwork, **conv_kwargs)
    bloom = Bloom(res=300)

    unpool = Unpooling(Rs_in, Rs_out, bloom_lmax, conv, bloom, single_conv)
    shape = torch.tensor([(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]).double()  # zigzag
    x = torch.ones(4, 1).double()
    data = DataNeighbors(x, shape, rmax, self_interaction=False)
    # x, new_pos, new_edge_index, new_edge_attr, new_batch 
    _ = unpool.unpool(data.x, data.pos, data.edge_index, data.edge_attr, batch=torch.zeros(4).long(), n_norm=2)
