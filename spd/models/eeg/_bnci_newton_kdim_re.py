import torch
from torch import nn

from spd.base_module_newton_re import E2R, AttentionManifold, Submanifold
from spd.modules import LogEig, ReEig


def _tri_dim(n):
    return n * (n + 1) // 2


class _SubmanifoldAttentionMixin:
    def _init_submanifold_blocks(
        self,
        n,
        k_dims,
        slice,
        num_classes,
        hidden_dim=256,
        context_dim=128,
    ):
        self.n = n
        self.k_dims = list(k_dims)
        self.subcov = Submanifold(
            self.n,
            self.k_dims,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
        )
        self.att_dims = self.k_dims + [self.n]
        self.attentions = nn.ModuleList([AttentionManifold(d, d) for d in self.att_dims])
        self.re = ReEig(threshold=1e-4)

        self.tangent = LogEig()
        self.flat = nn.Flatten()
        feature_dim = sum(_tri_dim(d) for d in self.att_dims)
        self.linear = nn.Linear(feature_dim * slice, num_classes, bias=True)

    def _attended_features(self, x):
        sub_x_list = [self.subcov(x[:, i, :, :]) for i in range(x.size(1))]
        grouped_sub_x = zip(*sub_x_list)

        features = []
        for idx, covs_k in enumerate(grouped_sub_x):
            d_tensor = torch.cat(covs_k, dim=1)
            d_feat = self.attentions[idx](d_tensor)
            features.append(self.tangent(self.re(d_feat)))

        main_feat = self.tangent(self.re(self.attentions[-1](x)))
        features.append(main_feat)
        return torch.cat(features, dim=-1)


class BNCINewtonKDimNet(nn.Module, _SubmanifoldAttentionMixin):
    def __init__(
        self,
        slice,
        k_dims,
        num_channels,
        num_classes,
        manifold_dim=40,
        temporal_kernel=63,
        submanifold_hidden_dim=256,
        submanifold_context_dim=128,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, manifold_dim, (num_channels, 1), bias=False)
        self.Bn1 = nn.BatchNorm2d(manifold_dim)
        self.conv2 = nn.Conv2d(
            manifold_dim,
            manifold_dim,
            (1, temporal_kernel),
            padding="same",
            bias=False,
        )
        self.Bn2 = nn.BatchNorm2d(manifold_dim)

        self.ract1 = E2R(slice)
        self._init_submanifold_blocks(
            n=manifold_dim,
            k_dims=k_dims,
            slice=slice,
            num_classes=num_classes,
            hidden_dim=submanifold_hidden_dim,
            context_dim=submanifold_context_dim,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)
        x = self._attended_features(x)
        x = self.flat(x)
        return self.linear(x)


class BNCI2014001NewtonKDimNet(BNCINewtonKDimNet):
    def __init__(self, slice, k_dims):
        super().__init__(
            slice=slice,
            k_dims=k_dims,
            num_channels=22,
            num_classes=4,
        )


class BNCI2015001NewtonKDimNet(BNCINewtonKDimNet):
    def __init__(self, slice, k_dims):
        super().__init__(
            slice=slice,
            k_dims=k_dims,
            num_channels=13,
            num_classes=2,
        )
