import torch
from torch import nn

from spd.base_module_newton_re_v1_1 import E2R, Submanifold, AttentionManifold
from spd.modules import ReEig, LogEig


def _tri_dim(n):
    return n * (n + 1) // 2


class _SubmanifoldAttentionMixin:
    def _init_submanifold_blocks(self, n, k_dims, slice, num_classes):
        self.n = n
        self.k_dims = list(k_dims)
        self.subcov = Submanifold(self.n, self.k_dims)
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


class BciNewtonKDimNet(nn.Module, _SubmanifoldAttentionMixin):
    def __init__(self, slice, k_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2 = nn.BatchNorm2d(20)

        self.ract1 = E2R(slice)
        self._init_submanifold_blocks(n=20, k_dims=k_dims, slice=slice, num_classes=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)
        x = self._attended_features(x)
        x = self.flat(x)
        return self.linear(x)


class MamemNewtonKDimNet(nn.Module, _SubmanifoldAttentionMixin):
    def __init__(self, slice, k_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 125, (8, 1))
        self.Bn1 = nn.BatchNorm2d(125)
        self.conv2 = nn.Conv2d(125, 15, (1, 36), padding=(0, 18))
        self.Bn2 = nn.BatchNorm2d(15)

        self.ract1 = E2R(slice)
        self._init_submanifold_blocks(n=15, k_dims=k_dims, slice=slice, num_classes=5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)
        x = self._attended_features(x)
        x = self.flat(x)
        return self.linear(x)


class ChaNewtonKDimNet(nn.Module, _SubmanifoldAttentionMixin):
    def __init__(self, slice, k_dims):
        super().__init__()
        self.slice = slice
        dim1 = 22
        self.conv1 = nn.Conv2d(self.slice, dim1 * self.slice, (56, 1), groups=self.slice)
        self.Bn1 = nn.BatchNorm2d(dim1 * self.slice)
        self.conv2 = nn.Conv2d(
            dim1 * self.slice,
            16 * self.slice,
            (1, 64),
            padding=(0, 32),
            groups=self.slice,
        )
        self.Bn2 = nn.BatchNorm2d(16 * self.slice)

        self.ract1 = E2R(slice, dim=1)
        self._init_submanifold_blocks(n=16, k_dims=k_dims, slice=slice, num_classes=2)

    def forward(self, x):
        x = self.conv1(x.repeat(1, self.slice, 1, 1))
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)
        x = self._attended_features(x)
        x = self.flat(x)
        return self.linear(x)
