from spd.base_module_newton import (
    signal2spd,
    E2R,
    AttentionManifold,
    SubmanifoldAttention,
    NewtonSchulzOrthogonalization,
    NewtonHyperNet as _NewtonHyperNet,
    Submanifold as _Submanifold,
)


class NewtonHyperNet(_NewtonHyperNet):
    def __init__(
            self,
            n,
            k_dims,
            hidden_dim=128,
            context_dim=64,
            dropout=0.0,
            base_init="branch",
            use_tanh=False,
            use_scale=False,
            dynamic_scale=0.1,
    ):
        super().__init__(
            n,
            k_dims,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            dropout=dropout,
            base_init=base_init,
            use_tanh=use_tanh,
            use_scale=use_scale,
            dynamic_scale=dynamic_scale,
        )


class Submanifold(_Submanifold):
    def __init__(
            self,
            n,
            k_dims,
            hidden_dim=128,
            context_dim=64,
            dropout=0.0,
            num_iterations=8,
            base_init="branch",
            use_tanh=False,
            use_scale=False,
            dynamic_scale=0.1,
    ):
        super().__init__(
            n,
            k_dims,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            dropout=dropout,
            num_iterations=num_iterations,
            base_init=base_init,
            use_tanh=use_tanh,
            use_scale=use_scale,
            dynamic_scale=dynamic_scale,
        )
