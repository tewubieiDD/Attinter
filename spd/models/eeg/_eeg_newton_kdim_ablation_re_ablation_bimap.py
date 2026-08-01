from spd.base_module_newton_re_ablation_bimap import E2R, Submanifold, AttentionManifold
from spd.models.eeg._eeg_newton_kdim_ablation_re_ablation_common import make_eeg_kdim_classes


BciNewtonKDimNet, MamemNewtonKDimNet, ChaNewtonKDimNet = make_eeg_kdim_classes(
    E2R,
    Submanifold,
    AttentionManifold,
)
