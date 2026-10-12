"""Evaluation adapter for the uploaded step_0050000.pt, not a training recipe.

Architecture is inferred from tensor shapes; max_cfg=7.5 comes from the
user's existing EXPERIMENT_RECORD.md. The raw checkpoint contains no training
mode, optimizer, normalization metadata, or verified cumulative step.
"""
config = dict(
    dataset='mnist', data_root='datasets', image_size=32,
    model=dict(input_size=32, patch_size=2, in_channels=1, dim=256,
               depth=6, num_heads=4, mlp_ratio=4.0, num_classes=10, max_cfg=7.5),
    meanflow=dict(mode='i-meanflow', normalizer=['minmax', None, None],
                  cfg_scale=[1.0, 7.5]),
    evaluation_notes=dict(
        architecture_source='inferred from uploaded tensor shapes; strict load succeeded',
        max_cfg_source='existing user EXPERIMENT_RECORD.md: max_cfg=7.5',
        normalizer_source='reference MNIST implementation minmax normalization',
        training_mode='unknown; i-meanflow is an unused sampler constructor setting',
        training_steps='50000 suggested by filename; not verified by checkpoint metadata',
    ),
)
