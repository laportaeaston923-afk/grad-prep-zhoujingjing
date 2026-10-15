"""Controlled one-step treatment: iMF JVP tangent = u_theta(z,t,t)."""
config = dict(
    name='mnist_imf_one_step', dataset='mnist', data_root='datasets',
    n_steps=10000, batch_size=32, num_workers=0, image_size=32,
    # More conservative, keep identical across both variants.\n    lr=1e-4, weight_decay=0.0, grad_clip=5.0, mixed_precision='no',
    scheduler=dict(warmup_steps=500, min_lr_ratio=0.1),
    model=dict(input_size=32, patch_size=4, in_channels=1,
               dim=256, depth=6, num_heads=8, num_classes=10, max_cfg=7.5),
    meanflow=dict(mode='i-meanflow', normalizer=['minmax', None, None],
                  cfg_scale=None),
    # Basic L2 as in paper Figure 3; no adaptive loss or CFG.\n    comparison_revision=2,  # FP32-JVP audited experiment; not compatible with v1 weights\n    comparison=dict(variant='imf', equal_time_prob=0.5,
                    normalizer=['minmax', None, None],
                    time_dist=['lognorm', -0.4, 1.0],
                    adaptive_l2_gamma=1.0, adaptive_l2_c=1e-2),
)
