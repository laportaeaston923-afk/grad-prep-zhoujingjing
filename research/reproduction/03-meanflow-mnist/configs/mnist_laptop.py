config = dict(
    name='mnist_laptop_imf', dataset='mnist', data_root='datasets',
    n_steps=10000, batch_size=32, num_workers=0, image_size=32,
    lr=1e-4, weight_decay=0.0, grad_clip=5.0, mixed_precision='bf16',
    scheduler=dict(warmup_steps=500, min_lr_ratio=0.1),
    model=dict(input_size=32, patch_size=4, in_channels=1,
               dim=256, depth=6, num_heads=8, num_classes=10, max_cfg=7.5),
    meanflow=dict(mode='i-meanflow', normalizer=['minmax', None, None],
                  time_dist=['lognorm', -0.4, 1.0], cfg_ratio=0.10,
                  cfg_scale=None, cfg_scale_ratio=0.75,
                  adaptive_l2_gamma=0.0, adaptive_l2_c=1e-2),
)
