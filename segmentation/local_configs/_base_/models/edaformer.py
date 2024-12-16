# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained="EFT_T.pth",
    backbone=dict(
        type='EFT_T',
        style='pytorch',
        sr_ratios=[1,1,1,1]),
    decode_head=dict(
        type='EDAFormerHead',
        in_channels=[64, 128, 256],
        in_index=[1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=128),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))