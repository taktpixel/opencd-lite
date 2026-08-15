"""Forward-pass shape tests for the model implementations."""

from __future__ import annotations

import torch


def test_cgnet_forward_shapes(cgnet_small) -> None:
    x1 = torch.randn(2, 3, 64, 64)
    x2 = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        change_map, final_map = cgnet_small(x1, x2)
    assert change_map.shape == (2, 1, 64, 64)
    assert final_map.shape == (2, 1, 64, 64)


def test_ifn_forward_shapes(ifn_small) -> None:
    x1 = torch.randn(2, 3, 64, 64)
    x2 = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        outputs = ifn_small(x1, x2)
    assert len(outputs) == 5
    # Deep supervision outputs at scales 1/16 ... 1/1.
    expected_sizes = [4, 8, 16, 32, 64]
    for logits, size in zip(outputs, expected_sizes, strict=True):
        assert logits.shape == (2, 1, size, size)


def test_ifn_encoder_is_shared_and_frozen(ifn_small) -> None:
    assert ifn_small.encoder1 is ifn_small.encoder2
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    outputs = ifn_small(x, x.clone())
    # The frozen encoder must not leak gradients into its parameters.
    outputs[-1].sum().backward()
    assert all(p.grad is None for p in ifn_small.encoder1.parameters())


def test_fc_ef_forward_shapes() -> None:
    from opencd_lite import FC_EF

    model = FC_EF(in_channels=6, base_channel=16).eval()
    with torch.inference_mode():
        (out,) = model(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 16, 64, 64)


def test_fc_siam_diff_forward_shapes() -> None:
    from opencd_lite import FC_Siam_diff

    model = FC_Siam_diff(in_channels=3, base_channel=16).eval()
    with torch.inference_mode():
        (out,) = model(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 16, 64, 64)


def test_fc_siam_conc_forward_shapes() -> None:
    from opencd_lite import FC_Siam_conc

    model = FC_Siam_conc(in_channels=3, base_channel=16).eval()
    with torch.inference_mode():
        (out,) = model(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 16, 64, 64)


def test_snunet_forward_shapes() -> None:
    from opencd_lite import SNUNet_ECAM

    model = SNUNet_ECAM(in_channels=3, base_channel=16).eval()
    with torch.inference_mode():
        (out,) = model(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 64, 64, 64)


def test_resnet_v1c_forward_shapes() -> None:
    from opencd_lite.models import ResNetV1c

    # The 3-stage layout used by the BIT config: strides (1, 2, 1),
    # single output at stage 3.
    model = ResNetV1c(
        depth=18,
        num_stages=3,
        out_indices=(2,),
        strides=(1, 2, 1),
        dilations=(1, 1, 1),
        contract_dilation=True,
    ).eval()
    with torch.inference_mode():
        (out,) = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 256, 8, 8)


def test_resnet_default_forward_shapes() -> None:
    from opencd_lite.models import ResNet

    model = ResNet(depth=18).eval()
    with torch.inference_mode():
        outs = model(torch.randn(2, 3, 64, 64))
    assert [tuple(o.shape) for o in outs] == [
        (2, 64, 16, 16),
        (2, 128, 8, 8),
        (2, 256, 4, 4),
        (2, 512, 2, 2),
    ]


def test_resnet_rejects_unsupported_variants() -> None:
    import pytest

    from opencd_lite.models import ResNet

    with pytest.raises(NotImplementedError):
        ResNet(depth=50)
    with pytest.raises(NotImplementedError):
        ResNet(style="caffe")
    with pytest.raises(NotImplementedError):
        ResNet(norm_cfg={"type": "GN"})


def test_feature_fusion_neck_policies() -> None:
    from opencd_lite.models import FeatureFusionNeck

    x1 = (torch.full((1, 2, 4, 4), 3.0), torch.full((1, 4, 2, 2), 5.0))
    x2 = (torch.full((1, 2, 4, 4), 1.0), torch.full((1, 4, 2, 2), 1.0))

    (concat0, concat1) = FeatureFusionNeck(policy="concat", out_indices=(0, 1))(x1, x2)
    assert concat0.shape == (1, 4, 4, 4)
    assert concat1.shape == (1, 8, 2, 2)

    (diff,) = FeatureFusionNeck(policy="diff", out_indices=(0,))(x1, x2)
    assert diff.eq(-2.0).all()
    (abs_diff,) = FeatureFusionNeck(policy="abs_diff", out_indices=(0,))(x1, x2)
    assert abs_diff.eq(2.0).all()
    (total,) = FeatureFusionNeck(policy="sum", out_indices=(1,))(x1, x2)
    assert total.eq(6.0).all()


def test_interaction_layers() -> None:
    from opencd_lite.models import ChannelExchange, SpatialExchange, TwoIdentity

    x1 = torch.arange(2 * 4 * 2 * 4, dtype=torch.float32).reshape(2, 4, 2, 4)
    x2 = -x1

    y1, y2 = TwoIdentity()(x1, x2)
    assert torch.equal(y1, x1) and torch.equal(y2, x2)

    # Channels at even indices are exchanged (p=1/2 -> period 2).
    y1, y2 = ChannelExchange(p=1 / 2)(x1, x2)
    assert torch.equal(y1[:, 0], x2[:, 0]) and torch.equal(y1[:, 1], x1[:, 1])
    assert torch.equal(y2[:, 0], x1[:, 0]) and torch.equal(y2[:, 1], x2[:, 1])

    # Columns at even indices are exchanged.
    y1, y2 = SpatialExchange(p=1 / 2)(x1, x2)
    assert torch.equal(y1[..., 0], x2[..., 0]) and torch.equal(y1[..., 1], x1[..., 1])
    assert torch.equal(y2[..., 0], x1[..., 0]) and torch.equal(y2[..., 1], x2[..., 1])


def test_ia_resnet_forward_shapes() -> None:
    from opencd_lite.models import IA_ResNetV1c

    model = IA_ResNetV1c(
        depth=18,
        interaction_cfg=(
            None,
            {"type": "SpatialExchange", "p": 1 / 2},
            {"type": "ChannelExchange", "p": 1 / 2},
            {"type": "ChannelExchange", "p": 1 / 2},
        ),
    ).eval()
    with torch.inference_mode():
        outs = model(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64))
    # Each stage output concatenates the two temporal feature maps.
    assert [tuple(o.shape) for o in outs] == [
        (2, 128, 16, 16),
        (2, 256, 8, 8),
        (2, 512, 4, 4),
        (2, 1024, 2, 2),
    ]


def test_changer_head_forward_shapes() -> None:
    from opencd_lite.models import Changer

    head = Changer(in_channels=(8, 16), channels=8, num_classes=2).eval()
    # Two stages of concatenated bi-temporal features at 1/4 and 1/8 scale.
    inputs = [torch.randn(2, 16, 16, 16), torch.randn(2, 32, 8, 8)]
    with torch.inference_mode():
        out = head(inputs)
    # The head classifies at the resolution of the first stage.
    assert out.shape == (2, 2, 16, 16)


def test_sta_head_forward_shapes() -> None:
    from opencd_lite.models import STAHead

    head = STAHead(in_channels=(8, 16), channels=8, sa_in_channels=16, sa_mode="PAM").eval()
    inputs = [torch.randn(2, 16, 16, 16), torch.randn(2, 32, 8, 8)]
    with torch.inference_mode():
        out = head(inputs)
    # A single-channel +/-100 pseudo-logit map at the first stage scale.
    assert out.shape == (2, 1, 16, 16)
    assert set(out.unique().tolist()) <= {-100.0, 100.0}
    with torch.inference_mode():
        dist = head.forward_distance(inputs)
    assert dist.shape == (2, 1, 16, 16)
    assert (dist >= 0).all()


def test_sta_head_bam_mode() -> None:
    from opencd_lite.models import STAHead

    head = STAHead(in_channels=(8,), channels=8, sa_in_channels=16, sa_mode="BAM", sa_ds=1).eval()
    with torch.inference_mode():
        out = head([torch.randn(1, 16, 8, 8)])
    assert out.shape == (1, 1, 8, 8)


def test_criss_cross_attention_shapes() -> None:
    from opencd_lite.models.lightcdnet import CrissCrossAttention

    attn = CrissCrossAttention(16).eval()
    x = torch.randn(2, 16, 8, 12)
    with torch.inference_mode():
        out = attn(x)
    assert out.shape == x.shape
    # gamma starts at 0, so the module is initialized as an identity.
    assert torch.equal(out, x)


def test_lightcdnet_forward_shapes() -> None:
    from opencd_lite.models import LightCDNet

    model = LightCDNet(stage_repeat_num=[4, 8, 4], net_type="small").eval()
    with torch.inference_mode():
        outs = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    # Early fused feature at 1/2 plus three downsampled stages.
    assert [tuple(o.shape) for o in outs] == [
        (1, 24, 32, 32),
        (1, 48, 16, 16),
        (1, 96, 8, 8),
        (1, 192, 4, 4),
    ]


def test_tiny_fpn_forward_shapes() -> None:
    from opencd_lite.models import TinyFPN

    neck = TinyFPN(
        in_channels=[24, 48, 96, 192],
        out_channels=48,
        num_outs=4,
        custom_block="conv",
        exist_early_x=True,
        early_x_for_fpn=True,
    ).eval()
    inputs = [
        torch.randn(1, 24, 32, 32),
        torch.randn(1, 48, 16, 16),
        torch.randn(1, 96, 8, 8),
        torch.randn(1, 192, 4, 4),
    ]
    with torch.inference_mode():
        outs = neck(inputs)
    # The early feature is prepended unchanged before the 4 FPN levels.
    assert torch.equal(outs[0], inputs[0])
    assert [tuple(o.shape) for o in outs[1:]] == [
        (1, 48, 32, 32),
        (1, 48, 16, 16),
        (1, 48, 8, 8),
        (1, 48, 4, 4),
    ]


def test_ds_fpn_head_forward_shapes() -> None:
    from opencd_lite.models import DS_FPNHead

    head = DS_FPNHead(in_channels=(8, 8), channels=8, num_classes=2, dropout_ratio=0.0).eval()
    inputs = [
        torch.randn(1, 4, 32, 32),  # early feature, dropped by the head
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 8, 8, 8),
    ]
    with torch.inference_mode():
        out = head(inputs)
    assert out.shape == (1, 2, 16, 16)


def test_mix_vision_transformer_forward_shapes() -> None:
    from opencd_lite.models import MixVisionTransformer

    model = MixVisionTransformer(
        embed_dims=8,
        num_layers=[1, 1, 1, 1],
        num_heads=[1, 2, 2, 4],
        sr_ratios=[8, 4, 2, 1],
    ).eval()
    with torch.inference_mode():
        outs = model(torch.randn(2, 3, 64, 64))
    # Stage widths are embed_dims * num_heads at strides 4/8/16/32.
    assert [tuple(o.shape) for o in outs] == [
        (2, 8, 16, 16),
        (2, 16, 8, 8),
        (2, 16, 4, 4),
        (2, 32, 2, 2),
    ]


def test_segformer_head_forward_shapes() -> None:
    from opencd_lite.models import SegformerHead

    head = SegformerHead(in_channels=(8, 16), channels=8, num_classes=2).eval()
    inputs = [torch.randn(2, 8, 16, 16), torch.randn(2, 16, 8, 8)]
    with torch.inference_mode():
        out = head(inputs)
    assert out.shape == (2, 2, 16, 16)


def test_farseg_fpn_forward_shapes() -> None:
    from opencd_lite.models import FarSegFPN

    neck = FarSegFPN(policy="concat", in_channels=(8, 16), out_channels=8, num_outs=2).eval()
    feats1 = (torch.randn(1, 8, 16, 16), torch.randn(1, 16, 8, 8))
    feats2 = (torch.randn(1, 8, 16, 16), torch.randn(1, 16, 8, 8))
    with torch.inference_mode():
        outs = neck(feats1, feats2)
    # Two fused FPN levels plus the fused global scene embedding.
    assert [tuple(o.shape) for o in outs] == [
        (1, 16, 16, 16),
        (1, 16, 8, 8),
        (1, 32, 1, 1),
    ]


def test_changestar_head_forward_shapes() -> None:
    from opencd_lite.models import ChangeStarHead

    head = ChangeStarHead(
        inference_mode="mean",
        seg_head_cfg={
            "type": "FarSegHead",
            "in_channels": (8, 8, 8, 8, 16),
            "fsr_channels": 8,
            "channels": 8,
        },
        changemixin_cfg={"in_channels": 16, "inner_channels": 8, "num_convs": 1},
        channels=8,
        num_classes=2,
        out_channels=1,
    ).eval()
    inputs = [
        torch.randn(1, 16, 32, 32),
        torch.randn(1, 16, 16, 16),
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 32, 1, 1),
    ]
    with torch.inference_mode():
        out = head(inputs)
    # Single-channel change logits at the finest FPN scale.
    assert out.shape == (1, 1, 32, 32)
    # The inner segmentation classifier is replaced with an identity.
    from torch import nn

    assert isinstance(head.seg_head.conv_seg, nn.Identity)


def test_tinynet_forward_shapes() -> None:
    from opencd_lite.models import TinyNet

    model = TinyNet(
        arch="S",
        widen_factor=0.5,
        output_early_x=True,
        strip_kernel_size=(7, 7, 7, 7),
    ).eval()
    with torch.inference_mode():
        outs = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    # Early concatenated stem feature plus the four trunk stages.
    assert [tuple(o.shape) for o in outs] == [
        (1, 16, 32, 32),
        (1, 8, 32, 32),
        (1, 16, 16, 16),
        (1, 16, 8, 8),
        (1, 24, 4, 4),
    ]


def test_tiny_fpn_tinyblock_forward_shapes() -> None:
    from opencd_lite.models import TinyFPN
    from opencd_lite.models.tinynet import TinyBlock

    neck = TinyFPN(
        in_channels=[8, 16],
        out_channels=8,
        num_outs=2,
        custom_block="tinyblock",
        exist_early_x=True,
    ).eval()
    assert isinstance(neck.fpn_convs[0], TinyBlock)
    inputs = [
        torch.randn(1, 16, 32, 32),  # early feature (not fed to the FPN)
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 16, 8, 8),
    ]
    with torch.inference_mode():
        outs = neck(inputs)
    assert torch.equal(outs[0], inputs[0])
    assert [tuple(o.shape) for o in outs[1:]] == [(1, 8, 16, 16), (1, 8, 8, 8)]


def test_tiny_head_forward_shapes() -> None:
    from opencd_lite.models import TinyHead

    head = TinyHead(
        in_channels=(16, 8, 8),
        feature_strides=(2, 2, 4),
        priori_attn=True,
        channels=8,
        num_classes=2,
        dropout_ratio=0.0,
    ).eval()
    inputs = [
        torch.randn(1, 16, 32, 32),  # early stem feature (attention gate)
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 8, 8, 8),
    ]
    with torch.inference_mode():
        out = head(inputs)
    # The gated output is resized to the early feature's resolution.
    assert out.shape == (1, 2, 32, 32)


def test_bit_head_forward_shapes() -> None:
    from opencd_lite.models import BITHead

    head = BITHead(
        in_channels=8,
        channels=8,
        embed_dims=16,
        enc_depth=1,
        dec_depth=2,
        num_heads=2,
        token_len=2,
        num_classes=2,
    ).eval()
    # Input is the channel-concatenated bi-temporal feature pair.
    with torch.inference_mode():
        out = head(torch.randn(2, 16, 8, 8))
    # pre_upsample x2, final upsample x4 => 8 -> 64.
    assert out.shape == (2, 2, 64, 64)


def test_bit_head_pooling_tokens() -> None:
    from opencd_lite.models import BITHead

    head = BITHead(
        in_channels=8,
        channels=8,
        embed_dims=16,
        enc_depth=1,
        dec_depth=1,
        num_heads=2,
        use_tokenizer=False,
        pool_size=2,
        num_classes=2,
    ).eval()
    with torch.inference_mode():
        out = head(torch.randn(1, 16, 8, 8))
    assert out.shape == (1, 2, 64, 64)


def test_conv_seg_head() -> None:
    from opencd_lite.models import ConvSegHead

    head = ConvSegHead(in_channels=16, num_classes=2).eval()
    with torch.inference_mode():
        out = head(torch.randn(2, 16, 32, 32))
    assert out.shape == (2, 2, 32, 32)


def test_registry_contains_supported_models() -> None:
    from opencd_lite import available_heads, available_models
    from opencd_lite.models import get_head_class, get_model_class

    assert available_models() == [
        "CGNet",
        "FC_EF",
        "FC_Siam_conc",
        "FC_Siam_diff",
        "IA_ResNet",
        "IA_ResNetV1c",
        "IA_ResNetV1d",
        "IFN",
        "LightCDNet",
        "SNUNet_ECAM",
        "TinyNet",
        "mmseg.MixVisionTransformer",
        "mmseg.ResNet",
        "mmseg.ResNetV1c",
    ]
    assert get_model_class("CGNet").__name__ == "CGNet"
    assert available_heads() == [
        "BITHead",
        "ChangeStarHead",
        "Changer",
        "DS_FPNHead",
        "STAHead",
        "TinyHead",
        "mmseg.SegformerHead",
    ]
    assert get_head_class("BITHead").__name__ == "BITHead"
