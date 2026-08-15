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
        "IFN",
        "SNUNet_ECAM",
        "mmseg.ResNet",
        "mmseg.ResNetV1c",
    ]
    assert get_model_class("CGNet").__name__ == "CGNet"
    assert available_heads() == ["BITHead"]
    assert get_head_class("BITHead").__name__ == "BITHead"
