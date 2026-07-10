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


def test_conv_seg_head() -> None:
    from opencd_lite.models import ConvSegHead

    head = ConvSegHead(in_channels=16, num_classes=2).eval()
    with torch.inference_mode():
        out = head(torch.randn(2, 16, 32, 32))
    assert out.shape == (2, 2, 32, 32)


def test_registry_contains_supported_models() -> None:
    from opencd_lite import available_models
    from opencd_lite.models import get_model_class

    assert available_models() == [
        "CGNet",
        "FC_EF",
        "FC_Siam_conc",
        "FC_Siam_diff",
        "IFN",
        "SNUNet_ECAM",
    ]
    assert get_model_class("CGNet").__name__ == "CGNet"
