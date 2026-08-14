"""Unit tests for desktop benchmark evaluation metrics (RF-75)."""

import numpy as np
import pytest

from backend.benchmark_ml import (
    compute_boundary_iou,
    compute_psnr,
    compute_ssim,
    generate_error_heatmap,
    mat_to_base64,
)


def test_compute_boundary_iou_identical_masks() -> None:
    """Identical non-empty masks should yield Boundary-IoU = 1.0."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    score = compute_boundary_iou(mask, mask, d=4)
    assert pytest.approx(score, rel=1e-3) == 1.0


def test_compute_boundary_iou_disjoint_masks() -> None:
    """Disjoint masks should yield Boundary-IoU = 0.0."""
    mask_a = np.zeros((100, 100), dtype=np.uint8)
    mask_b = np.zeros((100, 100), dtype=np.uint8)
    mask_a[10:30, 10:30] = 255
    mask_b[70:90, 70:90] = 255
    score = compute_boundary_iou(mask_a, mask_b, d=4)
    assert score == 0.0


def test_compute_ssim_identical_images() -> None:
    """Identical images outside mask hole should yield SSIM = 1.0."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    mask = np.ones((50, 50), dtype=np.uint8) * 255  # All background
    score = compute_ssim(img, img, mask)
    assert pytest.approx(score, rel=1e-3) == 1.0


def test_compute_psnr_identical_images() -> None:
    """Identical images outside mask hole should yield high PSNR (100 dB)."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    mask = np.ones((50, 50), dtype=np.uint8) * 255
    score = compute_psnr(img, img, mask)
    assert score >= 100.0


def test_generate_error_heatmap() -> None:
    """Error heatmap should generate valid 3-channel composite image."""
    pred = np.zeros((40, 40), dtype=np.uint8)
    gt = np.zeros((40, 40), dtype=np.uint8)
    orig = np.ones((40, 40, 3), dtype=np.uint8) * 100

    pred[10:30, 10:30] = 255
    gt[15:35, 15:35] = 255

    heatmap = generate_error_heatmap(pred, gt, orig)
    assert heatmap.shape == (40, 40, 3)
    assert heatmap.dtype == np.uint8


def test_mat_to_base64() -> None:
    """Base64 encoder should return non-empty valid PNG string."""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    b64 = mat_to_base64(img)
    assert isinstance(b64, str)
    assert len(b64) > 0
