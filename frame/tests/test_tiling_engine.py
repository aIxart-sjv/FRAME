"""frame.tiling.engine -- reconstruction, geometry, determinism, errors and the seam
diagnostic, using small deterministic fake models (no GPU, no real weights)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from frame.consistency.status import ComputationStatus
from frame.models.errors import ModelInferenceError, ModelWorkerError
from frame.tiling import (
    InvalidSceneError,
    InvalidTilingConfigError,
    ReconstructionError,
    TiledModel,
    TilingConfig,
    plan_tiles,
    run_tiled,
)
from frame.uncertainty import DEFAULT_TRANSFORMS, run_stochastic_uncertainty

SHAPES = [(128, 128), (128, 256), (256, 128), (300, 500), (511, 777)]


def perfect(x: torch.Tensor) -> torch.Tensor:
    """A position-consistent stand-in model: each output pixel depends only on the
    input pixel it came from, so the correct full-scene answer is known exactly."""
    return F.interpolate(x, scale_factor=4, mode="nearest")


def truth(scene: torch.Tensor) -> torch.Tensor:
    return perfect(scene[None])[0]


def scene_of(h: int, w: int, channels: int = 4, seed: int = 0) -> torch.Tensor:
    return torch.rand(channels, h, w, generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------ geometry & correctness


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("overlap", [0, 16, 32, 64])
def test_output_is_4h_by_4w_and_reproduces_a_position_consistent_model(shape, overlap):
    scene = scene_of(*shape)
    result = run_tiled(perfect, scene, TilingConfig(overlap=overlap))
    assert tuple(result.sr.shape) == (4, shape[0] * 4, shape[1] * 4)
    assert result.sr.dtype == torch.float32 and result.sr.device.type == "cpu"
    assert torch.allclose(result.sr, truth(scene), atol=1e-6)


@pytest.mark.parametrize("shape,expected", [((128, 128), (512, 512)), ((128, 256), (512, 1024)), ((300, 500), (1200, 2000))])
def test_documented_output_geometry_examples(shape, expected):
    assert tuple(run_tiled(perfect, scene_of(*shape)).sr.shape[1:]) == expected


@pytest.mark.parametrize("shape", SHAPES)
def test_overlap_zero_is_bit_exact_for_a_position_consistent_model(shape):
    scene = scene_of(*shape)
    assert torch.equal(run_tiled(perfect, scene, TilingConfig(overlap=0)).sr, truth(scene))


def test_a_single_tile_scene_is_bit_identical_to_calling_the_model_directly():
    scene = scene_of(128, 128)
    tiled = run_tiled(perfect, scene).sr
    assert torch.equal(tiled, perfect(scene[None])[0])


@pytest.mark.parametrize("shape", [(100, 90), (1, 1), (7, 300), (129, 129)])
def test_scenes_smaller_than_or_barely_larger_than_a_tile_are_covered_completely(shape):
    scene = scene_of(*shape)
    result = run_tiled(perfect, scene)
    assert tuple(result.sr.shape[1:]) == (shape[0] * 4, shape[1] * 4)  # padding is cropped away
    assert torch.allclose(result.sr, truth(scene), atol=1e-6)


def test_channels_are_preserved_in_order():
    scene = torch.zeros(4, 200, 300)
    for c in range(4):
        scene[c] = float(c + 1)
    out = run_tiled(perfect, scene).sr
    assert [float(out[c].mean()) for c in range(4)] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_the_model_only_ever_sees_full_size_tiles_on_the_scene_device_in_plan_order():
    seen = []

    def spy(x):
        seen.append(x.clone())
        return perfect(x)

    scene = scene_of(300, 500)
    plan = plan_tiles(300, 500)
    run_tiled(spy, scene)
    assert len(seen) == plan.tile_count == 15
    assert all(t.shape == (1, 4, 128, 128) and t.dtype == torch.float32 for t in seen)
    for x, tile in zip(seen, plan.tiles):  # row-major order; the valid part is the scene slice
        assert torch.equal(x[0, :, : tile.valid_height, : tile.valid_width], scene[:, tile.row_start : tile.row_end, tile.col_start : tile.col_end])


def test_padded_edge_tiles_are_reflections_and_never_leak_into_the_output():
    received = []

    def spy(x):
        received.append(x.clone())
        return perfect(x)

    scene = scene_of(100, 90)
    result = run_tiled(spy, scene)
    tile = received[0][0]
    assert torch.equal(tile[:, :100, :90], scene)
    assert torch.equal(tile[:, 100:, :90], scene[:, 71:99, :].flip(1))  # rows 98, 97, ... 71: mirrored about the bottom edge, edge row not repeated
    assert tuple(result.sr.shape[1:]) == (400, 360)


# --------------------------------------------------------- blending is not last-write-wins


def test_overlapping_tiles_are_blended_with_the_expected_weights_not_overwritten():
    """Two tiles (128 x 224 -> tile starts 0 and 96, overlap 32). The fake model adds a
    constant that differs per call (+1 then +3), so the overlap must be a smooth weighted
    mix of the two, computable in closed form."""
    calls = iter([1.0, 3.0])

    def offset_model(x):
        return perfect(x) + next(calls)

    scene = scene_of(128, 224)
    out = run_tiled(offset_model, scene).sr
    base = truth(scene)
    r = 32 * 4  # ramp length on the SR grid
    left_only, right_only = out[:, :, : 96 * 4], out[:, :, 128 * 4 :]
    assert torch.allclose(left_only, base[:, :, : 96 * 4] + 1.0, atol=1e-5)
    assert torch.allclose(right_only, base[:, :, 128 * 4 :] + 3.0, atol=1e-5)
    j = torch.arange(r, dtype=torch.float64)
    expected_offset = ((r - j - 0.5) / r) * 1.0 + ((j + 0.5) / r) * 3.0  # in (1, 3), rising
    got_offset = (out[:, :, 96 * 4 : 128 * 4] - base[:, :, 96 * 4 : 128 * 4])[0, 0].double()
    assert torch.allclose(got_offset, expected_offset, atol=1e-5)
    assert bool((got_offset[1:] > got_offset[:-1]).all())  # monotonic hand-over, no step


def test_overlap_zero_does_hand_over_abruptly_by_design():
    calls = iter([1.0, 3.0])
    out = run_tiled(lambda x: perfect(x) + next(calls), scene_of(128, 256), TilingConfig(overlap=0)).sr
    base = truth(scene_of(128, 256))
    assert torch.allclose(out[:, :, :512], base[:, :, :512] + 1.0, atol=1e-6)
    assert torch.allclose(out[:, :, 512:], base[:, :, 512:] + 3.0, atol=1e-6)


# ------------------------------------------------------------------------ determinism


def test_identical_runs_are_bit_identical():
    """A deterministic fake with a non-trivial nonlinearity and a call-order dependence."""

    def make():
        counter = {"n": 0}

        def model(x):
            counter["n"] += 1
            return F.interpolate(x, scale_factor=4, mode="bicubic", antialias=True).clamp(min=0) * 0.5 + 0.01 * counter["n"]

        return model

    scene = scene_of(300, 500)
    a = run_tiled(make(), scene).sr
    b = run_tiled(make(), scene).sr
    assert torch.equal(a, b)


def test_tile_order_is_row_major_and_progress_is_reported():
    events = []
    run_tiled(perfect, scene_of(300, 500), on_tile=lambda done, total, tile: events.append((done, total, tile.row_index, tile.col_index)))
    assert [e[0] for e in events] == list(range(1, 16)) and {e[1] for e in events} == {15}
    assert [(e[2], e[3]) for e in events] == [(r, c) for r in range(3) for c in range(5)]


def test_timing_is_recorded_per_tile():
    result = run_tiled(perfect, scene_of(300, 500))
    assert len(result.tile_seconds) == 15 and all(s >= 0 for s in result.tile_seconds)
    assert result.total_seconds >= sum(result.tile_seconds) * 0.99


# -------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "scene,match",
    [
        (torch.zeros(4, 0, 128), "empty"),
        (torch.zeros(4, 128, 0), "empty"),
        (torch.zeros(0, 128, 128), "no channels"),
        (torch.zeros(128, 128), "channels, height, width"),
        (torch.zeros(1, 4, 128, 128), "channels, height, width"),
        (torch.zeros(4, 128, 128, dtype=torch.int32), "floating-point"),
        ([[1.0]], "torch.Tensor"),
        (torch.full((4, 200, 200), float("nan")), "160000 non-finite"),
        (torch.cat([torch.zeros(4, 10, 10), torch.full((4, 10, 10), float("inf"))], dim=1), "non-finite"),
    ],
)
def test_invalid_scenes_are_rejected_before_any_inference(scene, match):
    calls = []
    with pytest.raises(InvalidSceneError, match=match):
        run_tiled(lambda x: calls.append(1) or perfect(x), scene)  # type: ignore[arg-type]
    assert calls == []


def test_an_invalid_config_cannot_reach_the_engine():
    with pytest.raises(InvalidTilingConfigError):
        run_tiled(perfect, scene_of(200, 200), TilingConfig(overlap=200))


def test_a_model_failure_aborts_the_scene_with_the_models_own_error_and_no_partial_result():
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if calls["n"] == 4:
            raise ModelWorkerError("The SEN2SR-Mamba worker stopped unexpectedly.", technical_detail="segfault")
        return perfect(x)

    seen = []
    with pytest.raises(ModelWorkerError, match="stopped unexpectedly") as excinfo:
        run_tiled(flaky, scene_of(300, 500), on_tile=lambda done, total, tile: seen.append(done))
    assert excinfo.value.technical_detail == "segfault"
    assert seen == [1, 2, 3]  # stopped at tile 4; nothing after it ran
    assert calls["n"] == 4


@pytest.mark.parametrize(
    "bad_output,match",
    [
        (lambda x: F.interpolate(x, scale_factor=2, mode="nearest"), "expected \\(1, C, 512, 512\\)"),
        (lambda x: F.interpolate(x, scale_factor=4, mode="nearest")[0], "expected \\(1, C, 512, 512\\)"),
        (lambda x: "not a tensor", "instead of a tensor"),
        (lambda x: F.interpolate(x, scale_factor=4, mode="nearest") * float("nan"), "NaN or Inf"),
    ],
)
def test_a_malformed_tile_output_is_reported_with_the_tile_position(bad_output, match):
    with pytest.raises(ModelInferenceError, match=match) as excinfo:
        run_tiled(bad_output, scene_of(200, 200))
    assert "tile (row 0, col 0)" in str(excinfo.value)


def test_a_change_of_channel_count_between_tiles_is_rejected():
    calls = {"n": 0}

    def shrinking(x):
        calls["n"] += 1
        y = perfect(x)
        return y if calls["n"] == 1 else y[:, :3]

    with pytest.raises(ModelInferenceError, match="3 channels.*earlier tiles had 4"):
        run_tiled(shrinking, scene_of(200, 200))


def test_an_incomplete_reconstruction_is_an_error_never_a_partial_raster(monkeypatch):
    import frame.tiling.engine as engine

    real_plan = engine.plan_tiles

    def plan_missing_a_tile(h, w, config):
        plan = real_plan(h, w, config)
        return type(plan)(plan.scene_height, plan.scene_width, plan.config, plan.row_starts, plan.col_starts, plan.tiles[:-1])

    monkeypatch.setattr(engine, "plan_tiles", plan_missing_a_tile)
    with pytest.raises(ReconstructionError, match="incomplete"):
        run_tiled(perfect, scene_of(300, 500))


# ----------------------------------------------------------------- seam diagnostic


def test_a_position_consistent_model_has_essentially_zero_seam_disagreement():
    seam = run_tiled(perfect, scene_of(300, 500)).seam_diagnostic
    assert seam.status == ComputationStatus.COMPUTABLE.value and seam.pair_count > 0
    assert seam.rmse < 1e-9 and seam.max_abs_difference < 1e-9


def test_seam_disagreement_is_measured_on_the_real_overlap_pixels():
    """Two tiles that differ by exactly 2.0 everywhere -> disagreement is exactly 2.0 over
    the 32-px (128 SR px) x 512-row overlap."""
    calls = iter([1.0, 3.0])
    seam = run_tiled(lambda x: perfect(x) + next(calls), scene_of(128, 224)).seam_diagnostic
    assert seam.pair_count == 1
    assert seam.overlap_pixel_count == (32 * 4) * 512
    assert seam.mean_abs_difference == pytest.approx(2.0) and seam.rmse == pytest.approx(2.0) and seam.max_abs_difference == pytest.approx(2.0)


def test_seam_diagnostic_is_not_computable_without_overlap_and_says_so():
    for scene, config in [(scene_of(128, 128), TilingConfig()), (scene_of(300, 500), TilingConfig(overlap=0))]:
        seam = run_tiled(perfect, scene, config).seam_diagnostic
        assert seam.status == ComputationStatus.NOT_COMPUTABLE.value and seam.rmse is None and seam.pair_count == 0
        assert "overlap" in seam.note


def test_seam_diagnostic_can_be_switched_off():
    assert run_tiled(perfect, scene_of(300, 500), collect_seam_diagnostic=False).seam_diagnostic is None


def test_seam_diagnostic_is_described_as_a_tiling_check_not_an_accuracy_measure():
    note = run_tiled(perfect, scene_of(300, 500)).seam_diagnostic.note.lower()
    assert "tiling" in note and "not an accuracy measure" in note


def test_seam_bookkeeping_memory_is_bounded_by_two_tile_rows():
    from frame.tiling.seams import SeamAccumulator

    plan = plan_tiles(700, 500)
    acc = SeamAccumulator(plan)
    peak = 0
    for tile in plan.tiles:
        acc.observe(tile, torch.zeros(4, tile.sr_valid_height, tile.sr_valid_width))
        peak = max(peak, len(acc._recent))
    assert plan.n_rows >= 6 and peak <= 2 * plan.n_cols


# ------------------------------------------------------------------------ TiledModel


def test_tiled_model_maps_a_batched_scene_to_a_batched_sr_scene():
    tm = TiledModel(perfect)
    x = scene_of(300, 500)[None]
    y = tm(x)
    assert y.shape == (1, 4, 1200, 2000) and torch.allclose(y[0], truth(x[0]), atol=1e-6)


@pytest.mark.parametrize("bad", [torch.zeros(4, 128, 128), torch.zeros(2, 4, 128, 128), "x"])
def test_tiled_model_rejects_anything_but_one_batched_scene(bad):
    with pytest.raises(InvalidSceneError, match=r"\(1, C, H, W\)"):
        TiledModel(perfect)(bad)  # type: ignore[arg-type]


def test_tiled_model_summary_is_human_readable_provenance():
    tm = TiledModel(perfect)
    assert tm.summary() == TilingConfig().describe()  # nothing run yet: just the configuration
    tm(scene_of(300, 500)[None])
    s = tm.summary()
    assert s["tile_size"] == 128 and s["overlap"] == 32 and s["padding_mode"] == "reflect" and s["blend_mode"] == "linear"
    assert s["scene_shape"] == [300, 500] and s["sr_shape"] == [1200, 2000]
    assert s["tile_grid"] == [3, 5] and s["tile_count"] == 15 and s["padded_tiles"] > 0
    assert s["scene_passes"] == 1 and s["tile_inferences"] == 15
    assert s["seam_diagnostic"]["status"] == ComputationStatus.COMPUTABLE.value


def test_tiled_model_works_inside_the_uncertainty_ensemble_on_a_rectangular_scene():
    """The 6-transform ensemble includes rot90/rot270, which TRANSPOSE a non-square scene:
    the tile engine must handle both orientations and the result must land back upright."""
    tm = TiledModel(perfect)
    scene = scene_of(200, 300)
    result = run_stochastic_uncertainty(
        tm, scene, transforms=DEFAULT_TRANSFORMS, seed=42, band_names=("B04", "B03", "B02", "B08"), keep_per_member_predictions=False
    )
    assert result.n == 6 and tuple(result.mean_prediction.shape) == (4, 800, 1200)
    assert torch.allclose(result.mean_prediction, truth(scene), atol=1e-5)
    assert float(result.std_prediction.max()) < 1e-5  # a position-consistent model is perfectly stable
    summary = tm.summary()
    assert summary["scene_passes"] == 6 and summary["scene_shape"] == [200, 300]  # first pass = identity = the input
    per_orientation = {(r.scene_shape) for r in tm.runs}
    assert per_orientation == {(200, 300), (300, 200)}
    assert summary["tile_inferences"] == sum(r.tile_count for r in tm.runs)


# ---------------------------------------------------------------- model independence


def test_the_tile_engine_contains_no_model_specific_logic():
    """Lite/Mamba selection happens before the tiler; nothing under frame/tiling may branch on a model."""
    root = Path(__file__).resolve().parents[1] / "tiling"
    forbidden_imports = ("frame.models.selection", "frame.models.config", "frame.models.mamba", "frame.api", "mlstac", "sen2sr")
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
                names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                assert not any(n and n.startswith(forbidden_imports) for n in names), (path.name, names)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                assert node.value.strip().lower() not in ("lite", "mamba"), (path.name, node.value)
