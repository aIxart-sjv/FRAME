"""frame.train.config -- the serializable, validated training configuration."""

from __future__ import annotations

import copy
import json

import pytest

from frame.train.config import ConfigError, TrainConfig


def base_dict():
    return {
        "name": "unit",
        "model": {"name": "tiny_cnn", "params": {"width": 8, "depth": 2}},
        "data": {"manifest": "manifest.jsonl"},
        "steps": 10,
        "output_dir": "out",
    }


def cfg(**overrides) -> TrainConfig:
    d = base_dict()
    for dotted, value in overrides.items():
        target = d
        *parents, leaf = dotted.split("__")
        for p in parents:
            target = target.setdefault(p, {})
        target[leaf] = value
    return TrainConfig.from_dict(d)


# ----------------------------------------------------------------------------- defaults and shape


def test_a_minimal_config_gets_conservative_defaults():
    c = cfg()
    assert (c.batch_size, c.grad_accum, c.seed) == (1, 1, 0)
    assert c.optim.name == "adamw" and c.scheduler.name == "none"
    assert c.loss.reconstruction == "l1" and c.loss.spectral_weight == 0.0 and c.loss.consistency_weight == 0.0
    assert c.precision.amp == "off"                                   # AMP is opt-in on a 4 GB card
    assert c.data.train_split == "train" and c.data.val_split == "val"
    assert c.data.lr_bands == ("B04", "B03", "B02", "B08")            # FRAME's RGBN order, by name
    assert c.deterministic is True and c.baselines == ("bicubic",)


def test_every_requested_field_is_present_in_the_config():
    d = cfg().to_dict()
    for key in ("model", "data", "seed", "steps", "batch_size", "grad_accum", "optim", "scheduler", "loss", "precision",
                "checkpoint_every", "validate_every", "output_dir"):
        assert key in d
    assert {"manifest", "train_split", "val_split"} <= set(d["data"]) and {"lr", "name"} <= set(d["optim"])


# ----------------------------------------------------------------------------- serialization


def test_the_config_round_trips_through_json_exactly():
    c = cfg(optim__lr=3e-4, loss__reconstruction="charbonnier", loss__spectral_weight=0.1, scheduler__name="cosine", precision__amp="fp16")
    text = c.to_json()
    assert TrainConfig.from_dict(json.loads(text)) == c
    assert TrainConfig.from_json(text) == c
    assert json.loads(text)["data"]["lr_bands"] == ["B04", "B03", "B02", "B08"]           # tuples serialise as lists


def test_json_serialisation_is_canonical_and_stable():
    a, b = cfg().to_json(), cfg().to_json()
    assert a == b and a.endswith("\n") and list(json.loads(a)) == sorted(json.loads(a))


def test_a_config_loads_from_a_file(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(base_dict()))
    assert TrainConfig.load(path) == cfg()


def test_unknown_keys_are_refused_so_a_typo_cannot_silently_use_a_default():
    d = base_dict()
    d["optim"] = {"lr": 1e-3, "learning_rate": 2e-3}
    with pytest.raises(ConfigError, match="learning_rate"):
        TrainConfig.from_dict(d)
    d = base_dict()
    d["stepz"] = 5
    with pytest.raises(ConfigError, match="stepz"):
        TrainConfig.from_dict(d)


def test_missing_required_fields_are_reported():
    d = base_dict()
    del d["model"]
    with pytest.raises(ConfigError, match="model"):
        TrainConfig.from_dict(d)
    d = base_dict()
    del d["data"]["manifest"]
    with pytest.raises(ConfigError, match="manifest"):
        TrainConfig.from_dict(d)


def test_unparseable_files_are_config_errors(tmp_path):
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(ConfigError, match="not valid JSON"):
        TrainConfig.load(tmp_path / "bad.json")
    with pytest.raises(ConfigError, match="not found"):
        TrainConfig.load(tmp_path / "absent.json")


# ----------------------------------------------------------------------------- invalid values


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"steps": 0}, "steps"),
        ({"steps": -3}, "steps"),
        ({"steps": 2.5}, "steps"),
        ({"steps": True}, "steps"),
        ({"batch_size": 0}, "batch_size"),
        ({"grad_accum": 0}, "grad_accum"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"optim__lr": 0.0}, "optim.lr"),
        ({"optim__lr": float("nan")}, "optim.lr"),
        ({"optim__lr": float("inf")}, "optim.lr"),
        ({"optim__weight_decay": -1e-4}, "optim.weight_decay"),
        ({"optim__name": "sgdd"}, "optim.name"),
        ({"optim__betas": [0.9, 1.0]}, "optim.betas"),
        ({"optim__grad_clip_norm": 0.0}, "optim.grad_clip_norm"),
        ({"loss__reconstruction": "mse"}, "loss.reconstruction"),
        ({"loss__charbonnier_eps": 0.0}, "loss.charbonnier_eps"),
        ({"loss__spectral_weight": -0.1}, "loss.spectral_weight"),
        ({"loss__consistency_weight": float("nan")}, "loss.consistency_weight"),
        ({"scheduler__name": "exp"}, "scheduler.name"),
        ({"scheduler__warmup_steps": -1}, "scheduler.warmup_steps"),
        ({"scheduler__warmup_steps": 10}, "scheduler.warmup_steps"),        # must be < steps
        ({"scheduler__name": "step", "scheduler__step_size": 0}, "scheduler.step_size"),
        ({"scheduler__name": "step", "scheduler__gamma": 1.5}, "scheduler.gamma"),
        ({"precision__amp": "int8"}, "precision.amp"),
        ({"checkpoint_every": -1}, "checkpoint_every"),
        ({"validate_every": -2}, "validate_every"),
        ({"log_every": 0}, "log_every"),
        ({"keep_last_checkpoints": 0}, "keep_last_checkpoints"),
        ({"device": "tpu"}, "device"),
        ({"name": ""}, "name"),
        ({"name": "has space"}, "name"),
        ({"name": "../escape"}, "name"),
        ({"output_dir": ""}, "output_dir"),
        ({"model__name": "resnet"}, "model.name"),
        ({"data__lr_patch": 4}, "data.lr_patch"),
        ({"data__val_lr_patch": 0}, "data.val_lr_patch"),
        ({"data__patches_per_pair": 0}, "data.patches_per_pair"),
        ({"data__min_valid_fraction": 0.0}, "data.min_valid_fraction"),
        ({"data__min_valid_fraction": 1.5}, "data.min_valid_fraction"),
        ({"data__lr_bands": []}, "data.lr_bands"),
        ({"data__lr_bands": ["B04", "B04"]}, "data.lr_bands"),
        ({"data__lr_bands": ["X9"]}, "data.lr_bands"),
        ({"data__manifest": ""}, "data.manifest"),
        ({"baselines": ["bicubic", "magic"]}, "baselines"),
    ],
)
def test_invalid_values_name_the_offending_field(overrides, field):
    with pytest.raises(ConfigError) as excinfo:
        cfg(**overrides)
    assert field in str(excinfo.value)
    assert excinfo.value.field == field


def test_test_data_can_never_be_the_train_or_validation_split():
    for key in ("data__train_split", "data__val_split"):
        with pytest.raises(ConfigError, match="test"):
            cfg(**{key: "test"})


def test_train_and_validation_must_be_different_splits():
    with pytest.raises(ConfigError, match="data.val_split"):
        cfg(data__train_split="train", data__val_split="train")


def test_an_unknown_split_name_is_refused():
    with pytest.raises(ConfigError, match="data.train_split"):
        cfg(data__train_split="training")


def test_a_valid_but_unusual_configuration_is_accepted():
    c = cfg(batch_size=2, grad_accum=4, precision__amp="bf16", scheduler__name="step", scheduler__step_size=5, scheduler__gamma=0.5,
            loss__reconstruction="charbonnier", loss__spectral_weight=0.05, loss__consistency_weight=0.1, optim__grad_clip_norm=1.0,
            data__lr_bands=["B08", "B04"], data__hr_bands=["B08", "B04"], baselines=["bicubic", "lite"], device="cuda:0")
    assert c.effective_batch_size == 8 and c.data.lr_bands == ("B08", "B04") and c.baselines == ("bicubic", "lite")


def test_lr_and_hr_band_lists_must_describe_the_same_number_of_channels():
    with pytest.raises(ConfigError, match="data.hr_bands"):
        cfg(data__lr_bands=["B04", "B03"], data__hr_bands=["B04", "B03", "B02"])


def test_hr_bands_default_to_the_lr_bands():
    assert cfg(data__lr_bands=["B08", "B04"]).data.hr_bands == ("B08", "B04")


# ----------------------------------------------------------------------------- resume signature


def test_the_resume_signature_ignores_only_fields_that_may_change_on_resume():
    base = cfg()
    for change in ({"steps": 500}, {"output_dir": "elsewhere"}, {"checkpoint_every": 7}, {"validate_every": 3}, {"log_every": 5},
                   {"keep_last_checkpoints": 9}, {"name": "renamed"}):
        assert cfg(**change).resume_signature() == base.resume_signature(), change
    for change in ({"seed": 1}, {"optim__lr": 2e-3}, {"batch_size": 2}, {"grad_accum": 2}, {"model__params": {"width": 16}}, {"loss__reconstruction": "charbonnier"},
                   {"precision__amp": "fp16"}, {"data__lr_patch": 64}, {"data__train_split": "val", "data__val_split": "train"}, {"scheduler__name": "cosine"}):
        assert cfg(**change).resume_signature() != base.resume_signature(), change


def test_the_full_digest_covers_everything():
    assert cfg().digest() != cfg(steps=11).digest() and cfg().digest() == cfg().digest()


def test_from_dict_does_not_mutate_its_input():
    d = base_dict()
    before = copy.deepcopy(d)
    TrainConfig.from_dict(d)
    assert d == before


def test_the_duplicated_band_list_matches_the_data_layers_canonical_one():
    from frame.data.bands import L2A_BANDS
    from frame.train.config import KNOWN_BANDS

    assert tuple(KNOWN_BANDS) == tuple(L2A_BANDS)


def test_the_lite_baseline_needs_the_native_128px_validation_tile():
    with pytest.raises(ConfigError, match="baselines"):
        cfg(baselines=["bicubic", "lite"], data__val_lr_patch=64)
    assert cfg(baselines=["bicubic", "lite"], data__val_lr_patch=128).baselines == ("bicubic", "lite")
