"""Attention across modality tokens within each sample, never across samples.

Frozen feature/anchor provenance must be established by the caller: each
backbone must exclude current validation/test labels. Fit uses only ``train``
rows. Optional scalar ``anchor_keys`` are averaged in physical yield units and
the head learns the residual; they are excluded from tokens by default.
"""
from __future__ import annotations
import hashlib
from typing import Mapping
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "thermal_zero_initial_correction_attention_v1"
_DEFAULTS = {"architecture": "attention", "width": 16, "num_heads": 2,
             "epochs": 36, "loss": "huber", "huber_delta": 1.,
             "learning_rate": .001, "weight_decay": .01, "dropout": .1,
             "batch_size": 8, "threads": 2, "clip_grad_norm": 1.,
             "clamp_min_zero": True, "anchor_keys": []}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 1:
        value = value[:, None]
    _require(value.ndim == 2 and value.shape[1] > 0, f"Feature {name!r} must be [N,D] or [N]")
    return value


def _config(features, supplied):
    supplied = dict(supplied or {})
    allowed = set(_DEFAULTS) | {"feature_keys", "experts"}
    _require(not set(supplied)-allowed, f"Unknown fusion config fields: {sorted(set(supplied)-allowed)}")
    config = {**_DEFAULTS, **supplied}
    _require(isinstance(features, Mapping) and features and all(isinstance(k, str) and k for k in features),
             "Features must be a nonempty mapping with string names")
    anchors = list(config["anchor_keys"])
    keys = list(config.get("feature_keys", sorted(set(features)-set(anchors))))
    _require(keys and len(set(keys)) == len(keys) and len(set(anchors)) == len(anchors), "Modality/anchor keys must be unique")
    _require(set(keys+anchors) <= set(features), "Configured modality/anchor key is absent")
    _require(all(isinstance(k, str) for k in keys+anchors), "Modality/anchor keys must be strings")
    config["feature_keys"], config["anchor_keys"] = keys, anchors
    _require(config["architecture"] in ("attention", "attention_moe"), "Unknown fusion architecture")
    experts = config.get("experts", 4 if config["architecture"] == "attention_moe" else 1)
    _require(experts == (4 if config["architecture"] == "attention_moe" else 1), "Use one MLP or four routed experts")
    config["experts"] = int(experts)
    _require(config["width"] in (16, 32) and config["num_heads"] in (2, 4) and config["width"] % config["num_heads"] == 0, "Expected width16/32 and two/four divisible attention heads")
    config["width"], config["num_heads"] = int(config["width"]), int(config["num_heads"])
    for name in ("epochs", "batch_size", "threads"):
        _require(isinstance(config[name], (int, np.integer)) and config[name] >= 1, f"{name} must be a positive integer")
        config[name] = int(config[name])
    _require(config["loss"] in ("l1", "huber", "mse"), "Loss must be l1, huber or mse")
    for name in ("learning_rate", "huber_delta", "clip_grad_norm"):
        _require(np.isfinite(config[name]) and config[name] > 0, f"Invalid {name}")
        config[name] = float(config[name])
    _require(np.isfinite(config["weight_decay"]) and config["weight_decay"] >= 0, "Invalid weight_decay")
    _require(np.isfinite(config["dropout"]) and 0 <= config["dropout"] < 1, "Invalid dropout")
    config["weight_decay"], config["dropout"] = float(config["weight_decay"]), float(config["dropout"])
    _require(isinstance(config["clamp_min_zero"], bool), "clamp_min_zero must be boolean")
    return config


def _inputs(features, keys, expected=None):
    _require(isinstance(features, Mapping) and set(keys) <= set(features), "Missing fusion input keys")
    arrays = {name: _array(features[name], name) for name in keys}
    sizes = {value.shape[0] for value in arrays.values()}
    _require(len(sizes) == 1, "Modality row counts differ")
    if expected is not None:
        _require({name: value.shape[1] for name, value in arrays.items()} == expected, "Modality feature dimensions differ")
    return arrays, next(iter(sizes))


def _transform(arrays, keys, normalizers, indices=None):
    result = []
    for name in keys:
        values = arrays[name] if indices is None else arrays[name][indices]
        _require(np.isfinite(values).all(), f"Nonfinite consumed feature values: {name}")
        state = normalizers[name]
        scaled = (values-np.asarray(state["center"]))/np.asarray(state["scale"])
        scaled = np.where(np.asarray(state["active"]), np.clip(scaled, -8., 8.), 0.)
        result.append(torch.from_numpy(scaled.astype(np.float32)))
    return result


def _anchor(arrays, keys, count, indices=None):
    if not keys:
        return np.zeros(count if indices is None else len(indices), np.float64)
    values = [arrays[name] if indices is None else arrays[name][indices] for name in keys]
    _require(all(v.shape[1] == 1 and np.isfinite(v).all() for v in values), "Anchors must be finite scalar predictions")
    return np.stack([v[:, 0] for v in values], axis=1).mean(axis=1)


class ModalityAttentionHead(nn.Module):
    """Modality-token attention followed by a residual MLP or four experts."""
    def __init__(self, dimensions, config):
        super().__init__()
        width = config["width"]
        self.projections = nn.ModuleList(nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.LayerNorm(width)) for d in dimensions)
        self.modality_embedding = nn.Parameter(torch.zeros(1, len(dimensions), width))
        nn.init.normal_(self.modality_embedding, std=.02)
        self.attention = nn.MultiheadAttention(width, config["num_heads"], dropout=config["dropout"], batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.pool_norm = nn.LayerNorm(width)
        self.direct = nn.Linear(width, 1)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(config["dropout"]), nn.Linear(width, 1))
                                     for _ in range(config["experts"]))
        self.router = nn.Linear(width, config["experts"]) if config["experts"] > 1 else None
        # Initial physical prediction is the training mean or mean-bias-corrected anchor.
        nn.init.zeros_(self.direct.weight);nn.init.zeros_(self.direct.bias)
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight);nn.init.zeros_(expert[-1].bias)

    def forward(self, features):
        tokens = torch.stack([projection(value) for projection, value in zip(self.projections, features, strict=True)], dim=1)
        tokens = tokens+self.modality_embedding
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        pooled = self.pool_norm(self.attention_norm(tokens+attended).mean(dim=1))
        outputs = torch.cat([expert(pooled) for expert in self.experts], dim=1)
        correction = outputs[:, :1] if self.router is None else (self.router(pooled).softmax(-1)*outputs).sum(-1, keepdim=True)
        return (self.direct(pooled)+correction).squeeze(-1)


def _loss(prediction, target, config):
    if config["loss"] == "l1":
        return F.l1_loss(prediction, target)
    if config["loss"] == "mse":
        return F.mse_loss(prediction, target)
    return F.huber_loss(prediction, target, delta=config["huber_delta"])


def fit_head(features, y, train, config=None, seed=20261012):
    """Return a torch-saveable checkpoint fitted only on declared training rows.

    Labels/features outside ``train`` may be NaN and do not affect the fit.
    The caller must separately establish upstream backbone/anchor isolation.
    """
    config = _config(features, config)
    names = list(dict.fromkeys(config["feature_keys"]+config["anchor_keys"]))
    arrays, count = _inputs(features, names)
    target = np.asarray(y, dtype=np.float64)
    indices = np.asarray(train)
    _require(indices.ndim == 1 and len(indices) and np.issubdtype(indices.dtype, np.integer), "Training indices must be a nonempty integer vector")
    indices = indices.astype(np.int64)
    _require(len(set(indices.tolist())) == len(indices) and indices.min() >= 0 and indices.max() < count, "Training indices must be unique and in range")
    _require(target.shape == (count,) and np.isfinite(target[indices]).all(), "Training targets must be finite and aligned")
    _require(isinstance(seed, (int, np.integer)) and 0 <= seed < 2**63, "Seed must be a nonnegative integer")
    normalizers, feature_hashes = {}, {}
    for name in names:
        selected = arrays[name][indices]
        _require(np.isfinite(selected).all(), f"Nonfinite training features: {name}")
        feature_hashes[name] = hashlib.sha256(selected.astype('<f8').tobytes(order='C')).hexdigest()
        if name in config["feature_keys"]:
            center, std = np.median(selected, axis=0), np.std(selected, axis=0)
            active = std >= 1e-6
            normalizers[name] = {"center": center.tolist(), "scale": np.where(active, std, 1.).tolist(), "active": active.tolist()}
    inputs = _transform(arrays, config["feature_keys"], normalizers, indices)
    anchor = _anchor(arrays, config["anchor_keys"], count, indices)
    residual = target[indices]-anchor
    center, std = float(residual.mean()), float(residual.std())
    scale = std if std >= 1e-6 else 1.
    targets = torch.from_numpy(((residual-center)/scale).astype(np.float32))
    dimensions = {name: arrays[name].shape[1] for name in names}
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    # Keep fit randomness seed-local and preserve the caller's CPU RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = ModalityAttentionHead([dimensions[k] for k in config["feature_keys"]], config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        rng, history = np.random.default_rng(seed), []
        for epoch in range(1, config["epochs"]+1):
            model.train();total = 0.
            order = rng.permutation(len(indices))
            for start in range(0, len(order), config["batch_size"]):
                rows = torch.from_numpy(order[start:start+config["batch_size"]])
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model([value[rows] for value in inputs]), targets[rows], config)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite fusion training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"], error_if_nonfinite=True)
                optimizer.step();total += float(loss.detach())*len(rows)
            history.append({"epoch": epoch, "training_loss_normalized": total/len(indices)})
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {"format_version": 1, "architecture": ARCHITECTURE, "config": config, "seed": int(seed),
            "input_dimensions": dimensions, "feature_keys": config["feature_keys"], "anchor_keys": config["anchor_keys"],
            "training_indices": indices.tolist(), "training_n": len(indices), "input_rows_at_fit": count,
            "normalizers": normalizers, "target_center": center, "target_scale": scale,
            "target_definition": "training yield minus mean scalar anchor" if config["anchor_keys"] else "training yield",
            "anchor_definition": "Arithmetic mean in physical yield units; never standardized" if config["anchor_keys"] else None,
            "training_feature_sha256": feature_hashes,
            "training_target_sha256": hashlib.sha256(target[indices].astype('<f8').tobytes()).hexdigest(),
            "normalization": "Training-only per-column median/std; constant columns set to zero; clip standardized values to [-8,8]",
            "backbone_provenance_requirement": "Caller must establish that upstream features and anchor predictions exclude heldout labels",
            "history": history, "state_dict": state}


@torch.inference_mode()
def predict_head(checkpoint, features):
    """Restore a head and predict without labels; a new row count is permitted."""
    _require(checkpoint.get("architecture") == ARCHITECTURE and checkpoint.get("format_version") == 1, "Unsupported attention checkpoint")
    config = checkpoint["config"]
    _require(checkpoint["feature_keys"] == config["feature_keys"] and checkpoint["anchor_keys"] == config["anchor_keys"], "Checkpoint modality order differs")
    names = list(dict.fromkeys(checkpoint["feature_keys"]+checkpoint["anchor_keys"]))
    arrays, count = _inputs(features, names, checkpoint["input_dimensions"])
    inputs = _transform(arrays, checkpoint["feature_keys"], checkpoint["normalizers"])
    anchor = _anchor(arrays, checkpoint["anchor_keys"], count)
    if count == 0:
        return np.empty(0, np.float64)
    torch.set_num_threads(config["threads"])
    with torch.random.fork_rng(devices=[]):
        model = ModalityAttentionHead([checkpoint["input_dimensions"][k] for k in checkpoint["feature_keys"]], config).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    predictions = []
    for start in range(0, count, config["batch_size"]):
        normalized = model([value[start:start+config["batch_size"]] for value in inputs]).numpy().astype(np.float64)
        predictions.append(normalized*checkpoint["target_scale"]+checkpoint["target_center"]+anchor[start:start+config["batch_size"]])
    result = np.concatenate(predictions)
    if config["clamp_min_zero"]:
        result = np.maximum(result, 0.)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite fusion predictions")
    return result
