"""Portable original attention-MoE architecture for fold-isolated fusion studies.

BatchAttentionMoE, routing, normalize, clip_independent and combination_seed
are copied verbatim from the prior repository. SOURCE_PROVENANCE pins the
original files; VENDORED_SEGMENTS pins each copied definition. The wrapper
fits one candidate at a time and adds L1/Huber choices to the original MSE.
There is no runtime import or dependency on the sibling repository.
"""
from __future__ import annotations

import hashlib
import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "original_batch_attention_moe_portable_v1"

SOURCE_PROVENANCE = {'attention_moe.py': '4ddb702fe99b4952128e4648ec3d9a937741abeb99227275b88a30f87c12cc06', 'expanded_heads.py': '422a4a0a1269b246129a39fadc3a1a1a451988835147e159df45800d83301d0c', 'multimodel_heads.py': '7c6a1135233141936c88b984dc982498a068bce4f9205028f0a8e3aced928a10'}
VENDORED_SEGMENTS = {'routing': '8ea86ea2f1ed8f6812f3c411ffafaec787b1069014b03963cb31e34abc1abaa1', 'BatchAttentionMoE': 'cf44e04b31d657289fdcc0efc7e51dedc42616532265bd074d751f97fda74018', 'normalize': '5147c2e419a820b103331b73fd04f0fc1a7056f9480ac58d5659e2d51f8bd33e', 'clip_independent': '2000c4dc9e2d29198100e2b05cc8cbfea738207f63c3c09563a0e184aaf8849a', 'combination_seed': 'e54125dcc2357623e22a4f6d1827866696ae2d2c6bcb6d31676c47dfe8fe3090'}


def routing(logits, top_k):
    probabilities = logits.softmax(-1)
    experts = logits.shape[-1]
    if top_k:
        selected = logits.topk(top_k,dim=-1).indices
        mask = torch.zeros_like(logits).scatter_(-1,selected,1.)
        weights = logits.masked_fill(mask==0,float('-inf')).softmax(-1)
        load = mask.mean(1).detach()/top_k
        balance = experts*(load*probabilities.mean(1)).sum(-1)
    else:
        weights = probabilities
        balance = experts*probabilities.mean(1).square().sum(-1)
    z_loss = logits.logsumexp(-1).square().mean(1)
    return weights,probabilities,balance,z_loss


class BatchAttentionMoE(nn.Module):
    """All candidate models have independent projections, experts and gates.

    Routing is at parcel level. Top-k weights select expert predictions; this
    CPU implementation evaluates all experts, so top-k is not a compute claim.
    """
    def __init__(self, dimensions,config,seeds):
        super().__init__()
        assert len(dimensions)==len(seeds) and len({len(d) for d in dimensions})==1
        assert config['width']%config['attention_heads']==0
        assert 0<=config['top_k']<=config['experts']
        self.config = dict(config)
        self.dimensions = [list(d) for d in dimensions]
        count,tokens = len(dimensions),len(dimensions[0])
        width,experts = config['width'],config['experts']
        maximum = max(max(d) for d in dimensions)
        generators = [torch.Generator().manual_seed(int(s)) for s in seeds]
        self.projection = nn.Parameter(torch.zeros(count,tokens,maximum,width))
        self.projection_bias = nn.Parameter(torch.zeros(count,tokens,width))
        self.identities = nn.Parameter(torch.empty(count,tokens,width))
        self.cls = nn.Parameter(torch.empty(count,experts,1,width))
        self.qkv = nn.Parameter(torch.empty(count,experts,width,3*width))
        self.qkv_bias = nn.Parameter(torch.empty(count,experts,1,3*width))
        self.attention_output = nn.Parameter(torch.empty(count,experts,width,width))
        self.attention_bias = nn.Parameter(torch.empty(count,experts,1,width))
        self.expert_output = nn.Parameter(torch.empty(count,experts,width,1))
        self.expert_bias = nn.Parameter(torch.empty(count,experts,1,1))
        self.gate = nn.Parameter(torch.empty(count,width,experts))
        self.gate_bias = nn.Parameter(torch.zeros(count,1,experts))
        self.token_norm_weight = nn.Parameter(torch.ones(count,1,1,width))
        self.token_norm_bias = nn.Parameter(torch.zeros(count,1,1,width))
        self.expert_norm_weight = nn.Parameter(torch.ones(count,experts,1,width))
        self.expert_norm_bias = nn.Parameter(torch.zeros(count,experts,1,width))
        with torch.no_grad():
            for i,generator in enumerate(generators):
                for j,dimension in enumerate(dimensions[i]):
                    self.projection[i,j,:dimension].uniform_(-1/math.sqrt(dimension),1/math.sqrt(dimension),generator=generator)
                    self.projection_bias[i,j].uniform_(-1/math.sqrt(dimension),1/math.sqrt(dimension),generator=generator)
                self.identities[i].normal_(0,.02,generator=generator)
                self.cls[i].normal_(0,.02,generator=generator)
                for parameter in (self.qkv,self.qkv_bias,self.attention_output,self.attention_bias,
                                  self.expert_output,self.expert_bias):
                    parameter[i].uniform_(-1/math.sqrt(width),1/math.sqrt(width),generator=generator)
                self.gate[i].normal_(0,.01,generator=generator)

    def forward(self,values,dropout=0):
        count,batch,tokens,maximum = values.shape
        width,experts = self.config['width'],self.config['experts']
        heads = self.config['attention_heads']
        v = values.permute(0,2,1,3).reshape(count*tokens,batch,maximum)
        projected = torch.bmm(v,self.projection.reshape(count*tokens,maximum,width))
        projected = projected.reshape(count,tokens,batch,width).permute(0,2,1,3)
        projected = F.gelu(projected+self.projection_bias[:,None]+self.identities[:,None])
        projected = F.layer_norm(projected,(width,))*self.token_norm_weight+self.token_norm_bias
        projected = F.dropout(projected,dropout,training=self.training)
        logits = torch.bmm(projected.mean(2),self.gate)+self.gate_bias
        weights,probabilities,balance,z_loss = routing(logits,self.config['top_k'])
        cls = self.cls[:,:,None].expand(count,experts,batch,1,width)
        body = projected[:,None].expand(count,experts,batch,tokens,width)
        expert_tokens = torch.cat([cls,body],dim=3)
        sequence = tokens+1
        inputs = expert_tokens.reshape(count*experts,batch*sequence,width)
        qkv = torch.bmm(inputs,self.qkv.reshape(count*experts,width,3*width))
        qkv = qkv.reshape(count,experts,batch,sequence,3*width)+self.qkv_bias[:,:,None]
        qkv = qkv.reshape(count,experts,batch,sequence,3,heads,width//heads)
        q,k,v = [part.permute(0,1,2,4,3,5) for part in qkv.unbind(4)]
        attended = F.scaled_dot_product_attention(q,k,v,dropout_p=dropout if self.training else 0.)
        context = attended.permute(0,1,2,4,3,5).reshape(count,experts,batch,sequence,width)[:,:,:,0,:]
        context = torch.bmm(context.reshape(count*experts,batch,width),self.attention_output.reshape(count*experts,width,width))
        context = context.reshape(count,experts,batch,width)+self.attention_bias+self.cls
        context = F.layer_norm(context,(width,))*self.expert_norm_weight+self.expert_norm_bias
        context = F.dropout(F.gelu(context),dropout,training=self.training)
        predictions = torch.bmm(context.reshape(count*experts,batch,width),self.expert_output.reshape(count*experts,width,1))
        predictions = (predictions.reshape(count,experts,batch,1)+self.expert_bias).squeeze(-1).transpose(1,2)
        prediction = (weights*predictions).sum(-1)
        auxiliary = self.config['balance_weight']*balance+self.config['router_z_weight']*z_loss
        return prediction,auxiliary,weights,probabilities,predictions


def normalize(values,center,scale,active):
    return ((values-center.astype(np.float32))/scale.astype(np.float32)).astype(np.float32)*active


def clip_independent(model, limit=1.):
    norms = sum(p.grad.square().flatten(1).sum(1) for p in model.parameters()).sqrt()
    factors = (limit/(norms+1e-6)).clamp(max=1.)
    for p in model.parameters():
        p.grad.mul_(factors.reshape(len(factors), *([1]*(p.ndim-1))))


def combination_seed(names, seed):
    return (int(hashlib.sha256('|'.join(names).encode()).hexdigest()[:8],16)+seed) % (2**31-1)


_DEFAULTS = {"architecture": "legacy_attention_moe", "width": 16, "experts": 4,
             "attention_heads": 2, "top_k": 0, "epochs": 36, "dropout": .2,
             "weight_decay": .01, "batch_size": 8, "learning_rate": .001,
             "balance_weight": .01, "router_z_weight": .001,
             "loss": "mse", "huber_delta": 1., "threads": 2,
             "clip_grad_norm": 1., "clamp_min_zero": True, "anchor_keys": []}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _config(features, supplied):
    supplied = dict(supplied or {})
    allowed = set(_DEFAULTS) | {"feature_keys", "num_heads", "name"}
    _require(not set(supplied)-allowed, f"Unknown legacy attention configuration fields: {sorted(set(supplied)-allowed)}")
    if "num_heads" in supplied:
        _require("attention_heads" not in supplied or supplied["num_heads"] == supplied["attention_heads"], "Attention head aliases differ")
        supplied["attention_heads"] = supplied.pop("num_heads")
    config = {**_DEFAULTS, **supplied}
    if config["architecture"] == "attention" and "experts" not in supplied:
        config["experts"] = 1
    _require(config["architecture"] in ("legacy_attention_moe", "attention", "attention_moe"), "Unknown legacy attention architecture")
    _require(isinstance(features, Mapping) and features and all(isinstance(k, str) and k for k in features), "Features must map nonempty names to arrays")
    config["anchor_keys"] = list(config["anchor_keys"])
    config["feature_keys"] = list(config.get("feature_keys", sorted(set(features)-set(config["anchor_keys"]))))
    for name in ("feature_keys", "anchor_keys"):
        keys = config[name]
        _require(len(set(keys)) == len(keys) and set(keys) <= set(features), f"Invalid {name}")
    _require(config["feature_keys"], "At least one feature token is required")
    _require(config["width"] in (16, 32) and config["experts"] in (1, 4), "Use width16/32 and one/four experts")
    _require(config["attention_heads"] == 2, "This declared family uses two attention heads")
    _require(isinstance(config["top_k"], int) and 0 <= config["top_k"] <= config["experts"], "Invalid top_k")
    if config["experts"] == 1:
        for name in ("balance_weight", "router_z_weight"):
            if name not in supplied:
                config[name] = 0.
    for name in ("epochs", "batch_size", "threads"):
        _require(isinstance(config[name], (int, np.integer)) and config[name] > 0, f"Invalid {name}")
        config[name] = int(config[name])
    for name in ("learning_rate", "huber_delta", "clip_grad_norm"):
        _require(np.isfinite(config[name]) and config[name] > 0, f"Invalid {name}")
    for name in ("weight_decay", "balance_weight", "router_z_weight"):
        _require(np.isfinite(config[name]) and config[name] >= 0, f"Invalid {name}")
    _require(0 <= config["dropout"] < 1 and np.isfinite(config["dropout"]), "Invalid dropout")
    _require(config["loss"] in ("l1", "huber", "mse"), "Loss must be l1, huber or mse")
    _require(isinstance(config["clamp_min_zero"], bool), "clamp_min_zero must be boolean")
    return config


def _inputs(features, names, expected=None):
    _require(isinstance(features, Mapping) and set(names) <= set(features), "Missing legacy attention inputs")
    arrays = {}
    for name in names:
        values = np.asarray(features[name])
        if values.ndim == 1:
            values = values[:, None]
        _require(values.ndim == 2 and values.shape[1] > 0 and values.dtype.kind in "fi", f"Invalid feature array {name}")
        arrays[name] = values
    _require(len({len(a) for a in arrays.values()}) == 1, "Feature row counts differ")
    if expected is not None:
        _require({name: a.shape[1] for name, a in arrays.items()} == expected, "Feature dimensions differ")
    return arrays, len(next(iter(arrays.values())))


def _values(arrays, names, transforms, rows=None):
    count = len(next(iter(arrays.values()))) if rows is None else len(rows)
    maximum = max(arrays[name].shape[1] for name in names)
    values = np.zeros((1, count, len(names), maximum), np.float32)
    for token, name in enumerate(names):
        raw = arrays[name] if rows is None else arrays[name][rows]
        _require(np.isfinite(raw).all(), f"Nonfinite consumed features: {name}")
        transform = transforms[name]
        values[0, :, token, :raw.shape[1]] = normalize(
            raw, np.asarray(transform["center"]), np.asarray(transform["scale"]), np.asarray(transform["active"]))
    _require(np.isfinite(values).all(), "Legacy normalization produced nonfinite values")
    return torch.from_numpy(values)


def _anchor(arrays, names, count, rows=None):
    if not names:
        return np.zeros(count if rows is None else len(rows), np.float64)
    columns = [arrays[name] if rows is None else arrays[name][rows] for name in names]
    _require(all(a.shape[1] == 1 and np.isfinite(a).all() for a in columns), "Anchors must be finite scalar predictions")
    return np.stack([a[:, 0].astype(np.float64) for a in columns], axis=1).mean(axis=1)


def fit_head(features, y, train, config=None, seed=20261012):
    """Fit only selected rows using the original attention expert equations.

    With MSE and no anchors this retains the old training rule, mean/std
    normalization without clipping, combination-seeded initialization, AdamW,
    routing auxiliaries and independent gradient clipping. Optional anchors
    define a residual target; their default is empty.
    """
    config = _config(features, config)
    names = list(dict.fromkeys(config["feature_keys"]+config["anchor_keys"]))
    arrays, count = _inputs(features, names)
    target, rows = np.asarray(y, np.float64), np.asarray(train)
    _require(rows.ndim == 1 and len(rows) and np.issubdtype(rows.dtype, np.integer), "Train must be a nonempty integer vector")
    rows = rows.astype(np.int64)
    _require(len(np.unique(rows)) == len(rows) and rows.min() >= 0 and rows.max() < count, "Training rows must be unique and in range")
    _require(target.shape == (count,) and np.isfinite(target[rows]).all(), "Training targets must be finite and aligned")
    _require(isinstance(seed, (int, np.integer)) and 0 <= seed < 2**63, "Invalid seed")
    transforms, feature_hashes = {}, {}
    for name in names:
        selected = arrays[name][rows]
        _require(np.isfinite(selected).all(), f"Nonfinite training features: {name}")
        feature_hashes[name] = hashlib.sha256(selected.astype("<f8").tobytes()).hexdigest()
        if name in config["feature_keys"]:
            center, std = selected.astype(float).mean(0), selected.astype(float).std(0)
            active = std >= 1e-6
            transforms[name] = {"center": center.tolist(), "scale": np.where(active, std, 1.).tolist(), "active": active.tolist()}
    values = _values(arrays, config["feature_keys"], transforms, rows)
    residual = target[rows]-_anchor(arrays, config["anchor_keys"], count, rows)
    mean, scale = float(residual.mean()), max(float(residual.std()), 1e-6)
    targets = torch.from_numpy(((residual-mean)/scale).astype(np.float32))
    dimensions = {name: arrays[name].shape[1] for name in names}
    model_seed = combination_seed(config["feature_keys"], int(seed))
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        model = BatchAttentionMoE([[dimensions[k] for k in config["feature_keys"]]], config, [model_seed])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        torch.manual_seed(int(seed))
        rng, history = np.random.default_rng(seed), []
        for epoch in range(1, config["epochs"]+1):
            model.train()
            order, total = rng.permutation(len(rows)), torch.zeros(1)
            for start in range(0, len(order), config["batch_size"]):
                local = order[start:start+config["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                prediction, auxiliary, *_ = model(values[:, local], config["dropout"])
                error = prediction-targets[local][None]
                if config["loss"] == "mse":
                    data_loss = error.square().mean(1)
                elif config["loss"] == "l1":
                    data_loss = error.abs().mean(1)
                else:
                    data_loss = F.huber_loss(prediction, targets[local][None], reduction="none", delta=config["huber_delta"]).mean(1)
                losses = data_loss+auxiliary
                if not torch.isfinite(losses).all():
                    raise RuntimeError("Nonfinite legacy attention training loss")
                losses.sum().backward()
                clip_independent(model, config["clip_grad_norm"])
                optimizer.step()
                total += losses.detach()*len(local)
            history.append({"epoch": epoch, "training_loss_normalized": float(total[0]/len(rows))})
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {"format_version": 1, "architecture": ARCHITECTURE, "config": config, "seed": int(seed),
        "feature_keys": config["feature_keys"], "anchor_keys": config["anchor_keys"], "input_dimensions": dimensions,
        "training_indices": rows.tolist(), "training_n": len(rows), "input_rows_at_fit": count,
        "normalizers": transforms, "target_center": mean, "target_scale": scale,
        "target_definition": "training yield minus mean scalar anchor" if config["anchor_keys"] else "training yield",
        "normalization": "Original training-only mean/std; float32 transform; inactive columns zero; no clipping",
        "model_initialization_seed": model_seed, "state_dict": state, "history": history,
        "training_feature_sha256": feature_hashes,
        "training_target_sha256": hashlib.sha256(target[rows].astype("<f8").tobytes()).hexdigest(),
        "source_provenance": SOURCE_PROVENANCE, "vendored_segments": VENDORED_SEGMENTS,
        "backbone_provenance_requirement": "Caller must establish upstream backbone and anchor isolation"}


@torch.inference_mode()
def predict_head(checkpoint, features):
    """Predict without labels with original all-row attention-head replay."""
    _require(checkpoint.get("architecture") == ARCHITECTURE and checkpoint.get("format_version") == 1, "Unsupported legacy attention checkpoint")
    config = checkpoint["config"]
    _require(checkpoint["feature_keys"] == config["feature_keys"] and checkpoint["anchor_keys"] == config["anchor_keys"], "Checkpoint feature order differs")
    names = list(dict.fromkeys(config["feature_keys"]+config["anchor_keys"]))
    arrays, count = _inputs(features, names, checkpoint["input_dimensions"])
    if not count:
        return np.empty(0, np.float64)
    values = _values(arrays, config["feature_keys"], checkpoint["normalizers"])
    anchor = _anchor(arrays, config["anchor_keys"], count)
    torch.set_num_threads(config["threads"])
    with torch.random.fork_rng(devices=[]):
        model = BatchAttentionMoE([[checkpoint["input_dimensions"][k] for k in config["feature_keys"]]], config, [0]).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    normalized, *_ = model(values)
    result = normalized[0].numpy().astype(float)*checkpoint["target_scale"]+checkpoint["target_center"]+anchor
    if config["clamp_min_zero"]:
        result = np.maximum(result, 0.)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite legacy attention predictions")
    return result
