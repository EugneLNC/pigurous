"""Hierarchical audio-conditioned compound-event transformer."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .representation import BIN_SIZES, STATE_DIM, LOW, STEP


@dataclass
class ModelConfig:
    audio_dim: int = 768
    d_model: int = 512
    layers: int = 8
    heads: int = 8
    planner_dim: int = 256
    planner_layers: int = 4
    max_entities: int = 128
    visual_codes: int = 64
    dropout: float = 0.1
    checkpointing: bool = True


def positions(length, dim, device):
    p = torch.arange(length, device=device)[:, None]
    f = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000) / dim))
    out = torch.zeros(length, dim, device=device)
    out[:, 0::2] = torch.sin(p * f)
    out[:, 1::2] = torch.cos(p * f)
    return out


class Choreographer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config
        d = c.d_model
        self.transient = nn.Sequential(
            nn.Conv1d(129, 64, 5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.GELU(),
        )
        self.audio = nn.Linear(c.audio_dim + 128, d)
        self.global_in = nn.Linear(c.audio_dim, c.planner_dim)
        self.planner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                c.planner_dim,
                8,
                c.planner_dim * 4,
                c.dropout,
                batch_first=True,
                norm_first=True,
            ),
            c.planner_layers,
            enable_nested_tensor=False,
        )
        self.plan_descriptor = nn.Linear(c.planner_dim, 8)
        self.plan_code = nn.Linear(c.planner_dim, c.visual_codes)
        self.plan_out = nn.Linear(c.planner_dim, d)
        self.state_in = nn.Linear(STATE_DIM, d)
        self.cat_embed = nn.ModuleList(
            [nn.Embedding(n, d) for n in (7, c.max_entities, 9, 6)]
        )
        self.bin_embed = nn.ModuleList([nn.Embedding(n, d) for n in BIN_SIZES])
        self.res_embed = nn.Linear(6, d)
        self.decoder = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d, c.heads, d * 4, c.dropout, batch_first=True, norm_first=True
                )
                for _ in range(c.layers)
            ]
        )
        self.norm = nn.LayerNorm(d)
        self.cat_head = nn.ModuleList(
            [nn.Linear(d, n) for n in (7, c.max_entities, 9, 6)]
        )
        self.bin_head = nn.ModuleList([nn.Linear(d, n) for n in BIN_SIZES])
        self.res_head = nn.Linear(d, 6)
        self.audio_align = nn.Linear(c.audio_dim, 128)
        self.visual_align = nn.Linear(d, 128)

    def plan(self, whole, mask=None):
        x = self.global_in(whole) + positions(
            whole.shape[1], self.config.planner_dim, whole.device
        )
        h = self.planner(x, src_key_padding_mask=mask)
        # Attend over individual beats, then predict one plan per 16-beat block.
        valid = (
            torch.ones(h.shape[:2], device=h.device)
            if mask is None
            else (~mask).float()
        )
        pad = (-h.shape[1]) % 16
        h = F.pad(h, (0, 0, 0, pad))
        valid = F.pad(valid, (0, pad))
        h = (h * valid[..., None]).reshape(len(h), -1, 16, h.shape[-1]).sum(
            2
        ) / valid.reshape(len(h), -1, 16).sum(2).clamp_min(1)[..., None]
        return h, self.plan_descriptor(h), self.plan_code(h)

    def embed(self, cats, bins, residual):
        x = self.res_embed(residual)
        for i, emb in enumerate(self.cat_embed):
            x = x + emb(cats[..., i])
        for i, emb in enumerate(self.bin_embed):
            x = x + emb(bins[..., i])
        return x / math.sqrt(11)

    def field_logits(self, h, prefix):
        conditioned = h
        for i in range(prefix.shape[-1]):
            conditioned = conditioned + self.cat_embed[i](prefix[..., i]) / 2
        return self.cat_head[prefix.shape[-1]](conditioned)

    def numeric(self, h, cats):
        for i, emb in enumerate(self.cat_embed):
            h = h + emb(cats[..., i]) / 2
        return [head(h) for head in self.bin_head], self.res_head(h)

    def condition(
        self,
        audio,
        transient,
        whole,
        plan_index,
        state,
        stage=2,
        audio_mask=None,
        whole_mask=None,
        plan=None,
    ):
        h, desc, codes = self.plan(whole, whole_mask) if plan is None else plan
        tr = self.transient(transient.transpose(1, 2)).transpose(1, 2)
        reduced = audio[:, ::4]
        memory = self.audio(torch.cat([reduced, tr], -1))
        original_mask = audio_mask
        if audio_mask is not None:
            audio_mask = audio_mask[:, ::4]
        memory = memory + positions(memory.shape[1], self.config.d_model, memory.device)
        if stage >= 2:
            chosen = h[torch.arange(len(h), device=h.device), plan_index][:, None]
            slot_identity = self.cat_embed[1].weight[None]
            memory = torch.cat(
                [memory, self.plan_out(chosen), self.state_in(state) + slot_identity], 1
            )
            if audio_mask is not None:
                audio_mask = F.pad(audio_mask, (0, 1 + state.shape[1]), value=False)
        av = (
            audio.mean(1)
            if original_mask is None
            else (audio * (~original_mask)[..., None]).sum(1)
            / (~original_mask).sum(1).clamp_min(1)[:, None]
        )
        return memory, audio_mask, desc, codes, av

    def forward(
        self,
        cats,
        bins,
        residual,
        audio,
        transient,
        whole,
        plan_index,
        state,
        stage=2,
        audio_mask=None,
        whole_mask=None,
        next_cats=None,
        conditioning=None,
        last_only=False,
    ):
        memory, audio_mask, desc, codes, av = (
            self.condition(
                audio,
                transient,
                whole,
                plan_index,
                state,
                stage,
                audio_mask,
                whole_mask,
            )
            if conditioning is None
            else conditioning
        )
        x = self.embed(cats, bins, residual) + positions(
            cats.shape[1], self.config.d_model, cats.device
        )
        elapsed = ((bins[..., 0] + residual[..., 0]) / 192).cumsum(1)
        frequencies = torch.exp(
            torch.arange(0, self.config.d_model, 2, device=x.device)
            * (-math.log(10000) / self.config.d_model)
        )
        phase = elapsed[..., None] * frequencies * 192
        timing = torch.stack([phase.sin(), phase.cos()], -1).flatten(-2)
        x = x + timing
        causal = torch.ones(
            x.shape[1], x.shape[1], device=x.device, dtype=torch.bool
        ).triu(1)
        padding = cats[..., 0] == 0
        for layer in self.decoder:

            def run(v, mem, layer=layer):
                return layer(
                    v,
                    mem,
                    tgt_mask=causal,
                    tgt_key_padding_mask=padding,
                    memory_key_padding_mask=audio_mask,
                )

            x = (
                checkpoint(run, x, memory, use_reentrant=False)
                if self.training and self.config.checkpointing
                else run(x, memory)
            )
        x = self.norm(x)
        if last_only:
            x = x[:, -1:]
        # Visual alignment uses target event embeddings only, not audio-conditioned hidden states.
        visual = self.embed(cats, bins, residual)
        valid = (~padding).float()[..., None]
        visual = (visual * valid).sum(1) / valid.sum(1).clamp_min(1)
        categorical = []
        prefix = cats[:, -1:, :0] if last_only else cats[..., :0]
        for i in range(4):
            logits = self.field_logits(x, prefix)
            categorical.append(logits)
            chosen = next_cats[..., i] if next_cats is not None else logits.argmax(-1)
            prefix = torch.cat([prefix, chosen[..., None]], -1)
        numeric, numeric_residual = self.numeric(x, prefix)
        # Apply the same lifecycle grammar during teacher forcing and sampling.
        change = ((cats[..., 0] == 3).float() - (cats[..., 0] == 6).float())[
            ..., None
        ] * F.one_hot(cats[..., 1], self.config.max_entities)
        alive = state[:, None, :, 0] + change.cumsum(1)
        has_alive = (alive > 0).any(-1)
        has_free = (alive <= 0).any(-1)
        has_history = (state[:, :, 0] > 0).any(-1)[:, None] | (
            (cats[..., 0] == 3).cumsum(1) > 0
        )
        if last_only:
            has_alive = has_alive[:, -1:]
            has_free = has_free[:, -1:]
            has_history = has_history[:, -1:]
        allowed = torch.zeros_like(categorical[0], dtype=torch.bool)
        allowed[..., 2] = has_history
        allowed[..., 3] = has_free
        allowed[..., 4:] = has_alive[..., None]
        categorical[0] = categorical[0].masked_fill(~allowed, -1e4)
        return dict(
            cats=categorical,
            bins=numeric,
            residual=numeric_residual,
            hidden=x,
            descriptor=desc,
            codes=codes,
            audio_embedding=F.normalize(self.audio_align(av), dim=-1),
            visual_embedding=F.normalize(self.visual_align(visual), dim=-1),
        )


def loss(
    output,
    target_cats,
    target_bins,
    target_residual,
    plan_target=None,
    code_target=None,
    stage=1,
):
    op = target_cats[..., 0]
    valid = op != 0
    action = op >= 3
    transform = op == 4
    create = op == 3
    total = F.cross_entropy(output["cats"][0][valid], op[valid])
    parts = {"op": float(total.detach())}
    for i, mask in [(1, action), (2, transform), (3, create)]:
        if mask.any():
            total = total + F.cross_entropy(
                output["cats"][i][mask], target_cats[..., i][mask]
            )
    masks = [action, transform | (op == 5), transform, transform, create, create]
    regression = total * 0
    for i, mask in enumerate(masks):
        if mask.any():
            total = total + F.cross_entropy(
                output["bins"][i][mask], target_bins[..., i][mask]
            )
            regression = regression + F.smooth_l1_loss(
                output["residual"][..., i][mask], target_residual[..., i][mask]
            )
    total = total + regression
    parts["residual"] = float(regression.detach())
    if transform.any():
        predicted = []
        truth = []
        for i in (2, 3):
            grid = torch.arange(BIN_SIZES[i], device=op.device)
            expected = (output["bins"][i][transform].float().softmax(-1) * grid).sum(-1)
            predicted.append(
                LOW[i] + STEP[i] * (expected + output["residual"][..., i][transform])
            )
            truth.append(
                LOW[i]
                + STEP[i]
                * (target_bins[..., i][transform] + target_residual[..., i][transform])
            )
        u = torch.linspace(0, 1, 5, device=op.device)
        curve = F.smooth_l1_loss(
            predicted[0][:, None] * (1 - u) + predicted[1][:, None] * u,
            truth[0][:, None] * (1 - u) + truth[1][:, None] * u,
        )
        total = total + 0.1 * curve
        parts["curve"] = float(curve.detach())
    if stage >= 2 and plan_target is not None:
        mask = torch.isfinite(plan_target).all(-1)
        total = total + F.smooth_l1_loss(output["descriptor"][mask], plan_target[mask])
        if code_target is not None:
            total = total + F.cross_entropy(output["codes"][mask], code_target[mask])
    if stage >= 3 and len(op) > 1:
        logits = output["audio_embedding"] @ output["visual_embedding"].T / 0.07
        labels = torch.arange(len(op), device=op.device)
        align = (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        ) / 2
        total = total + 0.1 * align
        parts["alignment"] = float(align.detach())
    parts["total"] = float(total.detach())
    return total, parts
