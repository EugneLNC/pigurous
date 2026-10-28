"""Learned whole-song action director; renderer has no audio decision logic."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def position(t, width):
    f = torch.exp(
        torch.arange(0, width, 2, device=t.device) * (-math.log(10000) / width)
    )
    x = t[..., None] * f
    return torch.stack((x.sin(), x.cos()), -1).flatten(-2)


class ActionDirector(nn.Module):
    def __init__(
        self, vocabulary, width=256, visibility_head=False, direct_effects=False
    ):
        super().__init__()
        self.register_buffer(
            "vocabulary", torch.as_tensor(vocabulary, dtype=torch.float32)
        )
        self.global_in = nn.Linear(768, width)
        self.planner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, 8, width * 4, 0.1, batch_first=True, norm_first=True
            ),
            4,
            enable_nested_tensor=False,
        )
        self.local_in = nn.Linear(897, width)
        self.local = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, 8, width * 4, 0.1, batch_first=True, norm_first=True
            ),
            4,
            enable_nested_tensor=False,
        )
        self.plan_head = nn.Linear(width, 12)
        self.token = nn.Linear(width, len(vocabulary))
        self.residual = nn.Linear(width, 12)
        self.accent = nn.Linear(width, 1)
        self.visibility = nn.Linear(width, 2) if visibility_head else None
        self.effects = nn.Linear(width, 3) if direct_effects else None

    def forward(self, audio, times, whole, whole_mask, indices):
        g = self.global_in(whole) + position(
            torch.arange(whole.shape[1], device=whole.device)[None] * 2,
            self.global_in.out_features,
        )
        g = self.planner(g, src_key_padding_mask=whole_mask)
        context = g.gather(1, indices[..., None].expand(-1, -1, g.shape[-1]))
        h = self.local_in(audio) + context + position(times, self.local_in.out_features)
        h = self.local(h)
        logits = self.token(h)
        residual = 0.25 * torch.tanh(self.residual(h))
        # Soft reconstruction trains both heads; inference executes a discrete word.
        motion = (logits.softmax(-1) @ self.vocabulary + residual).clamp(0, 1)
        result = dict(
            logits=logits,
            residual=residual,
            motion=motion,
            accent=self.accent(h).squeeze(-1),
            plan=torch.sigmoid(self.plan_head(g)),
        )
        if self.visibility is not None:
            state = self.visibility(h)
            motion = motion.clone()
            motion[..., [0, 7]] = state.sigmoid()
            result.update(motion=motion, visibility=state)
        if self.effects is not None:
            result["effects"] = self.effects(h)
        return result


def objective(pred, batch, vocabulary):
    y = batch["motion"]
    mask = batch["mask"]
    tokens = torch.cdist(y.float(), vocabulary.float()).argmin(-1)
    ce = F.cross_entropy(pred["logits"].transpose(1, 2), tokens, reduction="none")
    regression = F.smooth_l1_loss(pred["motion"], y, reduction="none").mean(-1)
    residual_target = (y - vocabulary[tokens]).clamp(-0.25, 0.25)
    residual = F.smooth_l1_loss(
        pred["residual"], residual_target, reduction="none"
    ).mean(-1)
    accent = F.binary_cross_entropy_with_logits(
        pred["accent"], batch["accent"], reduction="none", pos_weight=y.new_tensor(2.0)
    )
    valid = mask.sum().clamp_min(1)
    loss = ((ce + 12 * regression + 4 * residual + accent) * mask).sum() / valid
    if "visibility" in pred:
        state_loss = F.binary_cross_entropy_with_logits(
            pred["visibility"], y[..., [0, 7]], reduction="none"
        ).mean(-1)
        loss = loss + 2 * (state_loss * mask).sum() / valid
    if "effects" in pred:
        # Weak labels from human geometry, never from procedural audio rules.
        effect_target = torch.stack(
            [
                y[..., 5] + 0.25 * (1 - y[..., 8]),
                y[..., 4] + 0.25 * y[..., 8],
                y[..., 6] + 0.5 * y[..., 3],
            ],
            -1,
        )
        effect_target = effect_target / effect_target.sum(-1, keepdim=True).clamp_min(
            1e-6
        )
        effect_loss = -(effect_target * pred["effects"].log_softmax(-1)).sum(-1)
        loss = loss + (effect_loss * mask).sum() / valid
    pair = mask[:, 1:] * mask[:, :-1]
    velocity = F.smooth_l1_loss(
        torch.diff(pred["motion"], dim=1), torch.diff(y, dim=1), reduction="none"
    ).mean(-1)
    loss = loss + 2 * (velocity * pair).sum() / pair.sum().clamp_min(1)
    plan = ((pred["plan"] - batch["plan"]) ** 2).mean(-1)
    loss = (
        loss + 2 * (plan * (~batch["whole_mask"])).sum() / (~batch["whole_mask"]).sum()
    )
    return loss
