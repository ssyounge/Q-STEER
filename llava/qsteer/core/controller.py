from __future__ import annotations

import torch
import torch.nn as nn


class QSTEERController(nn.Module):
    @staticmethod
    def _sanitize_tensor(
        value: torch.Tensor,
        *,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
    ) -> torch.Tensor:
        out = value
        if not torch.isfinite(out).all():
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_min is not None or clamp_max is not None:
            out = out.clamp(min=clamp_min, max=clamp_max)
        return out

    def __init__(
        self,
        hidden_size: int,
        expert_num: int,
        icr_rank: int,
        route_layer_count: int | None = None,
        late_layer_count: int = 1,
        controller_hidden: int = 256,
        controller_layers: int = 2,
        controller_dropout: float = 0.0,
        g_temp: float = 1.0,
        num_heads: int = 1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.expert_num = int(expert_num)
        if route_layer_count is None:
            route_layer_count = late_layer_count
        self.route_layer_count = max(1, int(route_layer_count))
        self.late_layer_count = max(1, int(late_layer_count))
        self.icr_rank = int(icr_rank)
        self.g_temp = float(max(g_temp, 1e-6))
        self.num_heads = max(1, int(num_heads))
        self.strict_q_only = True

        self._gate_feat_dim = int(self.hidden_size) + 1  # [c ; d]
        self.old_route_w = nn.Parameter(
            torch.empty(self.route_layer_count, self.expert_num, self._gate_feat_dim)
        )
        self.old_route_b = nn.Parameter(torch.zeros(self.route_layer_count, self.expert_num))
        self.new_route_w = nn.Parameter(
            torch.empty(self.route_layer_count, self.expert_num, self._gate_feat_dim)
        )
        self.new_route_b = nn.Parameter(torch.zeros(self.route_layer_count, self.expert_num))

        for param in (self.old_route_w, self.new_route_w):
            nn.init.normal_(param, mean=0.0, std=0.02)

        mlp = []
        in_dim = self._gate_feat_dim
        for _ in range(max(int(controller_layers) - 1, 0)):
            mlp.append(nn.Linear(in_dim, int(controller_hidden)))
            mlp.append(nn.GELU())
            if float(controller_dropout) > 0.0:
                mlp.append(nn.Dropout(float(controller_dropout)))
            in_dim = int(controller_hidden)
        self.ctrl_mlp = nn.Sequential(*mlp) if mlp else nn.Identity()
        self.alpha_head = nn.Linear(in_dim, self.late_layer_count * self.icr_rank)
        self.gamma_head = nn.Linear(in_dim, self.late_layer_count * self.num_heads)
        self.lambda_head = nn.Linear(in_dim, 1)

    def set_old_branch_trainable(self, trainable: bool) -> None:
        enabled = bool(trainable)
        for param in (self.old_route_w, self.old_route_b):
            param.requires_grad_(enabled)

    def set_new_branch_trainable(self, trainable: bool) -> None:
        enabled = bool(trainable)
        for param in (self.new_route_w, self.new_route_b):
            param.requires_grad_(enabled)
        for module in (self.ctrl_mlp, self.alpha_head, self.gamma_head, self.lambda_head):
            for param in module.parameters():
                param.requires_grad_(enabled)

    def enforce_training_policy(self) -> None:
        """Keep the shared controller trainable throughout continual learning."""
        self.set_old_branch_trainable(True)
        self.set_new_branch_trainable(True)

    def branch_state(self) -> dict[str, bool]:
        return {
            "old_branch_trainable": bool(
                any(bool(p.requires_grad) for p in (self.old_route_w, self.old_route_b))
            ),
            "old_branch_frozen": bool(
                not any(bool(p.requires_grad) for p in (self.old_route_w, self.old_route_b))
            ),
            "new_branch_trainable": bool(
                all(bool(p.requires_grad) for p in (self.new_route_w, self.new_route_b))
            ),
            "alpha_head_trainable": bool(
                any(bool(p.requires_grad) for p in self.alpha_head.parameters())
            ),
            "gamma_head_trainable": bool(
                any(bool(p.requires_grad) for p in self.gamma_head.parameters())
            ),
            "lambda_head_trainable": bool(
                any(bool(p.requires_grad) for p in self.lambda_head.parameters())
            ),
        }

    @classmethod
    def promote_route_tensors(
        cls,
        old_route_w: torch.Tensor,
        old_route_b: torch.Tensor,
        new_route_w: torch.Tensor,
        new_route_b: torch.Tensor,
        *,
        promoted_expert_ids_by_layer: dict[int, list[int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        promoted_old_w = old_route_w.detach().clone()
        promoted_old_b = old_route_b.detach().clone()

        if not promoted_expert_ids_by_layer:
            return promoted_old_w, promoted_old_b

        for raw_layer_id, raw_expert_ids in dict(promoted_expert_ids_by_layer).items():
            try:
                layer_id = int(raw_layer_id)
            except Exception:
                continue
            if layer_id < 0 or layer_id >= int(promoted_old_w.size(0)):
                continue
            if raw_expert_ids is None:
                continue
            expert_ids = []
            for raw_expert_id in list(raw_expert_ids):
                try:
                    expert_id = int(raw_expert_id)
                except Exception:
                    continue
                if 0 <= expert_id < int(promoted_old_w.size(1)):
                    expert_ids.append(expert_id)
            if not expert_ids:
                continue
            idx = torch.tensor(
                sorted(set(expert_ids)),
                dtype=torch.long,
                device=promoted_old_w.device,
            )
            promoted_w = new_route_w[layer_id].index_select(0, idx)
            promoted_b = new_route_b[layer_id].index_select(0, idx)
            promoted_old_w[layer_id].index_copy_(0, idx, promoted_w)
            promoted_old_b[layer_id].index_copy_(0, idx, promoted_b)

        return promoted_old_w, promoted_old_b

    @torch.no_grad()
    def promote_new_branch_to_old(
        self,
        *,
        promoted_expert_ids_by_layer: dict[int, list[int]] | None = None,
    ) -> None:
        promoted_old_w, promoted_old_b = self.promote_route_tensors(
            self.old_route_w,
            self.old_route_b,
            self.new_route_w,
            self.new_route_b,
            promoted_expert_ids_by_layer=promoted_expert_ids_by_layer,
        )
        self.old_route_w.copy_(promoted_old_w)
        self.old_route_b.copy_(promoted_old_b)
        self.enforce_training_policy()

    def pool_context(self, inputs_embeds: torch.Tensor, q_mask: torch.Tensor = None):
        inputs_embeds = self._sanitize_tensor(inputs_embeds)
        if q_mask is None:
            if bool(getattr(self, "strict_q_only", False)):
                raise RuntimeError("QSTEER strict_q_only: q_mask is required but missing.")
            return self._sanitize_tensor(inputs_embeds.mean(dim=1), clamp_min=-1.0e4, clamp_max=1.0e4)

        mask = q_mask.to(device=inputs_embeds.device)
        if mask.dtype != torch.bool:
            mask = mask > 0
        weights = mask.to(dtype=inputs_embeds.dtype)

        denom = weights.sum(dim=1, keepdim=True)
        pooled = (inputs_embeds * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / denom.clamp(min=1.0)

        no_question = denom.squeeze(-1) == 0
        if no_question.any():
            if bool(getattr(self, "strict_q_only", False)):
                raise RuntimeError("QSTEER strict_q_only: q_mask is empty for some samples.")
            pooled[no_question] = inputs_embeds[no_question].mean(dim=1)
        return self._sanitize_tensor(pooled, clamp_min=-1.0e4, clamp_max=1.0e4)

    @staticmethod
    def _ensure_drift_column(c: torch.Tensor, d: torch.Tensor | None):
        if d is None:
            return torch.zeros(c.size(0), 1, dtype=c.dtype, device=c.device)
        if torch.is_tensor(d):
            d_val = d.to(device=c.device, dtype=c.dtype)
            if d_val.dim() == 0:
                d_val = d_val.view(1, 1).expand(c.size(0), 1)
            elif d_val.dim() == 1:
                d_val = d_val.view(-1, 1)
            elif d_val.dim() > 2:
                d_val = d_val.view(d_val.size(0), -1)[:, :1]
            if d_val.size(0) != c.size(0):
                if d_val.size(0) == 1:
                    d_val = d_val.expand(c.size(0), 1)
                else:
                    d_val = d_val[: c.size(0)]
            return d_val[:, :1]
        return torch.zeros(c.size(0), 1, dtype=c.dtype, device=c.device)

    def forward_old(self, c: torch.Tensor, d: torch.Tensor | None = None):
        """Return OLD-bank logits from question context and attention drift.

        Diagnostic prompt-only passes may omit ``d`` and use zero drift. The
        main forward supplies the measured drift to both routing branches.
        """

        d_col = self._ensure_drift_column(c, d)
        feat_old = torch.cat([c, d_col], dim=-1)  # [B, D+1]
        z_old = torch.einsum("bd,led->ble", feat_old, self.old_route_w) + self.old_route_b
        z_old = z_old / self.g_temp
        return self._sanitize_tensor(z_old, clamp_min=-60.0, clamp_max=60.0)

    def forward_new(self, c: torch.Tensor, d: torch.Tensor | None):
        d_col = self._ensure_drift_column(c, d)
        feat_new = torch.cat([c, d_col], dim=-1)  # [B, D+1]
        z_new = torch.einsum("bd,led->ble", feat_new, self.new_route_w) + self.new_route_b
        z_new = z_new / self.g_temp

        hidden = self.ctrl_mlp(feat_new)
        alpha = torch.tanh(self.alpha_head(hidden)).view(-1, self.late_layer_count, self.icr_rank)
        gamma = torch.sigmoid(self.gamma_head(hidden)).view(-1, self.late_layer_count, self.num_heads)
        lam = torch.sigmoid(self.lambda_head(hidden)).view(-1, 1)
        z_new = self._sanitize_tensor(z_new, clamp_min=-60.0, clamp_max=60.0)
        alpha = self._sanitize_tensor(alpha, clamp_min=-1.0, clamp_max=1.0)
        gamma = self._sanitize_tensor(gamma, clamp_min=0.0, clamp_max=1.0)
        lam = self._sanitize_tensor(lam, clamp_min=0.0, clamp_max=1.0)
        return z_new, alpha, gamma, lam

    def forward(self, c: torch.Tensor, d: torch.Tensor | None = None):
        z_old = self.forward_old(c, d)
        z_new, alpha, gamma, lam = self.forward_new(c, d)
        return {
            "z_old": z_old,
            "z_new": z_new,
            "alpha": alpha,
            "gamma": gamma,
            "lambda": lam,
        }
