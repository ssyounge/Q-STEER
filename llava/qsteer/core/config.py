from dataclasses import dataclass


@dataclass
class QSTEERConfig:
    # Preallocated expert bank size (E_max)
    expert_num: int = 32
    # Initially active expert slots per layer (E_init)
    expert_init: int = 8
    topk_update: int = 2
    late_layer_count: int = 6
    icr_rank: int = 8
    icr_scale: float = 0.3
    controller_hidden: int = 256
    controller_layers: int = 2
    controller_dropout: float = 0.0
    g_temp: float = 1.0
    # Prompt-only probe budget for each post-initial task.
    probe_samples: int = 128
    # Temporary probe slot count per candidate layer.
    probe_slots: int = 2
    # Promotion threshold coefficient beta_thr.
    beta_thr: float = 0.5
    # Candidate window from the controlled tail layers.
    expansion_candidate_last_n: int = 6
    # Strictly enforce question-only controller inputs.
    strict_q_only: bool = True
    # Two-stage refinement controls.
    drift_eps: float = 1e-8
    strict_gate_context: bool = True
    allow_uniform_gate_fallback: bool = False

    def __post_init__(self):
        self.expert_num = max(1, int(self.expert_num))
        self.expert_init = max(1, int(self.expert_init))
        if self.expert_init >= self.expert_num:
            raise ValueError(
                f"QSTEER requires expert_init < expert_num for probe expansion; "
                f"got expert_init={self.expert_init}, expert_num={self.expert_num}."
            )
        self.topk_update = max(1, int(self.topk_update))
        self.late_layer_count = max(1, int(self.late_layer_count))
        self.icr_rank = max(1, int(self.icr_rank))
        self.icr_scale = float(max(float(self.icr_scale), 0.0))
        self.controller_hidden = max(1, int(self.controller_hidden))
        self.controller_layers = max(1, int(self.controller_layers))
        self.controller_dropout = float(min(max(float(self.controller_dropout), 0.0), 0.99))
        self.g_temp = float(max(self.g_temp, 1e-6))
        self.probe_samples = max(1, int(self.probe_samples))
        self.probe_slots = max(1, int(self.probe_slots))
        self.beta_thr = float(self.beta_thr)
        self.expansion_candidate_last_n = max(1, int(self.expansion_candidate_last_n))
        self.strict_q_only = bool(self.strict_q_only)
        self.topk_update = min(self.topk_update, self.expert_num)
        self.drift_eps = float(max(float(self.drift_eps), 1e-12))
        self.strict_gate_context = bool(self.strict_gate_context)
        self.allow_uniform_gate_fallback = bool(self.allow_uniform_gate_fallback)
