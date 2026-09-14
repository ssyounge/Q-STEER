# Q-STEER Implementation Notes

This repository contains the core implementation of Q-STEER.

## Code Map

- `llava/qsteer/pipeline.py`  
  Two-stage diagnostic and main forward pipeline.

- `llava/qsteer/moe_lora.py`  
  Preallocated MoE-LoRA experts and routing.

- `llava/qsteer/core/controller.py`  
  Shared question-conditioned controller.

- `llava/qsteer/core/route_context.py`  
  Runtime routing context.

- `llava/qsteer/core/drift_buffer.py`  
  Attention-drift measurement and reference state.

- `llava/qsteer/core/attn_patch.py`  
  Attention-logit steering.

- `llava/qsteer/core/expansion.py`  
  Probe-based expert expansion.

- `llava/qsteer/core/task_finalize_cb.py`  
  Task-boundary expert and reference finalization.

## Execution Flow

```text
question-conditioned context
        ↓
prompt-only diagnostic pass
        ↓
attention-drift estimation
        ↓
shared controller
        ↓
MoE-LoRA routing + attention-logit steering
        ↓
main forward / backward
        ↓
task-boundary expert and reference update
```

The diagnostic pass is used to estimate routing and attention-drift signals before the main training pass. Attention-logit steering is applied to pre-softmax attention logits, while expert expansion is handled using preallocated expert slots.

For the complete method and experimental protocol, refer to:

**Attention-Logit Steering to Compositional Generalization for Continual VQA**  
Suyoung Yang, ECCV 2026  
DOI: `10.1007/978-3-032-37432-5_21`
