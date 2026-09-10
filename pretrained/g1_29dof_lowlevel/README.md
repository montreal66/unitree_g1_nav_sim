# Pretrained G1 29-DoF Low-Level Locomotion Policy

This directory contains the frozen low-level policy used by all hierarchical
navigation (high-level planner) tasks.

| Field | Value |
|-------|-------|
| Task | `Unitree-G1-29dof-LowLevel` (originally trained as Sequential curriculum) |
| Format | TorchScript (`policy.pt`) and ONNX (`policy.onnx`) |
| Action | 29-DoF joint position targets |
| Command input | `(lin_vel_x, lin_vel_y, ang_vel_z)` velocity commands |

## Files

- `policy.pt` — TorchScript export loaded by `PreTrainedPolicyAction` at training/play time
- `policy.onnx` — ONNX export (optional inspection / sim2real)
- `params/agent.yaml` — RSL-RL agent config used during training
- `params/env.yaml` — environment config snapshot from the training run

## Override

By default, navigation tasks resolve this path relative to the repo root.
You can override it with:

```bash
export UNITREE_G1_LOW_LEVEL_POLICY_PATH=/path/to/your/policy.pt
```

Students are **not** required to retrain this policy for the homework.
It is provided so that high-level navigation training can run out of the box.
Interested students may optionally train their own low-level policy via
`Unitree-G1-29dof-LowLevel` and point navigation at the exported JIT with the
env var above.
