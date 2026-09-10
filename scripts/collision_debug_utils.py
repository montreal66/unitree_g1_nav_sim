"""Shared helpers for Navigation V5 collision / pose-alignment debugging.

Import this module only after AppLauncher has started (needs Isaac/Omniverse).
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils  # noqa: E402 — requires Isaac Sim runtime


def log_pose_alignment(env, layout, env_id: int = 0, obstacle_index: int = 0) -> dict[str, float]:
    """Compare layout centers vs rigid-object collection poses for one active obstacle."""
    env_id = int(env_id)
    obstacle_index = int(obstacle_index)
    n_active = int(layout.num_active[env_id].item())
    if n_active == 0:
        print(f"[pose-align] env_{env_id}: no active obstacles")
        return {}

    if obstacle_index >= n_active:
        obstacle_index = 0

    slot_id = int(layout.active_slot_ids[env_id, obstacle_index].item())
    center_local = layout.centers_xy[env_id, obstacle_index]
    origin = env.scene.env_origins[env_id]
    layout_xy = origin[:2] + center_local
    layout_z = layout.heights[slot_id] * 0.5

    obstacles = env.scene[layout.cfg.obstacle_asset_name]
    sim_pos = obstacles.data.object_pos_w[env_id, slot_id]
    object_name = obstacles.object_names[slot_id] if slot_id < len(obstacles.object_names) else "?"

    slot_type = int(layout.slot_type[slot_id].item())
    footprint = float(layout.footprint_radius[slot_id].item())

    delta_xy = (sim_pos[:2] - layout_xy).detach().cpu().tolist()
    delta_z = float((sim_pos[2] - layout_z).item())
    xy_err = float(torch.linalg.norm(sim_pos[:2] - layout_xy).item())

    print(f"[pose-align] env_{env_id} obstacle_index={obstacle_index} slot_id={slot_id} name={object_name}")
    print(f"  slot_type={slot_type} footprint_radius={footprint:.3f} height_z={layout_z:.3f}")
    print(f"  layout center (world): xy=({layout_xy[0]:.3f}, {layout_xy[1]:.3f}) z={layout_z:.3f}")
    print(f"  sim object_pos_w     : xy=({sim_pos[0]:.3f}, {sim_pos[1]:.3f}) z={sim_pos[2]:.3f}")
    print(f"  delta                : dxy={delta_xy} dz={delta_z:.4f} |dxy|={xy_err:.4f}")

    if xy_err > 0.05 or abs(delta_z) > 0.05:
        print("  WARNING: layout vs sim pose mismatch (>5 cm) — ghost-through may be pose desync.")
    else:
        print("  OK: layout and sim poses agree within 5 cm.")

    return {"xy_err": xy_err, "dz_err": abs(delta_z), "slot_id": float(slot_id)}


def place_robot_toward_obstacle(
    env,
    layout,
    env_id: int = 0,
    obstacle_index: int = 0,
    standoff_m: float = 3.0,
) -> torch.Tensor:
    """Teleport the robot standoff_m away from an obstacle, facing it. Returns target xy (world)."""
    env_id = int(env_id)
    robot = env.scene["robot"]
    origin = env.scene.env_origins[env_id]
    slot_id = int(layout.active_slot_ids[env_id, obstacle_index].item())
    obstacle_local = layout.centers_xy[env_id, obstacle_index]
    target_xy = origin[:2] + obstacle_local

    # Approach from -X in env-local frame (world offset depends on env origin only).
    start_local = obstacle_local - torch.tensor([standoff_m, 0.0], device=env.device)
    start_xy = origin[:2] + start_local
    yaw = torch.atan2(obstacle_local[1] - start_local[1], obstacle_local[0] - start_local[0])
    quat = math_utils.quat_from_euler_xyz(
        torch.zeros(1, device=env.device),
        torch.zeros(1, device=env.device),
        yaw.unsqueeze(0),
    )[0]

    root_pose = torch.zeros(7, device=env.device)
    root_pose[0] = start_xy[0]
    root_pose[1] = start_xy[1]
    root_pose[2] = robot.data.default_root_state[env_id, 2]
    root_pose[3:7] = quat
    env_ids = torch.tensor([env_id], device=env.device)
    robot.write_root_pose_to_sim(root_pose.unsqueeze(0), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_ids)
    return target_xy


def velocity_cmd_toward_xy(robot, target_xy_w: torch.Tensor, forward_speed: float = 1.0) -> torch.Tensor:
    """Body-frame (vx, vy, yaw_rate) command pointing at target_xy_w."""
    pos_xy = robot.data.root_pos_w[:, :2]
    delta = target_xy_w.unsqueeze(0) - pos_xy
    dist = torch.linalg.norm(delta, dim=1, keepdim=True).clamp(min=1e-6)
    dir_w = delta / dist

    heading = robot.data.heading_w.squeeze(-1)
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    vx = dir_w[:, 0] * cos_h + dir_w[:, 1] * sin_h
    vy = -dir_w[:, 0] * sin_h + dir_w[:, 1] * cos_h

    yaw_err = torch.atan2(dir_w[:, 1], dir_w[:, 0]) - heading
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))

    cmd = torch.zeros(robot.num_instances, 3, device=robot.device)
    speed = min(forward_speed, 1.0)
    cmd[:, 0] = vx * speed
    cmd[:, 1] = vy * speed
    cmd[:, 2] = torch.clamp(yaw_err, -0.5, 0.5)
    return cmd


def max_non_foot_contact_norm(env, env_id: int = 0) -> float:
    """Max contact force norm on robot links excluding ankle roll feet."""
    sensor = env.scene.sensors["contact_forces"]
    forces = sensor.data.net_forces_w[env_id]
    body_names = sensor.body_names
    max_norm = 0.0
    for i, name in enumerate(body_names):
        if "ankle_roll" in name:
            continue
        norm = float(torch.linalg.norm(forces[i]).item())
        max_norm = max(max_norm, norm)
    return max_norm


def interpret_collision_sample(soft_penalty: float, contact_norm: float, xy_err: float) -> str:
    """Map logged metrics to the plan's interpretation table."""
    if soft_penalty > 0.1 and contact_norm < 1.0:
        if xy_err > 0.05:
            return "LIKELY pose desync (soft zone sees obstacle, sim pose offset, no contact)"
        return "LIKELY physics broken (soft zone active, negligible non-foot contact)"
    if soft_penalty > 0.1 and contact_norm >= 1.0:
        return "Physics OK (contact on non-foot links); policy may exploit gaps if ghosting in play"
    if soft_penalty <= 0.1 and contact_norm < 1.0:
        return "Robot not in soft zone (or layout inactive); drive longer or check placement"
    return "In soft zone with contact — expected near obstacle surface"


def print_step_log(step: int, root_pos, soft_penalty: float, contact_norm: float, xy_err: float):
    interp = interpret_collision_sample(soft_penalty, contact_norm, xy_err)
    print(
        f"step={step:4d}  pos=({root_pos[0]:.2f},{root_pos[1]:.2f},{root_pos[2]:.2f})  "
        f"soft={soft_penalty:.3f}  contact={contact_norm:.2f}N  |dxy_layout_sim|={xy_err:.4f}  -> {interp}"
    )
