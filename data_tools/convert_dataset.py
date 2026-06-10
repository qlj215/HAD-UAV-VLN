"""
convert_dataset.py
==================
将 TravelUAV 原始数据集转换为 HAD 项目统一 JSONL 格式。

输入格式 (TravelUAV):
    raw_dataset/
    └── BrushifyCountryRoads/          # 环境名 = scene_id
        └── {uuid}/                    # 轨迹目录 = trajectory_id
            ├── frontcamera/           # 前视摄像头 (每5帧一张)
            │   ├── 000000.png
            │   ├── 000005.png
            │   └── ...
            ├── downcamera/            # 俯视摄像头 (每5帧一张)
            ├── log/                   # 每仿真帧的传感器数据
            │   ├── 000000.json
            │   └── ...
            ├── mark.json              # 目标物体 + 起止位置
            ├── merged_data.json       # 累积轨迹 [dx,dy,dz,roll,pitch,yaw]
            └── object_description.json # 导航指令文本

输出格式 (HAD JSONL):
    每行一个时间步样本，字段见框架文件 4.1 节

    高度分段规则:
        low:  altitude < 10 m
        mid:  10 m <= altitude < 30 m
        high: altitude >= 30 m
"""

"""
运行命令：
  python data_tools/convert_dataset.py \
    --raw_dir /root/autodl-tmp/TravelUAV_mini_dataset \
    --out_dir ./data/processed
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- 高度分段常量 ----
LOW_ALT_THRESHOLD = 10.0   # < 10m → low
MID_ALT_THRESHOLD = 30.0   # 10-30m → mid, >= 30m → high

# TravelUAV 中 merged_data.trajectory 的条目数 = 相机帧数
# (已对齐, 每个 entry 对应一个相机帧的累积位姿, 无需取子集)


def get_height_stage(altitude: float) -> str:
    """根据绝对高度返回高度分段标签。"""
    if altitude < LOW_ALT_THRESHOLD:
        return "low"
    elif altitude < MID_ALT_THRESHOLD:
        return "mid"
    else:
        return "high"


def load_json(filepath: Path) -> dict:
    """加载 JSON 文件。"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(samples: List[dict], output_path: Path) -> None:
    """将样本列表写入 JSONL 文件（每行一个 JSON 对象）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  [WRITE] {len(samples)} 条样本 -> {output_path}")


def convert_traveluav_trajectory(
    traj_dir: Path,
    scene_id: str,
    out_image_dir: Path,
    copy_images: bool = True,
) -> List[dict]:
    """将单条 TravelUAV 轨迹转换为 HAD 标准样本列表。

    Args:
        traj_dir: 轨迹目录路径 (e.g., .../BrushifyCountryRoads/{uuid}/)
        scene_id: 场景编号 (e.g., "BrushifyCountryRoads")
        out_image_dir: 输出图像根目录 (会创建 front/ 和 down/ 子目录)
        copy_images: 是否拷贝图像文件

    Returns:
        样本字典列表，每个字典对应一个时间步
    """
    traj_id = traj_dir.name

    # ---- 加载元数据 ----
    mark = load_json(traj_dir / "mark.json")
    obj_desc = load_json(traj_dir / "object_description.json")
    merged = load_json(traj_dir / "merged_data.json")

    # 指令文本: object_description 可能是列表 (取第一条) 或字符串
    instruction = obj_desc[0] if isinstance(obj_desc, list) else str(obj_desc)
    target_position = mark["target"]["position"]
    start_position = mark["start"]  # [x, y, z]

    # merged_data.trajectory: 每个仿真步的累积位移 [total_dx, total_dy, total_dz, roll, pitch, yaw]
    trajectory = merged.get("trajectory", [])
    if not trajectory:
        print(f"  [WARN] {traj_id}: merged_data.trajectory 为空, 跳过")
        return []

    # ---- 获取相机图像文件列表 ----
    front_dir = traj_dir / "frontcamera"
    down_dir = traj_dir / "downcamera"

    # 图像文件按帧号排序
    front_images = sorted(
        front_dir.glob("*.png"),
        key=lambda p: int(p.stem),  # 按帧号数值排序
    )
    down_images = sorted(
        down_dir.glob("*.png"),
        key=lambda p: int(p.stem),
    )

    if len(front_images) != len(down_images):
        print(
            f"  [WARN] {traj_id}: 前视({len(front_images)})与俯视({len(down_images)})图像数不一致, "
            f"将取较小值"
        )
    num_camera_frames = min(len(front_images), len(down_images))

    # ---- 准备输出图像目录 ----
    out_front_dir = out_image_dir / "front"
    out_down_dir = out_image_dir / "down"
    out_front_dir.mkdir(parents=True, exist_ok=True)
    out_down_dir.mkdir(parents=True, exist_ok=True)

    # ---- 逐相机帧构建样本 ----
    # merged_data.trajectory[i] 直接对应第 i 个相机帧的累积位姿
    samples = []
    for cam_idx in range(num_camera_frames):
        # 从 merged_data 获取此帧的累积位移和姿态 (直接按相机帧索引)
        tdata = trajectory[cam_idx]
        cum_dx, cum_dy, cum_dz = tdata[0], tdata[1], tdata[2]
        roll, pitch, yaw = tdata[3], tdata[4], tdata[5]

        # 绝对世界坐标 = 起点 + 累积位移
        abs_x = start_position[0] + cum_dx
        abs_y = start_position[1] + cum_dy
        abs_z = start_position[2] + cum_dz

        # 高度 = abs(z), 高度分段
        altitude = abs(abs_z)
        height_stage = get_height_stage(altitude)

        # 姿态: [x, y, z, roll, pitch, yaw]
        pose = [abs_x, abs_y, abs_z, roll, pitch, yaw]

        # ---- 动作: 到下一个相机帧的相对运动 ----
        if cam_idx < num_camera_frames - 1:
            next_tdata = trajectory[cam_idx + 1]
            next_cum_dx = next_tdata[0]
            next_cum_dy = next_tdata[1]
            next_cum_dz = next_tdata[2]
            next_yaw = next_tdata[5]

            action = [
                next_cum_dx - cum_dx,
                next_cum_dy - cum_dy,
                next_cum_dz - cum_dz,
                next_yaw - yaw,
            ]
            done = False
        else:
            # 最后一步: 动作为零
            action = [0.0, 0.0, 0.0, 0.0]
            done = True

        # ---- 样本 ID 与图像路径 ----
        sample_id = f"{scene_id}_{traj_id}_step{cam_idx:04d}"

        # 拷贝/链接图像
        if copy_images:
            front_src = front_images[cam_idx]
            down_src = down_images[cam_idx]
            front_dst = out_front_dir / f"{sample_id}.png"
            down_dst = out_down_dir / f"{sample_id}.png"
            if not front_dst.exists():
                shutil.copy2(front_src, front_dst)
            if not down_dst.exists():
                shutil.copy2(down_src, down_dst)

        sample = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "trajectory_id": traj_id,
            "step_id": cam_idx,
            "instruction": instruction,
            "front_image": f"images/front/{sample_id}.png",
            "down_image": f"images/down/{sample_id}.png",
            "pose": pose,
            "altitude": altitude,
            "height_stage": height_stage,
            "action": action,
            "target_position": target_position,
            "done": done,
        }

        samples.append(sample)

    return samples


def is_traveluav_trajectory_dir(dir_path: Path) -> bool:
    """判断一个目录是否为有效的 TravelUAV 轨迹目录。

    检查必要文件: mark.json, merged_data.json, object_description.json
    """
    required = ["mark.json", "merged_data.json", "object_description.json"]
    return all((dir_path / f).exists() for f in required)


def collect_trajectories(raw_dir: Path) -> Dict[str, List[Path]]:
    """扫描原始数据集，按场景(scene)收集所有轨迹目录。

    TravelUAV 结构: raw_dir/{scene_name}/{traj_uuid}/

    Returns:
        {scene_id: [traj_dir_path, ...]}
    """
    scene_trajs: Dict[str, List[Path]] = {}

    for scene_dir in sorted(raw_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        trajs = []
        for traj_dir in sorted(scene_dir.iterdir()):
            if traj_dir.is_dir() and is_traveluav_trajectory_dir(traj_dir):
                trajs.append(traj_dir)
        if trajs:
            scene_trajs[scene_id] = trajs

    return scene_trajs


def convert_dataset(
    raw_dir: str,
    out_dir: str,
    dataset_name: str = "traveluav",
    copy_images: bool = True,
    split_ratio: Optional[Tuple[float, float, float]] = None,
) -> None:
    """主转换流程。

    Args:
        raw_dir: TravelUAV 原始数据集根目录
        out_dir: 处理后数据输出目录
        dataset_name: 数据集名称 (用于日志)
        copy_images: 是否拷贝图像文件到输出目录
        split_ratio: (train, val_seen, val_unseen) 比例, 默认全部放入 train
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_image_dir = out_path / "images"

    print(f"[INFO] 数据集: {dataset_name}")
    print(f"[INFO] 原始目录: {raw_path}")
    print(f"[INFO] 输出目录: {out_path}")

    # 扫描轨迹
    scene_trajs = collect_trajectories(raw_path)
    if not scene_trajs:
        print("[ERROR] 未找到有效轨迹目录! 请检查 raw_dir 路径。")
        return

    total_trajs = sum(len(t) for t in scene_trajs.values())
    print(f"[INFO] 发现 {len(scene_trajs)} 个场景, {total_trajs} 条轨迹")

    # 收集所有样本 (按场景、轨迹顺序)
    all_samples: List[dict] = []
    scene_traj_counts: List[Tuple[str, str, int]] = []  # (scene_id, traj_id, count)

    for scene_id, traj_dirs in scene_trajs.items():
        print(f"\n[SCENE] {scene_id} ({len(traj_dirs)} trajectories)")
        for traj_dir in traj_dirs:
            samples = convert_traveluav_trajectory(
                traj_dir, scene_id, out_image_dir, copy_images
            )
            if samples:
                all_samples.extend(samples)
                scene_traj_counts.append(
                    (scene_id, traj_dir.name, len(samples))
                )
                print(
                    f"  [{traj_dir.name}] {len(samples)} samples "
                    f"(alt: {samples[0]['altitude']:.1f}m ~ {samples[-1]['altitude']:.1f}m, "
                    f"target: {samples[0]['target_position']})"
                )

    print(f"\n[INFO] 总计 {len(all_samples)} 条样本")

    # ---- 划分并写入 JSONL ----
    # 目标:
    #   1) val_seen 只能来自 train 见过的场景。
    #   2) 每个可切分的 seen 场景都同时出现在 train 和 val_seen。
    #   3) 场景数 >= 4 时留出完整场景给 val_unseen；场景更多时再留出 test。
    train_ratio = split_ratio[0] if split_ratio is not None else 0.7
    train_ratio = min(max(train_ratio, 0.0), 1.0)

    train_samples: List[dict] = []
    val_seen_samples: List[dict] = []
    val_unseen_samples: List[dict] = []
    test_samples: List[dict] = []

    scene_to_trajs: Dict[str, List[Tuple[str, int, int]]] = {}
    sample_offset = 0
    for scene_id, traj_id, count in scene_traj_counts:
        scene_to_trajs.setdefault(scene_id, []).append((traj_id, sample_offset, count))
        sample_offset += count

    scene_ids = sorted(scene_to_trajs)
    test_scenes: List[str] = []
    val_unseen_scenes: List[str] = []

    if len(scene_ids) >= 4:
        # 4 个场景时只留 1 个 val_unseen；场景更多时再分出 test。
        n_test_scenes = max(1, round(len(scene_ids) * 0.1)) if len(scene_ids) >= 6 else 0
        n_test_scenes = min(n_test_scenes, max(0, len(scene_ids) - 2))

        remaining_for_seen_and_unseen = len(scene_ids) - n_test_scenes
        n_val_unseen_scenes = max(1, round(len(scene_ids) * 0.2))
        n_val_unseen_scenes = min(n_val_unseen_scenes, max(0, remaining_for_seen_and_unseen - 1))

        if n_test_scenes > 0:
            test_scenes = scene_ids[-n_test_scenes:]
            candidate_scenes = scene_ids[:-n_test_scenes]
        else:
            candidate_scenes = scene_ids

        if n_val_unseen_scenes > 0:
            val_unseen_scenes = candidate_scenes[-n_val_unseen_scenes:]
            seen_scenes = candidate_scenes[:-n_val_unseen_scenes]
        else:
            seen_scenes = candidate_scenes
    else:
        seen_scenes = scene_ids

    def extend_records(target: List[dict], records: List[Tuple[str, int, int]]) -> None:
        for _, start, count in records:
            target.extend(all_samples[start:start + count])

    for scene_id in seen_scenes:
        records = scene_to_trajs[scene_id]
        if len(records) == 1:
            # 只有一条轨迹时无法在不重复样本的前提下同时放入 train/val_seen。
            extend_records(train_samples, records)
            print(f"  [WARN] {scene_id}: 只有 1 条轨迹, 无法切分 val_seen")
            continue

        n_train = int(len(records) * train_ratio)
        n_train = min(max(1, n_train), len(records) - 1)
        extend_records(train_samples, records[:n_train])
        extend_records(val_seen_samples, records[n_train:])

    for scene_id in val_unseen_scenes:
        extend_records(val_unseen_samples, scene_to_trajs[scene_id])

    for scene_id in test_scenes:
        extend_records(test_samples, scene_to_trajs[scene_id])

    # 写入文件
    out_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_samples, out_path / "train.jsonl")
    write_jsonl(val_seen_samples, out_path / "val_seen.jsonl")
    write_jsonl(val_unseen_samples, out_path / "val_unseen.jsonl")
    write_jsonl(test_samples, out_path / "test.jsonl")

    # ---- 输出统计摘要 ----
    train_scene_names = sorted({s["scene_id"] for s in train_samples})
    val_seen_scene_names = sorted({s["scene_id"] for s in val_seen_samples})
    val_unseen_scene_names = sorted({s["scene_id"] for s in val_unseen_samples})
    test_scene_names = sorted({s["scene_id"] for s in test_samples})

    print(f"\n{'='*60}")
    print(f"转换完成!")
    print(f"  train.jsonl:       {len(train_samples)} 条, scenes={train_scene_names}")
    print(f"  val_seen.jsonl:    {len(val_seen_samples)} 条, scenes={val_seen_scene_names}")
    print(f"  val_unseen.jsonl:  {len(val_unseen_samples)} 条, scenes={val_unseen_scene_names}")
    print(f"  test.jsonl:        {len(test_samples)} 条, scenes={test_scene_names}")
    print(f"  图像目录:          {out_image_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="将 TravelUAV 原始数据转换为 HAD 项目统一 JSONL 格式"
    )
    parser.add_argument(
        "--raw_dir", type=str, required=True,
        help="TravelUAV 原始数据集根目录 "
             "(e.g., /root/autodl-tmp/TravelUAV_mini_dataset)",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="处理后数据输出目录 "
             "(e.g., ./data/processed)",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="traveluav",
        help="数据集名称 (用于日志, 默认 traveluav)",
    )
    parser.add_argument(
        "--no_copy_images", action="store_true",
        help="不拷贝图像文件 (仅生成 JSONL 标注)",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7,
        help="训练集轨迹比例 (默认 0.7)",
    )
    args = parser.parse_args()

    convert_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
        copy_images=not args.no_copy_images,
        split_ratio=(args.train_ratio, max(0.0, 1.0 - args.train_ratio), 0.2),
    )


if __name__ == "__main__":
    main()
