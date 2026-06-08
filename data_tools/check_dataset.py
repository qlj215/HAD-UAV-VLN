"""
check_dataset.py
================
检查 HAD 处理后数据集的完整性、一致性，并生成详细统计报告。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  功能一览
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  【数据校验】 (默认开启)
    1. 字段存在性与类型  ─ 检查 13 个必需字段 + 类型匹配
    2. 值域约束          ─ height_stage 合法性 / 与 altitude 一致性 /
                           pose(6) action(4) target_position(3) 长度
    3. 轨迹内部一致性    ─ step_id 连续性 / done 仅最后一步为 True /
                           最后一步 action 全零
    4. 图像文件存在性    ─ 检查 front/down 图像文件 (--check_images)
    5. 跨文件重复检查    ─ 无 sample_id 在不同 split 间重复

  【统计分析】 (默认开启, 可用 --no_stats 跳过)
    6. 高度分析          ─ 每个 split 的 low/mid/high 分布
                          各轨迹是否跨多个高度段 (混合高度轨迹识别)
                          每条轨迹的高度变化曲线数据
    7. 场景分析          ─ 场景数量 / 每场景轨迹数 / 每场景样本数
                          场景在各 split 间的分布 (是否存在场景泄露)
    8. 轨迹分析          ─ 轨迹长度分布 (min/max/mean/std)
                          轨迹级统计: 总位移 / 平均高度 / 动作幅度
    9. 动作分析          ─ dx,dy,dz,dyaw 各维度的分布统计
                          水平位移 vs 垂直位移的散点数据
                          方向偏好分析 (前后/左右/升降)
    10. 指令分析         ─ 指令长度分布 (字符数/词数)
                           指令多样性 (唯一指令数 vs 总轨迹数)

  【可视化】 (--plot 开启)
    11. 图表生成         ─ 高度分布饼图 / 轨迹长度直方图
                           动作幅度分布直方图 / 高度变化曲线
                           场景-轨迹统计柱状图 / 高度混合桑基图数据

  【导出】 (--export_stats 开启)
    12. JSON 统计导出    ─ 将所有统计数据导出为 JSON, 便于外部工具做图

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # 基础校验 + 统计 (最常用)
  python data_tools/check_dataset.py --data_dir ./data/processed

  # 同时检查图像文件
  python data_tools/check_dataset.py --data_dir ./data/processed --check_images

  # 生成可视化图表
  python data_tools/check_dataset.py --data_dir ./data/processed --plot --output_dir ./outputs/figures

  # 导出统计 JSON (供外部工具/notebook 使用)
  python data_tools/check_dataset.py --data_dir ./data/processed --export_stats --output_dir ./outputs/results

  # 完整检查: 校验 + 图像 + 图表 + 导出
  python data_tools/check_dataset.py --data_dir ./data/processed \
      --check_images --plot --export_stats --output_dir ./outputs/analysis

  # 只检查特定 split
  python data_tools/check_dataset.py --data_dir ./data/processed --splits train.jsonl val_seen.jsonl

  # 严格模式 (空文件也报错)
  python data_tools/check_dataset.py --data_dir ./data/processed --strict
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

# ---- 可选依赖 ----
try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互后端, 仅保存文件
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---- 字段规范 ----
REQUIRED_FIELDS = {
    "sample_id": str,
    "scene_id": str,
    "trajectory_id": str,
    "step_id": int,
    "instruction": str,
    "front_image": str,
    "down_image": str,
    "pose": list,
    "altitude": (int, float),
    "height_stage": str,
    "action": list,
    "target_position": list,
    "done": bool,
}

VALID_HEIGHT_STAGES = {"low", "mid", "high"}
POSE_LENGTH = 6
ACTION_LENGTH = 4
TARGET_POS_LENGTH = 3

# 高度分段阈值 (与 convert_dataset.py 保持一致)
LOW_ALT_MAX = 10.0
MID_ALT_MAX = 30.0


# ================================================================
# 工具函数
# ================================================================

def load_jsonl(filepath: Path) -> List[dict]:
    """加载 JSONL 文件。解析失败时记录并跳过该行。"""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [ERROR] {filepath.name}:{line_no} JSON 解析失败: {e}")
    return samples


def group_by_trajectory(samples: List[dict]) -> Dict[str, List[dict]]:
    """将样本按 trajectory_id 分组，每组内按 step_id 排序。"""
    groups = defaultdict(list)
    for s in samples:
        groups[s.get("trajectory_id", "unknown")].append(s)
    return {tid: sorted(steps, key=lambda x: x.get("step_id", 0))
            for tid, steps in groups.items()}


def group_by_scene(samples: List[dict]) -> Dict[str, List[dict]]:
    """将样本按 scene_id 分组。"""
    groups = defaultdict(list)
    for s in samples:
        groups[s.get("scene_id", "unknown")].append(s)
    return dict(groups)


# ================================================================
# 检查 1: 字段存在性与类型
# ================================================================

def check_fields(samples: List[dict], split_name: str) -> List[str]:
    """检查每条样本的必需字段和类型。"""
    errors = []
    for i, s in enumerate(samples):
        sid = s.get("sample_id", f"index_{i}")
        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in s:
                errors.append(f"[{split_name}] {sid}: 缺少字段 '{field}'")
                continue
            value = s[field]
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    errors.append(
                        f"[{split_name}] {sid}: 字段 '{field}' 类型错误 "
                        f"(期望 {expected_type}, 实际 {type(value).__name__})"
                    )
            else:
                if not isinstance(value, expected_type):
                    errors.append(
                        f"[{split_name}] {sid}: 字段 '{field}' 类型错误 "
                        f"(期望 {expected_type.__name__}, 实际 {type(value).__name__})"
                    )
    return errors


# ================================================================
# 检查 2: 值域约束
# ================================================================

def check_value_constraints(samples: List[dict], split_name: str) -> List[str]:
    """检查字段值是否在允许范围内。"""
    errors = []
    for s in samples:
        sid = s.get("sample_id", "unknown")
        stage = s.get("height_stage", "")
        alt = s.get("altitude", -1)

        if stage not in VALID_HEIGHT_STAGES:
            errors.append(
                f"[{split_name}] {sid}: height_stage='{stage}' 不合法"
            )

        if stage == "low" and alt >= LOW_ALT_MAX:
            errors.append(
                f"[{split_name}] {sid}: stage='low' 但 altitude={alt:.1f} >= {LOW_ALT_MAX}"
            )
        elif stage == "mid" and not (LOW_ALT_MAX <= alt < MID_ALT_MAX):
            errors.append(
                f"[{split_name}] {sid}: stage='mid' 但 altitude={alt:.1f} "
                f"不在 [{LOW_ALT_MAX}, {MID_ALT_MAX})"
            )
        elif stage == "high" and alt < MID_ALT_MAX:
            errors.append(
                f"[{split_name}] {sid}: stage='high' 但 altitude={alt:.1f} < {MID_ALT_MAX}"
            )

        pose = s.get("pose", [])
        if len(pose) != POSE_LENGTH:
            errors.append(f"[{split_name}] {sid}: pose 长度={len(pose)}")

        action = s.get("action", [])
        if len(action) != ACTION_LENGTH:
            errors.append(f"[{split_name}] {sid}: action 长度={len(action)}")

        tp = s.get("target_position", [])
        if len(tp) != TARGET_POS_LENGTH:
            errors.append(f"[{split_name}] {sid}: target_position 长度={len(tp)}")

        if s.get("step_id", -1) < 0:
            errors.append(f"[{split_name}] {sid}: step_id 为负数")

    return errors


# ================================================================
# 检查 3: 轨迹内部一致性
# ================================================================

def check_trajectory_consistency(
    samples: List[dict], split_name: str
) -> Tuple[List[str], Dict[str, List[dict]]]:
    """检查轨迹一致性, 返回 (errors, 按 trajectory_id 分组的样本)。"""
    errors = []
    traj_groups = group_by_trajectory(samples)

    for tid, steps in traj_groups.items():
        step_ids = [s["step_id"] for s in steps]
        n = len(steps)
        expected = list(range(n))

        if step_ids != expected:
            errors.append(
                f"[{split_name}] {tid}: step_id 不连续, "
                f"前5期望={expected[:5]}, 实际={step_ids[:5]}"
            )

        done_flags = [s.get("done", False) for s in steps]
        n_done = sum(done_flags)
        if n_done != 1:
            errors.append(
                f"[{split_name}] {tid}: done=True 出现 {n_done} 次"
            )
        elif not done_flags[-1]:
            errors.append(
                f"[{split_name}] {tid}: done=True 不在最后一步"
            )

        last_action = steps[-1].get("action", [])
        if any(a != 0.0 for a in last_action):
            errors.append(
                f"[{split_name}] {tid}: 最后一步 action 非零: {last_action}"
            )

    return errors, traj_groups


# ================================================================
# 检查 4: 图像文件存在性
# ================================================================

def check_images(
    samples: List[dict],
    data_dir: Path,
    split_name: str,
    max_errors: int = 10,
) -> List[str]:
    """检查 front_image 和 down_image 指向的文件是否存在。"""
    errors = []
    missing = 0
    for s in samples:
        for field in ["front_image", "down_image"]:
            full_path = data_dir / s.get(field, "")
            if not full_path.exists():
                missing += 1
                if len(errors) < max_errors:
                    errors.append(
                        f"[{split_name}] {s.get('sample_id', '?')}: "
                        f"{field} 缺失: {full_path}"
                    )
    if missing > max_errors:
        errors.append(
            f"[{split_name}] ...还有 {missing - max_errors} 个缺失, 共 {missing}"
        )
    return errors


# ================================================================
# 检查 5: 跨文件重复
# ================================================================

def check_duplicates(all_splits: Dict[str, List[dict]]) -> List[str]:
    """检查不同 split 之间是否有重复的 sample_id。"""
    errors = []
    seen = {}
    for split_name, samples in all_splits.items():
        for s in samples:
            sid = s.get("sample_id", "")
            if not sid:
                continue
            if sid in seen:
                errors.append(
                    f"sample_id '{sid}' 同时出现在 "
                    f"'{seen[sid]}' 和 '{split_name}' 中"
                )
            seen[sid] = split_name
    return errors


# ================================================================
#  统计分析 (新增)
# ================================================================

def analyze_heights(
    samples: List[dict],
    traj_groups: Dict[str, List[dict]],
    split_name: str,
) -> dict:
    """高度维度深度分析。

    Returns:
        { stage_distribution, mixed_height_trajectories, per_traj_altitude_stats,
          altitude_values (for histogram), stage_transitions }
    """
    stats = {}

    # 6.1 高度分段整体分布
    stage_counter = Counter(s.get("height_stage") for s in samples)
    stats["stage_distribution"] = dict(stage_counter)

    # 6.2 轨迹内高度混合分析: 哪些轨迹跨越了多个高度段
    mixed_trajs = []
    single_stage_trajs = defaultdict(list)
    for tid, steps in traj_groups.items():
        stages_in_traj = set(s["height_stage"] for s in steps)
        if len(stages_in_traj) > 1:
            mixed_trajs.append({
                "trajectory_id": tid,
                "stages": sorted(stages_in_traj),
                "num_steps": len(steps),
                "scene_id": steps[0].get("scene_id", ""),
            })
        else:
            single_stage_trajs[list(stages_in_traj)[0]].append(tid)

    stats["mixed_height_trajectories"] = mixed_trajs
    stats["num_mixed_trajs"] = len(mixed_trajs)
    stats["num_single_stage_trajs"] = {
        stage: len(tids) for stage, tids in single_stage_trajs.items()
    }

    # 6.3 每条轨迹的高度统计 (min/max/mean/std/start/end altitude)
    per_traj_alt = []
    altitude_all = []
    for tid, steps in traj_groups.items():
        alts = [s["altitude"] for s in steps]
        altitude_all.extend(alts)
        per_traj_alt.append({
            "trajectory_id": tid,
            "scene_id": steps[0].get("scene_id", ""),
            "num_steps": len(steps),
            "alt_min": min(alts),
            "alt_max": max(alts),
            "alt_mean": sum(alts) / len(alts),
            "alt_start": alts[0],
            "alt_end": alts[-1],
            "alt_range": max(alts) - min(alts),
            "stages_present": sorted(set(s["height_stage"] for s in steps)),
        })
    stats["per_trajectory_altitude"] = per_traj_alt
    stats["altitude_values"] = altitude_all  # 用于直方图

    # 6.4 高度变化趋势: 上升/下降/平稳 轨迹数
    n_ascending = sum(1 for t in per_traj_alt if t["alt_end"] > t["alt_start"] + 1.0)
    n_descending = sum(1 for t in per_traj_alt if t["alt_start"] > t["alt_end"] + 1.0)
    n_stable = len(per_traj_alt) - n_ascending - n_descending
    stats["altitude_trend"] = {
        "ascending": n_ascending,
        "descending": n_descending,
        "stable": n_stable,
    }

    return stats


# ================================================================
#  分析 7: 高度与位置数值分析 (新增)
# ================================================================

def compute_percentiles(
    values: List[float],
    percentiles: List[int] = None,
) -> Dict[int, float]:
    """计算数值列表的分位数。

    Args:
        values: 数值列表
        percentiles: 要计算的分位点列表, 默认 [0,5,10,25,50,75,90,95,100]

    Returns:
        {percentile: value}
    """
    if percentiles is None:
        percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]

    if not values:
        return {p: 0.0 for p in percentiles}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    result = {}
    for p in percentiles:
        if p == 0:
            result[p] = sorted_vals[0]
        elif p == 100:
            result[p] = sorted_vals[-1]
        else:
            k = (p / 100.0) * (n - 1)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                result[p] = sorted_vals[int(k)]
            else:
                result[p] = sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)
    return result


def build_histogram_bins(
    values: List[float],
    num_bins: int = 20,
) -> Dict:
    """构建直方图数据 (等宽分箱)。

    Returns:
        { bin_edges, bin_centers, counts, density }
    """
    if not values:
        return {"bin_edges": [], "bin_centers": [], "counts": [], "density": []}

    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    bin_width = (vmax - vmin) / num_bins
    bin_edges = [vmin + i * bin_width for i in range(num_bins + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]
    counts = [0] * num_bins

    for v in values:
        for i in range(num_bins):
            if i == num_bins - 1:
                if bin_edges[i] <= v <= bin_edges[i + 1]:
                    counts[i] += 1
                    break
            else:
                if bin_edges[i] <= v < bin_edges[i + 1]:
                    counts[i] += 1
                    break

    total = sum(counts)
    density = [c / max(total * bin_width, 1e-8) for c in counts]

    return {
        "bin_edges": [round(e, 2) for e in bin_edges],
        "bin_centers": [round(c, 2) for c in bin_centers],
        "counts": counts,
        "density": [round(d, 6) for d in density],
        "num_bins": num_bins,
        "range": [round(vmin, 2), round(vmax, 2)],
    }


def suggest_thresholds(
    altitude_values: List[float],
    current_low: float = LOW_ALT_MAX,
    current_mid: float = MID_ALT_MAX,
) -> Dict:
    """基于数据分布建议高度分段阈值。

    提供四种方案:
    1. equal_frequency  ─ 三等分样本数 (每段约 1/3 样本)
    2. equal_range       ─ 三等分高度范围
    3. percentile_33_66  ─ 第 33 和 66 百分位
    4. natural_breaks    ─ 基于直方图谷值的自然断点

    Args:
        altitude_values: 所有样本的 altitude 值
        current_low: 当前 low/mid 阈值
        current_mid: 当前 mid/high 阈值

    Returns:
        四种建议方案及当前方案的数据分布对比
    """
    if not altitude_values:
        return {}

    sorted_vals = sorted(altitude_values)
    n = len(sorted_vals)
    vmin, vmax = sorted_vals[0], sorted_vals[-1]

    # 方案1: 等频 (每个高度段约 n/3 个样本)
    idx_33 = n // 3
    idx_66 = 2 * n // 3
    eq_freq_low = round(sorted_vals[idx_33], 1)
    eq_freq_high = round(sorted_vals[idx_66], 1)

    # 方案2: 等距
    rng = vmax - vmin
    eq_range_low = round(vmin + rng / 3, 1)
    eq_range_high = round(vmin + 2 * rng / 3, 1)

    # 方案3: 33 和 66 百分位
    pcts = compute_percentiles(altitude_values, [33, 66])
    pct_low = round(pcts[33], 1)
    pct_high = round(pcts[66], 1)

    # 方案4: 自然断点 (直方图谷值)
    hist = build_histogram_bins(altitude_values, num_bins=30)
    natural_breaks = _find_histogram_valleys(hist)

    # 计算各方案下的分布
    def count_by_thresholds(lo, hi):
        cnt_low = sum(1 for v in altitude_values if v < lo)
        cnt_mid = sum(1 for v in altitude_values if lo <= v < hi)
        cnt_high = sum(1 for v in altitude_values if v >= hi)
        return {
            "low": cnt_low,
            "mid": cnt_mid,
            "high": cnt_high,
            "low_pct": round(100 * cnt_low / n, 1),
            "mid_pct": round(100 * cnt_mid / n, 1),
            "high_pct": round(100 * cnt_high / n, 1),
        }

    suggestions = {
        "data_range": {"min": round(vmin, 1), "max": round(vmax, 1), "n": n},
        "current": {
            "threshold_low": current_low,
            "threshold_high": current_mid,
            "distribution": count_by_thresholds(current_low, current_mid),
        },
        "equal_frequency": {
            "description": "每段约 1/3 样本数",
            "threshold_low": eq_freq_low,
            "threshold_high": eq_freq_high,
            "distribution": count_by_thresholds(eq_freq_low, eq_freq_high),
        },
        "equal_range": {
            "description": "三等分高度范围",
            "threshold_low": eq_range_low,
            "threshold_high": eq_range_high,
            "distribution": count_by_thresholds(eq_range_low, eq_range_high),
        },
        "percentile_33_66": {
            "description": "第 33 / 66 百分位",
            "threshold_low": pct_low,
            "threshold_high": pct_high,
            "distribution": count_by_thresholds(pct_low, pct_high),
        },
    }

    if natural_breaks and len(natural_breaks) >= 2:
        nb_low = natural_breaks[0]
        nb_high = natural_breaks[1]
        suggestions["natural_breaks"] = {
            "description": "直方图谷值 (自然断点)",
            "threshold_low": nb_low,
            "threshold_high": nb_high,
            "distribution": count_by_thresholds(nb_low, nb_high),
        }

    return suggestions


def _find_histogram_valleys(hist: Dict) -> List[float]:
    """在直方图中寻找谷值位置 (局部极小值点)。"""
    counts = hist["counts"]
    edges = hist["bin_edges"]
    if len(counts) < 5:
        return []

    valleys = []
    for i in range(1, len(counts) - 1):
        # 谷值: 比左右邻居都低
        if counts[i] < counts[i - 1] and counts[i] < counts[i + 1]:
            valley_pos = (edges[i] + edges[i + 1]) / 2
            valleys.append(round(valley_pos, 1))

    return valleys


def analyze_height_position_numerical(
    samples: List[dict],
    traj_groups: Dict[str, List[dict]],
    split_name: str,
) -> dict:
    """高度和位置的数值分析 (不限于三段式分类)。

    提供全量 altitude 和 position 的详细数值分布,
    以及多条阈值建议, 为后续调整 height_stage 分档提供数据依据。

    Returns:
        { altitude_numerical, position_numerical, threshold_suggestions,
          per_trajectory_boxplot_data }
    """
    stats = {}

    # 收集所有数值
    all_altitudes = [s["altitude"] for s in samples]
    all_pose_x = [s["pose"][0] for s in samples]
    all_pose_y = [s["pose"][1] for s in samples]
    all_pose_z = [s["pose"][2] for s in samples]

    # ---- 7.1 高度数值分析 ----
    alt_pcts = compute_percentiles(all_altitudes)
    alt_hist = build_histogram_bins(all_altitudes, num_bins=30)

    # altitude 统计量
    n = len(all_altitudes)
    alt_mean = sum(all_altitudes) / n if n > 0 else 0
    alt_var = sum((v - alt_mean) ** 2 for v in all_altitudes) / n if n > 0 else 0
    alt_std = math.sqrt(alt_var)

    altitude_numerical = {
        "count": n,
        "mean": round(alt_mean, 2),
        "std": round(alt_std, 2),
        "skewness": round(
            sum(((v - alt_mean) / max(alt_std, 1e-8)) ** 3 for v in all_altitudes) / max(n, 1),
            3,
        ) if alt_std > 0 else 0.0,
        "percentiles": alt_pcts,
        "histogram": alt_hist,
        "range": round(alt_pcts[100] - alt_pcts[0], 2) if n > 0 else 0,
        "iqr": round(alt_pcts[75] - alt_pcts[25], 2) if n > 0 else 0,
    }
    stats["altitude_numerical"] = altitude_numerical

    # ---- 7.2 位置数值分析 ----
    def pos_stats(values, name):
        pcts = compute_percentiles(values)
        n_v = len(values)
        mean_v = sum(values) / n_v if n_v > 0 else 0
        var_v = sum((v - mean_v) ** 2 for v in values) / n_v if n_v > 0 else 0
        return {
            "name": name,
            "count": n_v,
            "mean": round(mean_v, 2),
            "std": round(math.sqrt(var_v), 2),
            "percentiles": pcts,
            "range": round(pcts[100] - pcts[0], 2) if n_v > 0 else 0,
            "histogram": build_histogram_bins(values, num_bins=20),
        }

    position_numerical = {
        "x": pos_stats(all_pose_x, "X (世界坐标)"),
        "y": pos_stats(all_pose_y, "Y (世界坐标)"),
        "z": pos_stats(all_pose_z, "Z (世界坐标, 向上)"),
    }
    stats["position_numerical"] = position_numerical

    # ---- 7.3 每条轨迹的箱线图数据 (5-number summary) ----
    per_traj_boxplot = []
    for tid, steps in traj_groups.items():
        alts = [s["altitude"] for s in steps]
        if not alts:
            continue
        pcts = compute_percentiles(alts, [0, 25, 50, 75, 100])
        per_traj_boxplot.append({
            "trajectory_id": tid,
            "scene_id": steps[0].get("scene_id", ""),
            "num_steps": len(alts),
            "min": pcts[0],
            "q1": pcts[25],
            "median": pcts[50],
            "q3": pcts[75],
            "max": pcts[100],
            "mean": round(sum(alts) / len(alts), 2),
        })
    stats["per_trajectory_boxplot"] = per_traj_boxplot

    # ---- 7.4 阈值建议 ----
    stats["threshold_suggestions"] = suggest_thresholds(all_altitudes)

    return stats


def analyze_scenes(
    samples: List[dict],
    traj_groups: Dict[str, List[dict]],
    split_name: str,
) -> dict:
    """场景维度分析。

    Returns:
        { num_scenes, scenes_detail: [{scene_id, num_trajs, num_samples, ...}], ... }
    """
    stats = {}

    # 按场景分组
    scene_samples = group_by_scene(samples)

    # 按场景统计轨迹
    scene_trajs = defaultdict(set)
    for s in samples:
        scene_trajs[s.get("scene_id", "unknown")].add(s.get("trajectory_id", ""))

    scenes_detail = []
    for scene_id in sorted(scene_samples.keys()):
        traj_ids = scene_trajs[scene_id]
        scene_samps = scene_samples[scene_id]
        traj_lens = []
        for tid in traj_ids:
            traj_lens.append(sum(1 for s in scene_samps if s.get("trajectory_id") == tid))

        scenes_detail.append({
            "scene_id": scene_id,
            "num_trajectories": len(traj_ids),
            "num_samples": len(scene_samps),
            "trajectory_ids": sorted(traj_ids),
            "trajectory_lengths": traj_lens,
            "avg_traj_length": sum(traj_lens) / len(traj_lens) if traj_lens else 0,
        })

    stats["num_scenes"] = len(scene_samples)
    stats["scenes_detail"] = scenes_detail

    # 场景-轨迹分布矩阵数据 (用于跨 split 比较)
    stats["scene_traj_matrix"] = {
        scene_id: {
            "num_trajs": len(traj_ids),
            "num_samples": len(scene_samples[scene_id]),
        }
        for scene_id, traj_ids in scene_trajs.items()
    }

    return stats


def analyze_trajectories(
    samples: List[dict],
    traj_groups: Dict[str, List[dict]],
    split_name: str,
) -> dict:
    """轨迹维度分析。

    Returns:
        { length_distribution, lengths, total_displacement, ... }
    """
    stats = {}

    lengths = [len(steps) for steps in traj_groups.values()]
    stats["num_trajectories"] = len(lengths)
    stats["length_values"] = lengths  # 用于直方图
    if lengths:
        stats["length_stats"] = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "std": (
                (sum((l - sum(lengths) / len(lengths)) ** 2 for l in lengths) / len(lengths)) ** 0.5
                if len(lengths) > 1 else 0.0
            ),
            "total_steps": sum(lengths),
        }

    # 每条轨迹的总位移
    displacements = []
    for tid, steps in traj_groups.items():
        if len(steps) < 2:
            continue
        start_pose = steps[0]["pose"]
        end_pose = steps[-1]["pose"]
        dx = end_pose[0] - start_pose[0]
        dy = end_pose[1] - start_pose[1]
        dz = end_pose[2] - start_pose[2]
        h_dist = math.sqrt(dx ** 2 + dy ** 2)
        total_dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        displacements.append({
            "trajectory_id": tid,
            "horizontal_displacement": h_dist,
            "vertical_displacement": abs(dz),
            "total_displacement_3d": total_dist,
            "num_steps": len(steps),
        })
    stats["displacements"] = displacements

    return stats


def analyze_actions(
    samples: List[dict],
    traj_groups: Dict[str, List[dict]],
    split_name: str,
) -> dict:
    """动作维度分析。

    Returns:
        { per_dimension_stats, horizontal_vs_vertical, direction_preference }
    """
    stats = {}

    # 排除最后一步 (done=True, action 全零)
    actions = [s["action"] for s in samples if not s.get("done")]
    if not actions:
        return stats

    # 9.1 各维度统计
    dims = ["dx", "dy", "dz", "dyaw"]
    dim_values = {
        dim: [a[i] for a in actions] for i, dim in enumerate(dims)
    }
    per_dim = {}
    for dim, vals in dim_values.items():
        per_dim[dim] = {
            "mean": sum(vals) / len(vals),
            "std": (sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals)) ** 0.5,
            "min": min(vals),
            "max": max(vals),
            "values": vals,  # 用于直方图
        }
    stats["per_dimension"] = per_dim

    # 9.2 水平位移 vs 垂直位移
    h_disp = [math.sqrt(a[0]**2 + a[1]**2) for a in actions]
    v_disp = [abs(a[2]) for a in actions]
    stats["horizontal_displacement"] = {
        "mean": sum(h_disp) / len(h_disp),
        "min": min(h_disp),
        "max": max(h_disp),
        "values": h_disp,
    }
    stats["vertical_displacement"] = {
        "mean": sum(v_disp) / len(v_disp),
        "min": min(v_disp),
        "max": max(v_disp),
        "values": v_disp,
    }
    # 水平/垂直比率 (用于判断 UAV 更倾向于水平还是垂直运动)
    stats["h_v_ratio"] = {
        "mean_h_over_v": (
            (sum(h_disp) / len(h_disp)) / (sum(v_disp) / len(v_disp))
            if sum(v_disp) > 0 else float("inf")
        ),
    }

    # 9.3 方向偏好: 前进/后退/左转/右转
    n_forward = sum(1 for a in actions if a[0] > 0.1)   # dx > 0 → 前进
    n_backward = sum(1 for a in actions if a[0] < -0.1) # dx < 0 → 后退
    n_left = sum(1 for a in actions if a[1] > 0.1)      # dy > 0 → 左移
    n_right = sum(1 for a in actions if a[1] < -0.1)    # dy < 0 → 右移
    n_ascend = sum(1 for a in actions if a[2] > 0.1)    # dz > 0 → 上升
    n_descend = sum(1 for a in actions if a[2] < -0.1)  # dz < 0 → 下降

    stats["direction_preference"] = {
        "forward": n_forward,
        "backward": n_backward,
        "left": n_left,
        "right": n_right,
        "ascend": n_ascend,
        "descend": n_descend,
        "total": len(actions),
    }

    # 9.4 yaw 变化分布
    yaw_changes = dim_values["dyaw"]
    stats["yaw_distribution"] = {
        "mean_abs": sum(abs(y) for y in yaw_changes) / len(yaw_changes),
        "max_abs": max(abs(y) for y in yaw_changes),
        "left_turns": sum(1 for y in yaw_changes if y > 0.01),
        "right_turns": sum(1 for y in yaw_changes if y < -0.01),
        "straight": sum(1 for y in yaw_changes if abs(y) <= 0.01),
    }

    return stats


def analyze_instructions(
    samples: List[dict],
    traj_groups: Dict[str, List[dict]],
    split_name: str,
) -> dict:
    """指令维度分析。

    Returns:
        { length_stats, unique_instructions, diversity }
    """
    stats = {}

    # 每条轨迹取第一条样本的指令 (同轨迹指令相同)
    instructions = []
    for tid, steps in traj_groups.items():
        inst = steps[0].get("instruction", "")
        instructions.append(inst)

    # 字符长度统计
    char_lens = [len(inst) for inst in instructions]
    stats["char_length"] = {
        "mean": sum(char_lens) / len(char_lens) if char_lens else 0,
        "min": min(char_lens) if char_lens else 0,
        "max": max(char_lens) if char_lens else 0,
        "values": char_lens,
    }

    # 词数统计 (按空格分词)
    word_lens = [len(inst.split()) for inst in instructions]
    stats["word_length"] = {
        "mean": sum(word_lens) / len(word_lens) if word_lens else 0,
        "min": min(word_lens) if word_lens else 0,
        "max": max(word_lens) if word_lens else 0,
        "values": word_lens,
    }

    # 指令多样性: 唯一指令数
    unique_insts = set(instructions)
    stats["diversity"] = {
        "total_instructions": len(instructions),
        "unique_instructions": len(unique_insts),
        "uniqueness_ratio": len(unique_insts) / max(len(instructions), 1),
    }

    # 如果所有轨迹共享同一条指令则报警
    if len(unique_insts) == 1 and len(instructions) > 1:
        stats["diversity"]["warning"] = (
            f"所有 {len(instructions)} 条轨迹共享同一指令! "
            f"数据集可能缺乏指令多样性"
        )

    return stats


def analyze_cross_split(
    all_splits: Dict[str, List[dict]],
) -> dict:
    """跨 split 对比分析。

    Returns:
        { scenes_per_split, scene_leakage_check, ... }
    """
    stats = {}

    # 场景在各 split 间的分布
    split_scenes = {}
    for split_name, samples in all_splits.items():
        if not samples:
            continue
        scenes = set(s.get("scene_id") for s in samples)
        split_scenes[split_name] = scenes

    stats["scenes_per_split"] = {
        name: sorted(list(scenes)) for name, scenes in split_scenes.items()
    }

    # 场景泄露检查: 同一场景是否出现在多个 split 中
    all_scene_splits = defaultdict(set)
    for split_name, scenes in split_scenes.items():
        for scene in scenes:
            all_scene_splits[scene].add(split_name)

    leaked_scenes = {
        scene: sorted(list(splits))
        for scene, splits in all_scene_splits.items()
        if len(splits) > 1
    }
    stats["scene_leakage"] = {
        "has_leakage": len(leaked_scenes) > 0,
        "leaked_scenes": leaked_scenes,
        "note": (
            "同一场景出现在多个 split 中: train 和 val_seen 共享场景是正常的, "
            "但 train 和 val_unseen/test 不应共享场景"
        ),
    }

    return stats


def build_full_stats(
    all_splits: Dict[str, List[dict]],
) -> dict:
    """构建完整的统计报告字典 (用于导出和可视化)。"""
    full_stats = {
        "overview": {},
        "splits": {},
        "cross_split": {},
    }

    total_samples = 0
    total_trajs = set()
    total_scenes = set()

    for split_name, samples in all_splits.items():
        if not samples:
            full_stats["splits"][split_name] = {"num_samples": 0}
            continue

        traj_groups = group_by_trajectory(samples)

        n_samples = len(samples)
        n_trajs = len(traj_groups)
        n_scenes = len(set(s.get("scene_id") for s in samples))

        total_samples += n_samples
        total_trajs.update(traj_groups.keys())
        total_scenes.update(s.get("scene_id") for s in samples)

        split_stats = {
            "num_samples": n_samples,
            "num_trajectories": n_trajs,
            "num_scenes": n_scenes,
            "height_analysis": analyze_heights(samples, traj_groups, split_name),
            "height_position_numerical": analyze_height_position_numerical(samples, traj_groups, split_name),
            "scene_analysis": analyze_scenes(samples, traj_groups, split_name),
            "trajectory_analysis": analyze_trajectories(samples, traj_groups, split_name),
            "action_analysis": analyze_actions(samples, traj_groups, split_name),
            "instruction_analysis": analyze_instructions(samples, traj_groups, split_name),
        }
        full_stats["splits"][split_name] = split_stats

    full_stats["overview"] = {
        "total_samples": total_samples,
        "total_trajectories": len(total_trajs),
        "total_scenes": len(total_scenes),
        "splits_with_data": [n for n, s in all_splits.items() if s],
    }
    full_stats["cross_split"] = analyze_cross_split(all_splits)

    return full_stats


# ================================================================
#  文本统计报告
# ================================================================

def print_detailed_report(all_splits: Dict[str, List[dict]]) -> None:
    """打印详细的多维度统计报告。"""
    full_stats = build_full_stats(all_splits)

    ov = full_stats["overview"]
    print(f"\n{'=' * 65}")
    print(f"  HAD 数据集详细统计报告")
    print(f"{'=' * 65}")
    print(f"  总样本数:    {ov['total_samples']}")
    print(f"  总轨迹数:    {ov['total_trajectories']}")
    print(f"  总场景数:    {ov['total_scenes']}")
    print(f"  有数据的 split: {', '.join(ov['splits_with_data'])}")

    for split_name, st in full_stats["splits"].items():
        if st.get("num_samples", 0) == 0:
            continue

        print(f"\n  {'─' * 55}")
        print(f"  [{split_name}]  (样本: {st['num_samples']}, "
              f"轨迹: {st['num_trajectories']}, 场景: {st['num_scenes']})")
        print(f"  {'─' * 55}")

        # --- 高度分析 ---
        ha = st["height_analysis"]
        sd = ha["stage_distribution"]
        print(f"\n  📏 高度分析")
        print(f"     分布: low={sd.get('low',0)}, mid={sd.get('mid',0)}, "
              f"high={sd.get('high',0)}")
        print(f"     单一高度段轨迹: {ha['num_single_stage_trajs']}")
        if ha["num_mixed_trajs"] > 0:
            print(f"     ⚠ 混合高度轨迹: {ha['num_mixed_trajs']} 条")
            for mt in ha["mixed_height_trajectories"]:
                print(f"       - {mt['trajectory_id'][:20]}... : "
                      f"{mt['stages']} ({mt['num_steps']}步)")
        else:
            print(f"     混合高度轨迹: 0 条 (每条轨迹高度段单一)")

        trend = ha["altitude_trend"]
        print(f"     高度趋势: ↑上升{trend['ascending']}条 "
              f"↓下降{trend['descending']}条 →平稳{trend['stable']}条")

        # --- 高度与位置数值分析 ---
        hpn = st["height_position_numerical"]
        if hpn.get("altitude_numerical"):
            an = hpn["altitude_numerical"]
            pcts = an["percentiles"]
            print(f"\n  📏 高度数值分析 (全量, 不限于三段)")
            print(f"     样本数: {an['count']}")
            print(f"     均值={an['mean']:.1f}m, 标准差={an['std']:.1f}m, "
                  f"偏度={an['skewness']}")
            print(f"     极差={an['range']:.1f}m, IQR={an['iqr']:.1f}m")
            print(f"     分位数 (m):")
            for p_label, p_vals in [("min..max", [0, 5, 10, 25, 50, 75, 90, 95, 100])]:
                items = [f"P{p}={pcts[p]:.1f}" for p in p_vals]
                print(f"       {'  '.join(items)}")

            # 直方图概要
            hist = an["histogram"]
            if hist["counts"]:
                max_bin = max(hist["counts"])
                peak_bins = [
                    hist["bin_centers"][i]
                    for i, c in enumerate(hist["counts"]) if c == max_bin
                ]
                print(f"     直方图峰值位置: {[round(b,1) for b in peak_bins]}m "
                      f"({max_bin} 样本)")

        # 位置数值分析
        pn = hpn.get("position_numerical", {})
        if pn:
            print(f"\n  📍 位置数值分析")
            for axis_key in ["x", "y", "z"]:
                axis = pn.get(axis_key, {})
                if not axis:
                    continue
                ap = axis["percentiles"]
                print(f"     {axis['name']:25s}: "
                      f"range=[{ap[0]:.0f}, {ap[100]:.0f}], "
                      f"P50={ap[50]:.1f}, "
                      f"mean={axis['mean']:.1f}, "
                      f"std={axis['std']:.1f}")

        # 每条轨迹的箱线图数据摘要
        box_data = hpn.get("per_trajectory_boxplot", [])
        if box_data:
            print(f"\n  📦 每条轨迹 altitude 箱线图数据 (5-number summary):")
            for bd in box_data:
                print(f"     [{bd['trajectory_id'][:20]}...] "
                      f"min={bd['min']:.1f} Q1={bd['q1']:.1f} "
                      f"median={bd['median']:.1f} Q3={bd['q3']:.1f} "
                      f"max={bd['max']:.1f} | mean={bd['mean']:.1f} "
                      f"({bd['num_steps']}步)")

        # 阈值建议
        ts = hpn.get("threshold_suggestions", {})
        if ts:
            print(f"\n  🎯 高度分段阈值建议 (为后续调整提供依据):")
            for scheme_name in ["current", "equal_frequency",
                               "equal_range", "percentile_33_66",
                               "natural_breaks"]:
                scheme = ts.get(scheme_name)
                if not scheme:
                    continue
                dist = scheme["distribution"]
                marker = " ← 当前" if scheme_name == "current" else ""
                desc = scheme.get("description", "")
                print(f"     [{scheme_name}] {desc}{marker}")
                print(f"       阈值: low<{scheme['threshold_low']}  "
                      f"mid<{scheme['threshold_high']}  high")
                print(f"       分布: low={dist['low_pct']}%  "
                      f"mid={dist['mid_pct']}%  high={dist['high_pct']}% "
                      f"({dist['low']}/{dist['mid']}/{dist['high']} 样本)")

        # --- 场景分析 ---
        sa = st["scene_analysis"]
        print(f"\n  🗺 场景分析")
        print(f"     场景数: {sa['num_scenes']}")
        for sd_detail in sa["scenes_detail"]:
            print(f"     [{sd_detail['scene_id']}]: "
                  f"{sd_detail['num_trajectories']}条轨迹, "
                  f"{sd_detail['num_samples']}个样本, "
                  f"平均轨迹长度 {sd_detail['avg_traj_length']:.1f}步")

        # --- 轨迹分析 ---
        ta = st["trajectory_analysis"]
        if ta.get("length_stats"):
            ls = ta["length_stats"]
            print(f"\n  🛤 轨迹分析")
            print(f"     轨迹长度: min={ls['min']}, max={ls['max']}, "
                  f"mean={ls['mean']:.1f}, std={ls['std']:.1f}")
            print(f"     总步数: {ls['total_steps']}")

            # 位移统计
            if ta.get("displacements"):
                h_d = [d["horizontal_displacement"] for d in ta["displacements"]]
                v_d = [d["vertical_displacement"] for d in ta["displacements"]]
                print(f"     水平位移 (m): mean={sum(h_d)/len(h_d):.1f}, "
                      f"min={min(h_d):.1f}, max={max(h_d):.1f}")
                print(f"     垂直位移 (m): mean={sum(v_d)/len(v_d):.1f}, "
                      f"min={min(v_d):.1f}, max={max(v_d):.1f}")

        # --- 动作分析 ---
        aa = st["action_analysis"]
        if aa.get("per_dimension"):
            print(f"\n  🎮 动作分析")
            for dim in ["dx", "dy", "dz", "dyaw"]:
                d = aa["per_dimension"][dim]
                print(f"     {dim:6s}: mean={d['mean']:+.3f}, "
                      f"std={d['std']:.3f}, range=[{d['min']:+.2f}, {d['max']:+.2f}]")

            hd = aa["horizontal_displacement"]
            vd = aa["vertical_displacement"]
            print(f"     水平位移/步 (m): mean={hd['mean']:.2f}, "
                  f"max={hd['max']:.2f}")
            print(f"     垂直位移/步 (m): mean={vd['mean']:.2f}, "
                  f"max={vd['max']:.2f}")
            print(f"     水平/垂直比: {aa['h_v_ratio']['mean_h_over_v']:.1f}")

            dp = aa["direction_preference"]
            print(f"     方向偏好: 前{dp['forward']} 后{dp['backward']} "
                  f"左{dp['left']} 右{dp['right']} "
                  f"升{dp['ascend']} 降{dp['descend']} "
                  f"(共{dp['total']}步)")

            yd = aa["yaw_distribution"]
            print(f"     Yaw变化: 左转{yd['left_turns']} 右转{yd['right_turns']} "
                  f"直行{yd['straight']}, 平均|Δyaw|={yd['mean_abs']:.3f}")

        # --- 指令分析 ---
        ia = st["instruction_analysis"]
        if ia.get("char_length"):
            cl = ia["char_length"]
            wl = ia["word_length"]
            div = ia["diversity"]
            print(f"\n  📝 指令分析")
            print(f"     字符长度: mean={cl['mean']:.0f}, "
                  f"min={cl['min']}, max={cl['max']}")
            print(f"     词数:     mean={wl['mean']:.0f}, "
                  f"min={wl['min']}, max={wl['max']}")
            print(f"     多样性:   {div['unique_instructions']}/{div['total_instructions']} "
                  f"唯一指令 (比率 {div['uniqueness_ratio']:.1%})")
            if "warning" in div:
                print(f"     ⚠ {div['warning']}")

    # --- 跨 split 对比 ---
    cs = full_stats["cross_split"]
    print(f"\n  {'─' * 55}")
    print(f"  [跨 Split 对比]")
    print(f"  {'─' * 55}")
    for split_name, scenes in cs["scenes_per_split"].items():
        print(f"  {split_name:20s}: {len(scenes)} 个场景 {scenes}")

    leakage = cs["scene_leakage"]
    if leakage["has_leakage"]:
        print(f"\n  ⚠ 场景泄露警告!")
        for scene, splits in leakage["leaked_scenes"].items():
            print(f"    场景 '{scene}' 出现在: {splits}")
    else:
        print(f"\n  ✅ 无场景泄露 (不同 split 间场景不重叠)")

    print(f"\n{'=' * 65}\n")


# ================================================================
#  可视化 (--plot)
# ================================================================

def plot_statistics(full_stats: dict, output_dir: Path) -> List[str]:
    """生成统计图表并保存到 output_dir。返回生成的文件路径列表。"""
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib 未安装, 跳过图表生成. "
              "安装: pip install matplotlib")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 中文字体设置 (如果可用)
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    # 收集所有 split 中非空的样本
    all_splits_nonempty = {
        k: v for k, v in full_stats["splits"].items()
        if v.get("num_samples", 0) > 0
    }

    # ---- 图1: 高度分段分布 (饼图 × split 数) ----
    n_splits = len(all_splits_nonempty)
    if n_splits > 0:
        fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 4))
        if n_splits == 1:
            axes = [axes]
        for ax, (split_name, st) in zip(axes, all_splits_nonempty.items()):
            ha = st["height_analysis"]
            sd = ha["stage_distribution"]
            labels = [f"{k}\n({v})" for k, v in sd.items() if v > 0]
            sizes = [v for v in sd.values() if v > 0]
            colors = ["#2ecc71", "#f39c12", "#e74c3c"][:len(sizes)]
            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90)
            ax.set_title(f"{split_name}\nHeight Stage Distribution", fontsize=12)
        plt.tight_layout()
        path = output_dir / "height_stage_pie.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    # ---- 图2: 轨迹长度直方图 ----
    all_lengths = []
    all_labels = []
    for split_name, st in all_splits_nonempty.items():
        ta = st["trajectory_analysis"]
        if ta.get("length_values"):
            all_lengths.append(ta["length_values"])
            all_labels.append(split_name)

    if all_lengths:
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = max(5, min(20, max(max(ll) for ll in all_lengths) // 5))
        for lengths, label in zip(all_lengths, all_labels):
            ax.hist(lengths, bins=bins, alpha=0.6, label=label, edgecolor="black")
        ax.set_xlabel("Trajectory Length (steps)")
        ax.set_ylabel("Count")
        ax.set_title("Trajectory Length Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = output_dir / "trajectory_length_hist.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    # ---- 图3: 动作幅度分布 ----
    fig, axes = plt.subplots(2, 5, figsize=(18, 9))
    row_idx = 0
    for split_name, st in all_splits_nonempty.items():
        aa = st["action_analysis"]
        if not aa.get("per_dimension"):
            continue
        if row_idx >= 2:
            break

        dims = ["dx", "dy", "dz", "dyaw"]
        for col, dim in enumerate(dims):
            ax = axes[row_idx, col]
            vals = aa["per_dimension"][dim]["values"]
            ax.hist(vals, bins=30, alpha=0.7, color="#3498db", edgecolor="white")
            ax.set_xlabel(dim)
            ax.set_ylabel("Count")
            ax.set_title(f"{split_name} - {dim}")
            ax.grid(True, alpha=0.3, axis="y")

        # 第5列: 水平位移直方图
        ax = axes[row_idx, 4]
        h_vals = aa["horizontal_displacement"]["values"]
        ax.hist(h_vals, bins=30, alpha=0.7, color="#e74c3c", edgecolor="white")
        ax.set_xlabel("horizontal displacement (m)")
        ax.set_ylabel("Count")
        ax.set_title(f"{split_name} - Horizontal Disp")
        ax.grid(True, alpha=0.3, axis="y")

        row_idx += 1

    # 隐藏未使用的子图
    for r in range(row_idx, 2):
        for c in range(5):
            axes[r, c].set_visible(False)

    plt.tight_layout()
    path = output_dir / "action_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))

    # ---- 图4: 每条轨迹的高度变化曲线 ----
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(range(len(all_splits_nonempty)))
    for (split_name, st), color in zip(all_splits_nonempty.items(), colors):
        ha = st["height_analysis"]
        for traj in ha["per_trajectory_altitude"]:
            ax.axhline(y=LOW_ALT_MAX, color="green", linestyle=":", alpha=0.3)
            ax.axhline(y=MID_ALT_MAX, color="red", linestyle=":", alpha=0.3)
            # 这里没有逐 step 的高度值, 用起点和终点连线
            ax.plot([0, traj["num_steps"] - 1],
                    [traj["alt_start"], traj["alt_end"]],
                    color=color, alpha=0.5, linewidth=1.5,
                    marker="o", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Per-Trajectory Altitude Profile (start → end)")
    ax.axhline(y=LOW_ALT_MAX, color="green", linestyle="--", alpha=0.5, label="low/mid boundary")
    ax.axhline(y=MID_ALT_MAX, color="red", linestyle="--", alpha=0.5, label="mid/high boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "altitude_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))

    # ---- 图5: 场景-轨迹统计柱状图 ----
    scene_names = []
    scene_traj_counts = []
    for split_name, st in all_splits_nonempty.items():
        sa = st["scene_analysis"]
        for sd_detail in sa["scenes_detail"]:
            scene_names.append(f"{sd_detail['scene_id']}\n({split_name})")
            scene_traj_counts.append(sd_detail["num_trajectories"])

    if scene_names:
        fig, ax = plt.subplots(figsize=(max(6, len(scene_names) * 1.2), 5))
        bars = ax.bar(range(len(scene_names)), scene_traj_counts,
                      color="#3498db", edgecolor="white")
        ax.set_xticks(range(len(scene_names)))
        ax.set_xticklabels(scene_names, fontsize=8)
        ax.set_ylabel("Number of Trajectories")
        ax.set_title("Trajectories per Scene")
        for bar, count in zip(bars, scene_traj_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(count), ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = output_dir / "scene_trajectory_counts.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    # ---- 图6: altitude 全量分布直方图 (含分位数标记) ----
    for split_name, st in all_splits_nonempty.items():
        hpn = st.get("height_position_numerical", {})
        an = hpn.get("altitude_numerical", {})
        if not an.get("histogram"):
            continue

        hist = an["histogram"]
        pcts = an.get("percentiles", {})

        fig, ax = plt.subplots(figsize=(10, 5))

        # 直方图
        ax.bar(hist["bin_centers"], hist["counts"],
               width=(hist["bin_edges"][1] - hist["bin_edges"][0]) if len(hist["bin_edges"]) > 1 else 1,
               alpha=0.7, color="#3498db", edgecolor="white",
               label=f"n={an['count']}")

        # 分位数标记线
        for p_val, color, ls in [(25, "#e67e22", "--"), (50, "#e74c3c", "-"),
                                   (75, "#e67e22", "--")]:
            if p_val in pcts:
                ax.axvline(x=pcts[p_val], color=color, linestyle=ls,
                          linewidth=2, alpha=0.8,
                          label=f"P{p_val}={pcts[p_val]:.1f}m")

        # 当前阈值标记
        ax.axvline(x=LOW_ALT_MAX, color="green", linestyle=":", linewidth=2,
                  alpha=0.6, label=f"low/mid={LOW_ALT_MAX}m")
        ax.axvline(x=MID_ALT_MAX, color="red", linestyle=":", linewidth=2,
                  alpha=0.6, label=f"mid/high={MID_ALT_MAX}m")

        ax.set_xlabel("Altitude (m)")
        ax.set_ylabel("Count")
        ax.set_title(f"{split_name} - Altitude Distribution (P25/P50/P75 + thresholds)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = output_dir / f"altitude_histogram_{split_name.replace('.jsonl','')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    # ---- 图7: 位置 (X/Y/Z) 分布直方图 ----
    for split_name, st in all_splits_nonempty.items():
        pn = st.get("height_position_numerical", {}).get("position_numerical", {})
        if not pn:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, axis_key, color in zip(axes, ["x", "y", "z"],
                                        ["#3498db", "#e74c3c", "#2ecc71"]):
            axis_data = pn.get(axis_key, {})
            hist = axis_data.get("histogram", {})
            if not hist.get("bin_centers"):
                ax.set_title(f"{axis_data.get('name', axis_key)} (no data)")
                continue

            ax.bar(hist["bin_centers"], hist["counts"],
                   width=(hist["bin_edges"][1] - hist["bin_edges"][0]) if len(hist["bin_edges"]) > 1 else 1,
                   alpha=0.7, color=color, edgecolor="white")
            pcts = axis_data.get("percentiles", {})
            if 50 in pcts:
                ax.axvline(x=pcts[50], color="black", linestyle="--",
                          linewidth=1.5, alpha=0.7, label=f"P50={pcts[50]:.0f}")
            ax.set_xlabel(axis_data.get("name", axis_key))
            ax.set_ylabel("Count")
            ax.set_title(f"{axis_data.get('name', axis_key)}\n"
                        f"range=[{axis_data.get('percentiles',{}).get(0,0):.0f}, "
                        f"{axis_data.get('percentiles',{}).get(100,0):.0f}]")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = output_dir / f"position_histograms_{split_name.replace('.jsonl','')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    # ---- 图8: 阈值建议对比图 ----
    for split_name, st in all_splits_nonempty.items():
        ts = st.get("height_position_numerical", {}).get("threshold_suggestions", {})
        if not ts:
            continue

        schemes = [k for k in ["current", "equal_frequency",
                               "equal_range", "percentile_33_66",
                               "natural_breaks"] if k in ts]
        if not schemes:
            continue

        fig, ax = plt.subplots(figsize=(10, len(schemes) * 0.8 + 1))
        scheme_labels = []
        low_pcts = []
        mid_pcts = []
        high_pcts = []

        for scheme_name in schemes:
            scheme = ts[scheme_name]
            dist = scheme["distribution"]
            label = f"{scheme_name}\n({scheme.get('description','')})\n"
            label += f"lo<{scheme['threshold_low']} mid<{scheme['threshold_high']} hi"
            scheme_labels.append(label)
            low_pcts.append(dist["low_pct"])
            mid_pcts.append(dist["mid_pct"])
            high_pcts.append(dist["high_pct"])

        y_pos = range(len(schemes))
        bar_height = 0.35
        ax.barh([y + bar_height for y in y_pos], low_pcts, bar_height,
                color="#2ecc71", alpha=0.8, label="low", edgecolor="white")
        ax.barh(y_pos, mid_pcts, bar_height,
                color="#f39c12", alpha=0.8, label="mid", edgecolor="white")
        ax.barh([y - bar_height for y in y_pos], high_pcts, bar_height,
                color="#e74c3c", alpha=0.8, label="high", edgecolor="white")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(scheme_labels, fontsize=8)
        ax.set_xlabel("Percentage of samples (%)")
        ax.set_title(f"{split_name} - Threshold Scheme Comparison\n"
                     f"(data range: {ts['data_range']['min']}m ~ {ts['data_range']['max']}m)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        path = output_dir / f"threshold_comparison_{split_name.replace('.jsonl','')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    return saved


# ================================================================
#  JSON 导出 (--export_stats)
# ================================================================

class StatsEncoder(json.JSONEncoder):
    """自定义 JSON encoder: 处理 set / Path 等不可序列化对象。"""
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return sorted(list(obj))
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def export_stats_json(full_stats: dict, output_dir: Path) -> str:
    """将完整统计数据导出为 JSON 文件。

    注意: 移除了 'values' 列表 (用于直方图的原始数据) 以控制文件大小。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 深拷贝并移除大数组
    import copy
    export = copy.deepcopy(full_stats)

    for split_name, st in export.get("splits", {}).items():
        if not st:
            continue
        # 移除原始值数组, 只保留统计摘要
        ha = st.get("height_analysis", {})
        ha.pop("altitude_values", None)
        for traj in ha.get("per_trajectory_altitude", []):
            pass  # per_trajectory_altitude 本身很小, 保留

        ta = st.get("trajectory_analysis", {})
        ta.pop("length_values", None)

        aa = st.get("action_analysis", {})
        for dim_info in aa.get("per_dimension", {}).values():
            dim_info.pop("values", None)
        aa.get("horizontal_displacement", {}).pop("values", None)
        aa.get("vertical_displacement", {}).pop("values", None)

        ia = st.get("instruction_analysis", {})
        ia.get("char_length", {}).pop("values", None)
        ia.get("word_length", {}).pop("values", None)

    path = output_dir / "dataset_statistics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2, cls=StatsEncoder)

    return str(path)


# ================================================================
#  主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HAD 数据集完整性检查 + 详细统计 + 可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础校验 + 统计
  python data_tools/check_dataset.py --data_dir ./data/processed

  # 含图像检查
  python data_tools/check_dataset.py --data_dir ./data/processed --check_images

  # 生成图表 + 导出 JSON
  python data_tools/check_dataset.py --data_dir ./data/processed --plot --export_stats --output_dir ./outputs/analysis
        """,
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="处理后数据目录, 包含 train.jsonl / val_seen.jsonl 等",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+",
        default=["train.jsonl", "val_seen.jsonl", "val_unseen.jsonl", "test.jsonl"],
        help="要检查的 JSONL 文件名 (默认全部)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/analysis",
        help="图表和统计 JSON 的输出目录 (默认 ./outputs/analysis)",
    )
    parser.add_argument(
        "--check_images", action="store_true",
        help="检查图像文件是否存在 (较慢)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="严格模式: 空文件也视为错误",
    )
    parser.add_argument(
        "--no_stats", action="store_true",
        help="跳过详细统计分析 (仅做基础校验)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="生成统计图表 (需要 matplotlib)",
    )
    parser.add_argument(
        "--export_stats", action="store_true",
        help="将统计数据导出为 JSON 文件",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"[ERROR] 目录不存在: {data_dir}")
        return 1

    # ---- 加载 ----
    all_splits: Dict[str, List[dict]] = {}
    total_errors: List[str] = []

    print(f"[INFO] 数据目录: {data_dir}")
    print(f"[INFO] 检查 splits: {args.splits}")

    for filename in args.splits:
        filepath = data_dir / filename
        if not filepath.exists():
            if args.strict:
                total_errors.append(f"文件不存在: {filepath}")
                print(f"  [MISS] {filepath}")
            else:
                print(f"  [SKIP] {filepath} (不存在)")
            continue

        print(f"\n  [{filename}]")
        samples = load_jsonl(filepath)
        print(f"    加载 {len(samples)} 条样本")
        all_splits[filename] = samples

        if not samples:
            if args.strict:
                total_errors.append(f"{filename}: 文件为空")
            continue

        # --- 校验 ---
        print(f"    检查字段...", end=" ")
        errs = check_fields(samples, filename)
        total_errors.extend(errs)
        print(f"{'OK' if not errs else f'{len(errs)} errors'}")

        print(f"    检查值域...", end=" ")
        errs = check_value_constraints(samples, filename)
        total_errors.extend(errs)
        print(f"{'OK' if not errs else f'{len(errs)} errors'}")

        print(f"    检查轨迹一致性...", end=" ")
        errs, traj_groups = check_trajectory_consistency(samples, filename)
        total_errors.extend(errs)
        print(f"{'OK' if not errs else f'{len(errs)} errors'} "
              f"({len(traj_groups)} 条轨迹)")

        if args.check_images:
            print(f"    检查图像文件...", end=" ")
            errs = check_images(samples, data_dir, filename)
            total_errors.extend(errs)
            print(f"{'OK' if not errs else f'{len(errs)} errors'}")

    # --- 跨文件检查 ---
    print(f"\n  检查跨文件重复...", end=" ")
    errs = check_duplicates(all_splits)
    total_errors.extend(errs)
    print(f"{'OK' if not errs else f'{len(errs)} errors'}")

    # --- 统计报告 ---
    if not args.no_stats:
        print_detailed_report(all_splits)

    # --- 可视化 ---
    if args.plot:
        if not HAS_MATPLOTLIB:
            print("[WARN] matplotlib 未安装, 跳过图表生成")
        else:
            output_dir = Path(args.output_dir)
            full_stats = build_full_stats(all_splits)
            saved = plot_statistics(full_stats, output_dir)
            if saved:
                print(f"\n[PLOT] 已生成 {len(saved)} 张图表:")
                for p in saved:
                    print(f"  -> {p}")

    # --- JSON 导出 ---
    if args.export_stats:
        output_dir = Path(args.output_dir)
        full_stats = build_full_stats(all_splits)
        json_path = export_stats_json(full_stats, output_dir)
        print(f"\n[EXPORT] 统计数据已导出: {json_path}")

    # --- 结果汇总 ---
    print(f"\n{'=' * 60}")
    if total_errors:
        print(f"检查完成: {len(total_errors)} 个问题")
        print(f"{'=' * 60}")
        for err in total_errors[:30]:
            print(f"  - {err}")
        if len(total_errors) > 30:
            print(f"  ... 还有 {len(total_errors) - 30} 个问题未显示")
        print(f"{'=' * 60}")
        return 1
    else:
        total = sum(len(s) for s in all_splits.values())
        print(f"检查完成: 全部通过! ({total} 条样本, 0 个问题)")
        print(f"{'=' * 60}")
        return 0


if __name__ == "__main__":
    exit(main())
