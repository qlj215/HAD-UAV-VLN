"""Frozen experiment matrix for the 0718 formal supplement.

This module is deliberately declarative.  The five user-facing shell scripts
select one protocol; they do not expose the matrix as dozens of CLI options.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


SEEDS = (42, 43, 44)
TARGET_ON_PROTOCOLS = frozenset({"P2", "P3", "P4", "P5"})


def uses_target_condition(protocol: str) -> bool:
    """Return whether a protocol belongs to the target-conditioned main task."""
    return protocol.upper() in TARGET_ON_PROTOCOLS


@dataclass(frozen=True)
class HadJob:
    name: str
    seed: int
    vision_mode: str = "dual"
    fusion_type: str = "height_cond"
    fixed_gate_alpha: Optional[float] = None
    reliability_mode: Optional[str] = None
    yaw_strategy: str = "baseline"
    dz_strategy: str = "baseline"
    dz_sign_aux: bool = False
    epochs: int = 30
    batch_size: int = 192
    learning_rate: float = 1.0e-4
    warmup_epochs: int = 3
    # Long-running formal jobs can override this to retain a rolling, fully
    # resumable last checkpoint plus best without per-epoch snapshots.
    keep_epoch_checkpoints: bool = True

    def serializable(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QwenJob:
    name: str
    model_size: str
    output_mode: str
    seed: int = 42
    epochs: int = 3
    learning_rate: float = 1.0e-4
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    effective_batch: int = 16
    image_tokens_per_view: int = 49

    @property
    def micro_batch(self) -> int:
        return 2 if self.model_size == "8b" else 4

    @property
    def gradient_accumulation(self) -> int:
        return self.effective_batch // self.micro_batch

    def serializable(self) -> Dict[str, object]:
        payload = asdict(self)
        payload.update({
            "micro_batch": self.micro_batch,
            "gradient_accumulation": self.gradient_accumulation,
        })
        return payload


def p1_had_jobs() -> List[HadJob]:
    return [
        HadJob(
            name=f"had_main_seed{seed}", seed=seed,
            fusion_type="height_cond", yaw_strategy="stage_split",
            dz_strategy="baseline", dz_sign_aux=True,
            epochs=15, batch_size=96, learning_rate=5.0e-5, warmup_epochs=2,
        )
        for seed in SEEDS
    ]


def p2_had_jobs() -> List[HadJob]:
    methods = (
        ("front_only", "front_only", "concat", None),
        ("down_only", "down_only", "concat", None),
        ("fixed_fusion", "dual", "height_cond", 0.5),
        ("concat", "dual", "concat", None),
        ("cross_attn", "dual", "cross_attn", None),
        ("ha_dvf", "dual", "height_cond", None),
    )
    return [
        HadJob(
            name=f"{name}_seed{seed}", seed=seed,
            vision_mode=vision_mode, fusion_type=fusion_type,
            fixed_gate_alpha=alpha,
            keep_epoch_checkpoints=False,
        )
        for name, vision_mode, fusion_type, alpha in methods
        for seed in SEEDS
    ]


def p3_had_jobs() -> List[HadJob]:
    return [
        HadJob(
            name=f"reliability_{mode}_seed{seed}", seed=seed,
            fusion_type="height_cond", reliability_mode=mode,
            keep_epoch_checkpoints=False,
        )
        for mode in ("height_only", "content_only", "combined")
        for seed in SEEDS
    ]


def p4_had_jobs() -> List[HadJob]:
    matrix = (
        ("joint", "baseline", "baseline"),
        ("yaw_only", "stage_split", "baseline"),
        ("dz_only", "baseline", "direction_magnitude"),
        ("yaw_dz", "stage_split", "direction_magnitude"),
    )
    return [
        HadJob(
            name=f"{name}_seed{seed}", seed=seed,
            fusion_type="height_cond", reliability_mode="combined",
            yaw_strategy=yaw, dz_strategy=dz,
            epochs=15, batch_size=96, learning_rate=5.0e-5, warmup_epochs=2,
            keep_epoch_checkpoints=False,
        )
        for name, yaw, dz in matrix
        for seed in SEEDS
    ]


def jobs_for(protocol: str) -> tuple[List[HadJob], List[QwenJob]]:
    key = protocol.upper()
    if key == "P1":
        return p1_had_jobs(), [QwenJob("qwen3vl_8b_raw_seed42", "8b", "raw_json")]
    if key == "P2":
        return p2_had_jobs(), []
    if key == "P3":
        return p3_had_jobs(), []
    if key == "P4":
        return p4_had_jobs(), []
    if key == "P5":
        return [], [
            QwenJob("qwen3vl_2b_raw_seed42", "2b", "raw_json"),
            QwenJob("qwen3vl_2b_fixed4_seed42", "2b", "fixed4_json"),
            QwenJob("qwen3vl_2b_action_query_seed42", "2b", "action_query_regression"),
        ]
    raise ValueError(f"Unknown protocol: {protocol}")


def protocol_manifest(protocol: str) -> Dict[str, object]:
    key = protocol.upper()
    had, qwen = jobs_for(key)
    target_on = uses_target_condition(key)
    development_only = key == "P2"
    rolling_checkpoints = key in {"P2", "P3", "P4"}
    return {
        "protocol": key,
        "had_jobs": [job.serializable() for job in had],
        "qwen_jobs": [job.serializable() for job in qwen],
        "task_condition": (
            "target_on" if target_on else "observable_state_ablation"
        ),
        "coordinate_frame": (
            "target_aligned_local" if target_on else "current_yaw_local_ned"
        ),
        "state_frame": (
            "target_aligned_local" if target_on else "start_yaw_local_ned"
        ),
        "target_conditioning": (
            {
                "enabled": True,
                "instruction": "relative target bearing and yaw",
                "numeric_state": ["target_local_position", "target_local_yaw"],
                "absolute_target_position_used_by_model": False,
            }
            if target_on
            else {
                "numeric_target_aligned_state": False,
                "instruction_relative_bearing": True,
            }
        ),
        "had_seeds": list(SEEDS),
        "qwen_seed": 42,
        "checkpoint_retention": (
            {
                "kept": ["best_model.pth", "last_model.pth"],
                "last_is_rolling_resume_state": True,
                "per_epoch_checkpoints": False,
            }
            if rolling_checkpoints
            else {"per_epoch_checkpoints": True}
        ),
        "checkpoint_selection": {
            "valid_output_minimum_qwen": 0.995,
            "primary": "val_seen normalized_action_mae",
            "tie_break": ["stop_f1 descending", "earlier epoch"],
        },
        "p5_output_interface_protocol": (
            {
                "comparison": [
                    "raw decimal JSON generation",
                    "four-decimal JSON generation",
                    "one input query marker with a five-value continuous regression head",
                ],
                "action_query_loss": (
                    "terminal-masked wrapped residual MSE divided by target-on "
                    "train-set action standard deviation; stop uses 0.5 BCEWithLogits"
                ),
                "token_accounting": {
                    "text_interfaces": "generated JSON payload tokens",
                    "action_query": {
                        "generated_output_tokens": 0,
                        "input_query_markers": 1,
                        "continuous_output_values": 5,
                    },
                },
                "latency_benchmark": {
                    "split": "val_unseen",
                    "stratified_samples": 512,
                    "batch_sizes": [1, 128],
                    "warmup_batches": 4,
                    "repeats": 3,
                    "model_loading_included": False,
                },
            }
            if key == "P5"
            else None
        ),
        "evaluation_splits": (
            ["val_seen", "val_unseen"]
            if development_only
            else ["val_seen", "val_unseen", "deferred_new_test"]
        ),
        "test_policy": (
            "new-scene test deferred; this P2 run stops after frozen val_seen and val_unseen"
            if development_only
            else "read only after freeze receipt; never select parameters"
        ),
    }
