# 历史报告归档索引

2026-07-10 仿真收敛重构将原 `reports/` 根目录下的 15 份历史报告按主题归档。2026-08-17 又追加了 3 份 Qwen3-VL 历史指南/审计原文。归档文件不随当前实现改写；下表保存路径与 SHA-256，便于复核引用及内容完整性。

当前结论请优先查看：

- [`reports/project_timeline.md`](../project_timeline.md)
- [`reports/model_summary.md`](../model_summary.md)
- [`reports/simulation_summary.md`](../simulation_summary.md)
- [`docs/model_usage.md`](../../docs/model_usage.md)
- [`docs/simulation_usage.md`](../../docs/simulation_usage.md)

## 模型、输入与离线动作实验

| 旧路径 | 新路径 | SHA-256 |
| --- | --- | --- |
| `reports/experiment_design_report.md` | [`reports/archive/model/experiment_design_report.md`](model/experiment_design_report.md) | `2822c577266084036a31f49085027e3dd879a3817044c1aefcdcc769907b10b3` |
| `reports/experiment_failure_analysis_and_fix_report.md` | [`reports/archive/model/experiment_failure_analysis_and_fix_report.md`](model/experiment_failure_analysis_and_fix_report.md) | `e4d8967fb6791059b92f2e2f51775722dcb66e4029cf408f974c00932288ebae` |
| `reports/yaw_posxyz_experiment_comparison_report.md` | [`reports/archive/model/yaw_posxyz_experiment_comparison_report.md`](model/yaw_posxyz_experiment_comparison_report.md) | `8c2d6fb0209eab28ac82f88dbcfdccc5345427a52118510e4561913107d42f35` |
| `reports/target_aligned_experiment_analysis_20260616_vs_20260614.md` | [`reports/archive/model/target_aligned_experiment_analysis_20260616_vs_20260614.md`](model/target_aligned_experiment_analysis_20260616_vs_20260614.md) | `79819ac8f18aa014af2b1758c48e2e12b53e475156d931b293d9d5f47af7b577` |
| `reports/ha_dvf_yaw_ablation_result_analysis_20260617.md` | [`reports/archive/model/ha_dvf_yaw_ablation_result_analysis_20260617.md`](model/ha_dvf_yaw_ablation_result_analysis_20260617.md) | `fc7eb6c12a676580c598f31c107ea764b52bf7a63c9e0e5794dc121ec5f8e967` |
| `reports/ha_dvf_dz_ablation_result_analysis_20260619.md` | [`reports/archive/model/ha_dvf_dz_ablation_result_analysis_20260619.md`](model/ha_dvf_dz_ablation_result_analysis_20260619.md) | `5208f2698b90a46a7c87f8fd85f741d98f4d16b5b90615d51ec347db39041db8` |
| `reports/had_dz_yaw_code_and_tuning_review_20260621.md` | [`reports/archive/model/had_dz_yaw_code_and_tuning_review_20260621.md`](model/had_dz_yaw_code_and_tuning_review_20260621.md) | `05ca9d5e4925125b5d56e4ca79f629378ef2bdbf20f79e75ace468c29aa5cb2a` |
| `reports/ha_dvf_dz_sign_comprehensive_tuning_report_20260622.md` | [`reports/archive/model/ha_dvf_dz_sign_comprehensive_tuning_report_20260622.md`](model/ha_dvf_dz_sign_comprehensive_tuning_report_20260622.md) | `c13130a10e4cbed6fdedab04c28dbef35f8c61dd4d499f25bc11682cc54061db` |

## Qwen3-VL 历史指南与审计

| 归档文件 | SHA-256 |
| --- | --- |
| [`QWEN3VL_TRAINING_INFERENCE_GUIDE_20260713.md`](model/qwen/QWEN3VL_TRAINING_INFERENCE_GUIDE_20260713.md) | `16e89897782e263204928bfec1fcdf50d0d51c5deb21dd4acd5308a5c9c8f146` |
| [`qwen3vl_2b_full_results_and_action_space_review_20260713.md`](model/qwen/qwen3vl_2b_full_results_and_action_space_review_20260713.md) | `29d335645f9b30ec41b2671936150f6a807050aed4b550082e913c5296506184` |
| [`qwen3vl_8b_full_results_and_mainline_review_20260716.md`](model/qwen/qwen3vl_8b_full_results_and_mainline_review_20260716.md) | `145c6b8544fae3af146eb4f4bfaa30861c1a2d02209af06ea4ddb4e06e6f4227` |

## 仿真控制历史

| 旧路径 | 新路径 | SHA-256 |
| --- | --- | --- |
| `reports/traveluav_closed_loop_control_fixes_20260624.md` | [`reports/archive/simulation/control_history/traveluav_closed_loop_control_fixes_20260624.md`](simulation/control_history/traveluav_closed_loop_control_fixes_20260624.md) | `58164b6ccc1872c82aaddf0bc25c748eec5e150cf01d1e6d10d78a5461f9a39d` |
| `reports/traveluav_expert_rollout_diagnosis_20260626.md` | [`reports/archive/simulation/control_history/traveluav_expert_rollout_diagnosis_20260626.md`](simulation/control_history/traveluav_expert_rollout_diagnosis_20260626.md) | `7ad0e647ff55de247c30b72ce107ece4d7e63911e48f1613c8dcc31aaa77ec90` |
| `reports/traveluav_closed_loop_simulation_workflow.md` | [`reports/archive/simulation/control_history/traveluav_closed_loop_simulation_workflow.md`](simulation/control_history/traveluav_closed_loop_simulation_workflow.md) | `a87215b84be3a73c25f50befa6241754eca6909511334d9af100c43e98314f13` |

## 渲染域诊断

| 旧路径 | 新路径 | SHA-256 |
| --- | --- | --- |
| `reports/traveluav_initial_render_reproducibility_ab_analysis_20260701.md` | [`reports/archive/simulation/render_domain/traveluav_initial_render_reproducibility_ab_analysis_20260701.md`](simulation/render_domain/traveluav_initial_render_reproducibility_ab_analysis_20260701.md) | `c26d6a3ad298691777ba30e6783877243b110fde2441d42ab1e3bc215c054c03` |
| `reports/traveluav_airsim_pseudo_offline_eval_train_val_seen_20260704.md` | [`reports/archive/simulation/render_domain/traveluav_airsim_pseudo_offline_eval_train_val_seen_20260704.md`](simulation/render_domain/traveluav_airsim_pseudo_offline_eval_train_val_seen_20260704.md) | `d4f2fe8acb694bab2b7d8598b37e94578e5ee6b4cf967854269b8b9f847c938c` |
| `reports/traveluav_author_reply_minimal_check_20260707.md` | [`reports/archive/simulation/render_domain/traveluav_author_reply_minimal_check_20260707.md`](simulation/render_domain/traveluav_author_reply_minimal_check_20260707.md) | `5e931f0fb9aced207af3c81d5bb4b7ec0321efd538bcefe11828b57b6cdac26e` |

## 深度诊断

| 旧路径 | 新路径 | SHA-256 |
| --- | --- | --- |
| `reports/20260708_airsim_depth_da_sample_report.md` | [`reports/archive/simulation/depth/20260708_airsim_depth_da_sample_report.md`](simulation/depth/20260708_airsim_depth_da_sample_report.md) | `b2cd98daa878f20f8e92eb0ff16a7b2781a047c7f9491331529c8b5f57a457b1` |

## 引用规则

历史报告记录当时的代码、路径和中间结论，原文不随当前实现更新。需要判断当前状态时，以活动总结为入口，再回到上表所列原始证据；不要把 2026-06-26 的 expert 中途快照当作完整训练集结果，也不要把缺少 `Carla_Town06` 图像的 2026-07-04 评估描述成完整 `val_unseen` 评估。P1-P5 的 `val_unseen` 已参与历史开发，应称 historical unseen-dev；所有离线结果都不能替代一次性新 test 或闭环 SR/SPL。
