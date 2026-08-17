# DepthAnythingV2 与 AirSim 深度图抽样对比

- 输出目录: `/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260708_airsim_depth_da_sample`
- 抽样: `12` 个 pose, `24` 张深度图(front/down)
- 位姿来源: `merged_data.json -> trajectory_raw[step]` 的 AirSim 世界坐标和四元数；未使用局部/目标对齐坐标。
- AirSim 深度: `DepthPerspective` float metric depth；DepthAnythingV2 深度: 每张图独立 min-max 到 uint8 的相对深度。
- 因 DepthAnythingV2 是相对深度，本报告比较每张图归一化后的相关性，并同时检测 direct/inverse 两种极性。

## 结论

明确：本次 BrushifyCountryRoads 的 AirSim 深度采集结果不可用，表现为 DepthPerspective 在所有抽样 front/down 图上都返回常量 1.0，因此 airsim_depth_vis.png 全黑。这更符合该场景在当前 offscreen/AirSim 深度渲染链路下的 depth buffer 输出退化问题，而不是位姿或坐标放置错误；同一脚本在 BrushifyUrban 上能得到正常变化的 metric depth，且所有样本位姿误差均为 0.0000m / 0.000deg。

修正后的有效结论是：BrushifyUrban 中 front 视角的 DepthAnythingV2 相对深度与 AirSim metric depth 结构相似度较好；down 视角相似度明显偏弱。因此 DepthAnythingV2 深度可作为 front 视角的粗略相对深度辅助，但不能直接当作严格的 AirSim 深度真值，尤其不能用当前 BrushifyCountryRoads 的 AirSim 深度结果做判断。

## 有效性检查

| scene | AirSim depth 状态 | 处理方式 |
| --- | --- | --- |
| BrushifyCountryRoads | front/down 抽样全部为常量 1.0，可视化全黑 | 判定为 AirSim 深度渲染退化，剔除出有效相关性结论 |
| BrushifyUrban | front/down 深度存在正常动态范围 | 保留用于有效对比 |

## 修正后汇总指标（剔除常量 AirSim 深度）

| scope | valid pairs | mean abs Spearman | median abs Spearman | 结论 |
| --- | ---: | ---: | ---: | --- |
| overall valid | 12 | 0.606 | 0.744 | 中等，受 down 视角拖低 |
| front valid | 6 | 0.877 | 0.930 | 相似度较好 |
| down valid | 6 | 0.336 | 0.182 | 相似度较弱且不稳定 |

## 原始汇总指标（包含常量深度，仅供排查，不作为最终结论）

| scope | mean abs Spearman | median abs Spearman | mean MAE(norm) | polarity |
| --- | ---: | ---: | ---: | --- |
| overall | 0.710 | 0.890 | 0.409 | direct:19, inverse:5 |
| front | 0.931 | 0.958 | 0.432 | direct:11, inverse:1 |
| down | 0.489 | 0.498 | 0.386 | direct:8, inverse:4 |

## 样本明细

| scene | trajectory | frame | view | abs Spearman | MAE(norm) | pose error(m/deg) | grid |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| BrushifyCountryRoads | `0008c004` | 0 | front | 0.999 | 0.323 | 0.0000/0.000 | `samples/BrushifyCountryRoads_0008c004-9c02-40d3-928f-b7228c17a39d_000000/front/comparison_grid.png` |
| BrushifyCountryRoads | `0008c004` | 0 | down | 0.055 | 0.292 | 0.0000/0.000 | `samples/BrushifyCountryRoads_0008c004-9c02-40d3-928f-b7228c17a39d_000000/down/comparison_grid.png` |
| BrushifyCountryRoads | `0008c004` | 210 | front | 0.990 | 0.253 | 0.0000/0.000 | `samples/BrushifyCountryRoads_0008c004-9c02-40d3-928f-b7228c17a39d_000210/front/comparison_grid.png` |
| BrushifyCountryRoads | `0008c004` | 210 | down | 0.943 | 0.575 | 0.0000/0.000 | `samples/BrushifyCountryRoads_0008c004-9c02-40d3-928f-b7228c17a39d_000210/down/comparison_grid.png` |
| BrushifyCountryRoads | `0008c004` | 411 | front | 0.994 | 0.187 | 0.0000/0.000 | `samples/BrushifyCountryRoads_0008c004-9c02-40d3-928f-b7228c17a39d_000411/front/comparison_grid.png` |
| BrushifyCountryRoads | `0008c004` | 411 | down | 0.761 | 0.332 | 0.0000/0.000 | `samples/BrushifyCountryRoads_0008c004-9c02-40d3-928f-b7228c17a39d_000411/down/comparison_grid.png` |
| BrushifyCountryRoads | `00cae84c` | 0 | front | 0.930 | 0.290 | 0.0000/0.000 | `samples/BrushifyCountryRoads_00cae84c-2e1b-4b7b-95e7-7d9505670795_000000/front/comparison_grid.png` |
| BrushifyCountryRoads | `00cae84c` | 0 | down | 0.352 | 0.591 | 0.0000/0.000 | `samples/BrushifyCountryRoads_00cae84c-2e1b-4b7b-95e7-7d9505670795_000000/down/comparison_grid.png` |
| BrushifyCountryRoads | `00cae84c` | 115 | front | 0.998 | 0.231 | 0.0000/0.000 | `samples/BrushifyCountryRoads_00cae84c-2e1b-4b7b-95e7-7d9505670795_000115/front/comparison_grid.png` |
| BrushifyCountryRoads | `00cae84c` | 115 | down | 0.968 | 0.611 | 0.0000/0.000 | `samples/BrushifyCountryRoads_00cae84c-2e1b-4b7b-95e7-7d9505670795_000115/down/comparison_grid.png` |
| BrushifyCountryRoads | `00cae84c` | 224 | front | 0.999 | 0.222 | 0.0000/0.000 | `samples/BrushifyCountryRoads_00cae84c-2e1b-4b7b-95e7-7d9505670795_000224/front/comparison_grid.png` |
| BrushifyCountryRoads | `00cae84c` | 224 | down | 0.780 | 0.331 | 0.0000/0.000 | `samples/BrushifyCountryRoads_00cae84c-2e1b-4b7b-95e7-7d9505670795_000224/down/comparison_grid.png` |
| BrushifyUrban | `00d1fb8b` | 0 | front | 0.865 | 0.633 | 0.0000/0.000 | `samples/BrushifyUrban_00d1fb8b-d31a-44d0-b0da-aa91b30cc0ae_000000/front/comparison_grid.png` |
| BrushifyUrban | `00d1fb8b` | 0 | down | 0.844 | 0.154 | 0.0000/0.000 | `samples/BrushifyUrban_00d1fb8b-d31a-44d0-b0da-aa91b30cc0ae_000000/down/comparison_grid.png` |
| BrushifyUrban | `00d1fb8b` | 85 | front | 0.955 | 0.663 | 0.0000/0.000 | `samples/BrushifyUrban_00d1fb8b-d31a-44d0-b0da-aa91b30cc0ae_000085/front/comparison_grid.png` |
| BrushifyUrban | `00d1fb8b` | 85 | down | 0.061 | 0.264 | 0.0000/0.000 | `samples/BrushifyUrban_00d1fb8b-d31a-44d0-b0da-aa91b30cc0ae_000085/down/comparison_grid.png` |
| BrushifyUrban | `00d1fb8b` | 170 | front | 0.962 | 0.753 | 0.0000/0.000 | `samples/BrushifyUrban_00d1fb8b-d31a-44d0-b0da-aa91b30cc0ae_000170/front/comparison_grid.png` |
| BrushifyUrban | `00d1fb8b` | 170 | down | 0.106 | 0.405 | 0.0000/0.000 | `samples/BrushifyUrban_00d1fb8b-d31a-44d0-b0da-aa91b30cc0ae_000170/down/comparison_grid.png` |
| BrushifyUrban | `02872f4c` | 0 | front | 0.620 | 0.664 | 0.0000/0.000 | `samples/BrushifyUrban_02872f4c-6fe4-4b46-b33f-e6ba10f391c4_000000/front/comparison_grid.png` |
| BrushifyUrban | `02872f4c` | 0 | down | 0.643 | 0.414 | 0.0000/0.000 | `samples/BrushifyUrban_02872f4c-6fe4-4b46-b33f-e6ba10f391c4_000000/down/comparison_grid.png` |
| BrushifyUrban | `02872f4c` | 85 | front | 0.945 | 0.649 | 0.0000/0.000 | `samples/BrushifyUrban_02872f4c-6fe4-4b46-b33f-e6ba10f391c4_000085/front/comparison_grid.png` |
| BrushifyUrban | `02872f4c` | 85 | down | 0.259 | 0.309 | 0.0000/0.000 | `samples/BrushifyUrban_02872f4c-6fe4-4b46-b33f-e6ba10f391c4_000085/down/comparison_grid.png` |
| BrushifyUrban | `02872f4c` | 163 | front | 0.914 | 0.318 | 0.0000/0.000 | `samples/BrushifyUrban_02872f4c-6fe4-4b46-b33f-e6ba10f391c4_000163/front/comparison_grid.png` |
| BrushifyUrban | `02872f4c` | 163 | down | 0.102 | 0.349 | 0.0000/0.000 | `samples/BrushifyUrban_02872f4c-6fe4-4b46-b33f-e6ba10f391c4_000163/down/comparison_grid.png` |

## 严谨性说明

- 每个 sample 放置后都记录 `position_error_m` 和 `orientation_error_deg`；超过阈值会直接失败，不进入指标汇总。
- 深度对比不能使用像素值绝对大小判断，因为 DepthAnythingV2 输出已按单图 min-max 压缩到 uint8。
- 本次只验证 front/down 两个已有 DepthAnythingV2 深度目录，未覆盖 left/right/rear。
