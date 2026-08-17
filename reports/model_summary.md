# HAD-UAV-VLN 模型阶段总结

更新时间：2026-08-17

## 结论

P1-P5 已形成可复现的模型与离线实验主线，但不存在跨指标、跨 split 全面最优的单一结构：

- P2 中，简单 concat 在 `val_seen` 的 nMAE 最低，fixed 0.5 在 historical unseen-dev 最低；旧 HA-DVF 在两个 split 均排第 4。
- P3 中，combined reliability 没有稳定超过 height-only 或 content-only；双视角相对 gray-masked 单视角在 seen 有益，在 Town06 overall 反而形成干扰。
- P4 中，yaw stage split 稳定降低首步偏航误差并提高稀有大转角 F1；dz direction-magnitude 在平均误差、方向、尾部和 stop 之间存在取舍。
- P5 中，raw JSON 的 historical unseen-dev MAE 最低；action-query 的 MSE 和推理时延最低，但 MAE 更高。

这些结论均是 teacher-forced 离线动作预测证据，不是新场景测试或闭环导航证据。

## P1-P5 结果

| 协议 | 已完成 | 主要结果 | 结论边界 |
| --- | --- | --- | --- |
| P1 可观测坐标 | HAD 3 seeds；Qwen-8B seed 42 | Qwen/HAD seen MAE 为 `0.2060/0.2567`，unseen-dev 为 `0.1530/0.1944` | 同时改变状态与动作坐标，且保留语言方位，不能解释成纯 target-off 收益 |
| P2 公平融合 | 6 方法 x 3 seeds | seen: concat nMAE `0.2219+/-0.0051`；unseen-dev: fixed 0.5 nMAE `0.1720+/-0.0042` | 只用 `val_seen` 选 best，但 historical unseen-dev 早期已被查看 |
| P3 可靠性门控 | 3 模式 x 3 seeds | seen 最低 MSE 为 content-only `0.5340+/-0.0154`；unseen-dev 最低 MAE 为 height-only `0.2207+/-0.0165` | gray masking 是同一 dual 模型诊断，不是独立训练单视角 |
| P4 动作分解 | 4 变体 x 3 seeds | yaw+dz 将首步 yaw MAE 从 `0.5880/0.6331` 降至 `0.3071/0.2716`（seen/unseen-dev） | stop F1 与部分 MAE、方向指标退化，不能称双重分解全面最优 |
| P5 输出接口 | Qwen3-VL-2B，3 接口，seed 42 | unseen-dev raw MAE `0.1452`；query MSE `0.1752`；batch 128 query `45.769 ms/sample`，raw `110.360` | query 的 parse rate 不适用；只有一个训练 seed |

P5 三个接口均完成相同 3 epoch/8,817 step 预算并选择 epoch 3。unseen-dev 上 action-query 的 MAE 为 `0.1746`，比 raw 高约 `20.2%`，因此其价值是平方误差、尾部和实时性权衡，而不是全面提高精度。

## 数据与指标

正式 development 数据身份固定为：

| split | 帧 | 轨迹 | 用途 |
| --- | ---: | ---: | --- |
| train | 47,014 | 962 | 训练 |
| val_seen | 20,351 | 413 | checkpoint 选择与开发评估 |
| historical unseen-dev | 20,536 | 385 | 历史分析；仅 Carla_Town06 |

终点帧只进入 stop 指标；动作误差排除终点并使用 sample-micro 聚合，yaw 残差 wrap。MSE 等权混合 `m^2` 和 `rad^2`，只能在相同标签与聚合口径下比较。historical unseen-dev 的 high 仅 45 个动作帧、2 条轨迹，不能支持稳定高空泛化结论。

## 尚未完成

- P1-P5 的 freeze receipt 均记录 `test_data_read=false`；尚无冻结后的新场景一次性测试。
- Qwen/P5 仍缺多训练 seed；Qwen 的 gray-masked 视角诊断不能替代独立 front/down 训练。
- AirSim 闭环仍需在相同执行器下报告 SR、SPL、NE、碰撞、轨迹效率、平滑性和实时性。
- P1 仍需在同一 current-yaw 坐标下只移除语言方位，才能回答纯方位依赖问题。

模型运行方法见 [`docs/model_usage.md`](../docs/model_usage.md)。2B/8B 的完整历史审计与绝对指标保存在 [`reports/archive/model/qwen/`](archive/model/qwen/)。
