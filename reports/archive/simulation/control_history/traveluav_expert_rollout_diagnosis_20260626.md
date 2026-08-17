# 专家轨迹闭环仿真中期诊断

## 结论

本次结论基于 `2026-06-26 16:29:03 CST` 的只读快照。运行仍在继续：`430/962` 条已落盘，BrushifyCountryRoads 已完成 `224/224`，BrushifyUrban 已完成 `206/319`，Carla_Town02 尚未开始。因此下列比例不是最终训练集指标。

1. **专家轨迹的高失败率不能主要归因于电脑配置不足，也不是无解。** 当前专家模式不是直接设置 GT 位姿；它把 JSONL 的 GT 4 维 action 转为世界系增量，再从 AirSim 当前实际位姿出发执行 `moveOnPathAsync`。任何控制误差、碰撞或任务未到端点都会累积到下一步。当前失败的主因是 AirSim 控制/物理闭环没有完成，不是模型预测误差。
2. **`oracle_success` 多于 `success` 是当前代码定义的必然结果，并不表示专家轨迹真正完整成功。** 它表示任一计划 waypoint 或任一观测位置曾进入目标 20m 范围；`success` 还要求最后的 STOP 已执行且该时刻实际位置仍在 20m 内。异常退出前经过目标附近会得到 `oracle`，但不会得到 `success`。

## 已落盘结果

| 项目 | 数值 |
|---|---:|
| 已完成轨迹 | 430 |
| `success` | 72 (16.7%) |
| `oracle`（非成功） | 220 (51.2%) |
| `fail` | 138 (32.1%) |
| 终止：`collision` | 166 |
| 终止：`timeout` | 171 |
| 终止：`stalled` | 18 |
| 终止：`completed` | 75 |

按状态与终止原因交叉统计：

```text
success: completed=72
oracle : timeout=123, collision=96, completed=1
fail   : timeout=48, collision=70, stalled=18, completed=2
```

`220` 条 oracle 非成功中，`219` 条由 timeout 或 collision 终止；其中 `205` 条终止时实际终点已经在目标 20m 内，但没有走到 GT 序列最后的 `done=true` STOP。因此 oracle 与 success 的差距有明确的控制终止原因，并非计数错误。

## 为什么不是“电脑慢导致无解”

活动配置为 `clock_speed=1`、`velocity=1 m/s`、`move_timeout_scale=1.5`、`move_timeout_margin_s=3s`、`move_timeout_max_s=30s`、终点容差 `1m`。主机快照为 20 个逻辑 CPU、load average `4.22/5.31/5.65`；Unreal 占约 475% CPU，但系统并未显示 CPU 饱和。GPU 的瞬时利用率为 0%，该单点读数不能证明未使用 GPU，但也没有显示显存或 GPU 已耗尽。

更关键的是 timeout 的实际记录：

- `171` 条 timeout 中，`169` 条 phase 为 `move_future_result`，仅 `2` 条为 hover 稳定失败。
- 这些 timeout 的终止 action 中，`170` 次 Future 已正常完成且返回 `False`，没有记录到 Future/RPC 异常。
- 终止 action 的 `move_elapsed_s` 中位数是 **0.0217s**，而其动态允许时间中位数是 **6.875s**。
- 具体样例 `02406dbc-...`：期望移动 `4.881m`、允许 `10.322s`，但 Future 在 `2.065s` 返回 `False`，终点误差 `12.789m`。

所以这里的 `timeout` 多数并不是“主机卡到等满 timeout”；代码会在 Future 返回 `False` 且终点误差超过 1m 时标记为 `timeout`。代码位置：`engine/evaluate_traveluav_smoke.py:950-986`。

碰撞同样不能由“模型推理慢”直接解释。`166` 条碰撞由 AirSim `simGetCollisionInfo()` 检出；其中 `Landscape_0` 出现 100 次，此外还有城市地基等静态对象。代码没有过滤这些碰撞：移动等待、Future 完成后和 hover 期间都立即终止（`engine/evaluate_traveluav_smoke.py:935-947, 968-1011`）。这证明 AirSim 报告了物理接触；是否是场景地形/碰撞体设置不合理，仍需结合对应轨迹视频或场景碰撞体进一步验证，不能仅凭名称断言为真碰撞或假碰撞。

## 代码如何造成 Oracle 大于 Success

专家模式逐行读取 JSONL 的 `action` 与 `done`（`engine/evaluate_traveluav_smoke.py:137-190`），并把 action 从目标局部系通过固定起点旋转变成世界系增量，再叠加到**当前 AirSim 实际位置**（`719-742`）。它不是 teleport 到 GT 下一帧。

每步会先把 action 插为最多 5 个 world waypoint；只要任一 waypoint 距目标不超过 `success_threshold=20m`，就置 `oracle_success=True`，此时尚未执行移动（`1382-1384`）。移动后的任一观测点进入 20m 也会置 oracle（`1443-1457`）。

相反，success 只在 `stopped` 为真、且该步真实 `distance_to_target <= 20m` 时成立（`1517-1526`）。专家模式只在 JSONL 最后一条 `done=true` 时输出 stop；前面任何 collision、timeout 或 stalled 都会在到达该 STOP 前跳出循环（`1529-1530`）。最终目录名也严格按 `success -> oracle -> fail` 选择（`1540-1543`）。

因此，当前 oracle 更接近“计划或实际曾经过成功半径”，而不是“完整、安全地执行专家轨迹成功”。若将它理解为专家上界，会系统性高估成功率。

## 后续验证顺序

本轮结束后应先做小规模可复现实验，不应直接把 collision 或 timeout 判定删除：

1. 对固定失败样本离线验证：以每个 GT 原始状态为起点调用当前 `waypoints_from_action`，检查计算出的 `next_world_position` 是否与下一条 GT world position 一致。这样可先排除 action 坐标系/尺度/方向不匹配。
2. 对代表性的 `Landscape_0`、`move_future_result=False`、`stalled` 各抽取若干条，保存外部跟随视角视频和每步高度/碰撞信息，区分真实地形接触与场景碰撞体问题。
3. 对同一固定子集做控制器消融：保留 `clock_speed=1`，仅比较当前 `moveOnPathAsync`、一次受控重试/悬停恢复，以及更宽 endpoint/超时设置。由于多数 Future 在远早于 deadline 时返回 False，单纯加大 timeout 上限不是首选修复。
