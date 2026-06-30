# TravelUAV 闭环控制修复与 ClockSpeed A/B 报告

日期：2026-06-24

## 修改范围

本次只修改了以下文件：

- `engine/evaluate_traveluav_smoke.py`
- `scripts/simulation/run_traveluav_closed_loop_eval.sh`
- `scripts/simulation/run_traveluav_closed_loop_split.sh`
- `/home/qlj/h3c_pro/TravelUAV/airsim_plugin/AirVLNSimulatorServerTool.py`

## 六项修改

### 1. ClockSpeed 默认值从 10 改为 1

- 修改前：`AirVLNSimulatorServerTool.py:23` 为 `"ClockSpeed": 10`。
- 修改后：`AirVLNSimulatorServerTool.py:23` 为 `"ClockSpeed": 1`。
- `AirVLNSimulatorServerTool.py:313-318` 支持环境变量
  `TRAVELUAV_AIRSIM_CLOCK_SPEED`，用于严格 A/B 时显式覆盖。
- 参数传递位置：
  `evaluate_traveluav_smoke.py:1463`、
  `run_traveluav_closed_loop_eval.sh:191-197,243-249,343-349`、
  `run_traveluav_closed_loop_split.sh:88-94,129-135,328-334`。

### 2. 保存并等待 moveOnPathAsync Future

- 修改前：`evaluate_traveluav_smoke.py:662-669` 调用
  `moveOnPathAsync()` 后丢弃返回值，未读取服务端结果或异常。
- 修改后：`evaluate_traveluav_smoke.py:771-799` 保存 Future，等待其 resolve，
  并调用 `future.get()`；不能只调用 `join()`，因为当前
  `msgpackrpc.future.Future.join()` 不会抛出 Future 内保存的服务端错误。
- `evaluate_traveluav_smoke.py:800-811` 捕获异常类型和消息。
- 每步日志记录 `move_future_completed`、`move_future_result`、
  `move_future_exception`；轨迹 summary 汇总 `move_future_exceptions`。

### 3. 删除“距离增大即经过 waypoint”的判定

- 修改前：`evaluate_traveluav_smoke.py:648-650,671-703` 使用
  `current_idx`、`last_distance`，并在 `new_distance > last_distance` 时认为
  waypoint 已经过。
- 该逻辑已完全删除。
- 修改后：`evaluate_traveluav_smoke.py:781-826` 以 AirSim Future 的完成结果
  和最终 waypoint 的欧氏误差共同判定：
  - Future 服务端异常：记录异常并终止；
  - Future 返回 `False`：`timeout_phase=move_future_result`；
  - Future 返回 `True` 但终点误差大于容差：`stalled`；
  - Future 返回 `True` 且终点误差不大于容差：进入 hover。
- 终点容差默认 `1.0 m`，定义于
  `evaluate_traveluav_smoke.py:1420-1426`。

AirSim 官方实现最终返回 `waiter.isComplete()`；只有内部路径状态到达末端并调用
`waiter.complete()` 才返回 `True`，因此不能把 Future 返回 `False` 的动作仅凭
几何误差较小改判为完成：
<https://github.com/microsoft/AirSim/blob/main/AirLib/src/vehicles/multirotor/api/MultirotorApiBase.cpp#L2631-L2642>
和
<https://github.com/microsoft/AirSim/blob/main/AirLib/src/vehicles/multirotor/api/MultirotorApiBase.cpp#L2758-L2799>。

### 4. 使用 split JSONL 中的完整 instruction

- 修改前：`evaluate_traveluav_smoke.py:178-188` 只从轨迹目录内的
  `merged_data.json`/`object_description.json` 回退读取 instruction；
  split runner 原 `290-318` 行没有把 split JSONL 传给评估器。
- 修改后：`evaluate_traveluav_smoke.py:104-133` 按
  `(scene_id, trajectory_id)` 加载 JSONL 中的完整 `instruction`，并检查同一轨迹
  是否存在冲突文本；`210-235` 强制命中后再构造 case。
- `run_traveluav_closed_loop_split.sh:306-334` 把当前 `${SPLIT_FILE}` 传到单场景
  runner；单场景 runner 在 `389-395` 检查文件并传给 Python。
- 实测 train JSONL 得到 962 条唯一轨迹映射；A/B 使用的 10 条全部命中。
  第一条输出 summary 的 `instruction_source` 为
  `split_jsonl:.../train.jsonl`，instruction 长度为 522 字符。

### 5. hover 超时先重试，阈值放宽

- 修改前：`evaluate_traveluav_smoke.py:705-738` 首次 hover RPC 或稳定等待失败
  即终止，默认阈值为 `0.15 m/s`。
- 修改后：`evaluate_traveluav_smoke.py:406-423` 对 hover Future 调用 `get()`；
  `831-884` 在失败后执行 `cancelLastTask()`、重新启用 API control、重新 arm，
  再发送 hover。
- 默认重试 2 次、稳定速度阈值 `0.25 m/s`，定义于
  `evaluate_traveluav_smoke.py:1427-1449`。
- 日志记录 `hover_attempts`、`hover_errors` 和最终速度。mock 测试已验证：
  第一次 hover RPC 超时、第二次稳定时，动作继续完成而不终止。

### 6. 扩展飞行与碰撞日志

- `evaluate_traveluav_smoke.py:385-395` 保存 AirSim `CollisionInfo` 全部字段：
  `has_collided`、`object_name`、`object_id`、`time_stamp`、
  `penetration_depth`、`impact_point`、`position`、`normal`。
- `evaluate_traveluav_smoke.py:480-544` 保存：
  `landed_state`、`landed_state_name`、`ready`、`ready_message`、
  `can_arm`、`api_control_enabled` 和 rotor 状态。
- AirSim Python API 没有权威 `isArmed()` 查询。本次同时记录
  `armDisarm(True)` 的返回值 `arm_command_succeeded`，并把 rotor speed 推断值
  明确命名为 `armed_estimate`，不把推断伪装成官方 arm 状态。

## 严格 A/B

两组均使用同一版修改后代码、同一 checkpoint、同一 train JSONL、同一 10 个
trajectory ID，以及相同的 200 步、`velocity=1`、`move_timeout_s=5`、
`waypoint_count=5`、终点容差和 hover 参数。程序比较两个 `config.json` 后确认：
除 `clock_speed` 和输出目录外，参数完全一致；trajectory ID 顺序完全一致。

| 指标 | ClockSpeed=10 | ClockSpeed=1 |
|---|---:|---:|
| 轨迹数 | 10 | 10 |
| Collision | 0 | 0 |
| Timeout | 10 | 10 |
| Future exception | 0 | 0 |
| NE | 250.101632 | 250.245917 |
| SR / OSR / SPL | 0 / 0 / 0 | 0 / 0 / 0 |
| 平均终止动作墙钟时间 | 0.5105 s | 4.5714 s |
| 平均预测路径长度 | 10.1973 m | 10.0754 m |

输出目录：

- `sim_eval_outputs/20260624_150500_strict_ab_clock10_train_closed_loop`
- `sim_eval_outputs/20260624_150500_strict_ab_clock1_train_closed_loop`

本次 A/B 没有出现旧结果中的碰撞，但两组最终均被
`timeout_phase=move_future_result` 终止。证据表明当前主导限制是
`move_timeout_s=5`：模型终止动作多为约 4 m，`velocity=1 m/s` 时还需要加速、
转向和终点收敛；ClockSpeed=1 下终止动作平均约 4.57 秒墙钟时间，AirSim 内部
路径 waiter 随后返回未完成。下一轮应单独对 `move_timeout_s` 做对照，例如按
`path_length / velocity + margin` 动态设置；本次未擅自改变该参数，以保证
ClockSpeed A/B 只有一个自变量。

## 验证

- Python：`py_compile` 通过。
- Shell：两个 runner 的 `bash -n` 通过。
- Git：相关文件 `git diff --check` 通过。
- 参数链：split dry-run 已确认 JSONL、ClockSpeed、终点容差、hover 参数和
  10 个 trajectory ID 全部传到评估器。
- 运行结束后未残留 TravelUAV server 或 BrushifyCountryRoads 仿真进程。

## 后续修复：动态超时与航向控制

### 动态移动超时

固定 `move_timeout_s=5` 会把约 4–5 m、`velocity=1 m/s` 的动作卡在理论最短
运动时间附近，没有给加速、偏航和终点收敛留余量。现在
`evaluate_traveluav_smoke.py:739-783` 使用：

```text
nominal_translation = path_length / velocity
nominal_yaw = abs(yaw_delta) / assumed_yaw_rate
nominal = max(nominal_translation, nominal_yaw)
timeout = clamp(max(minimum, nominal * scale + margin), maximum)
```

默认参数：

- minimum：`5 s`
- scale：`1.5`
- margin：`3 s`
- assumed yaw rate：`45 deg/s`
- maximum：`30 s`

参数定义位于 `evaluate_traveluav_smoke.py:1526-1549`；单场景 runner 的定义和
传递位于 `run_traveluav_closed_loop_eval.sh:60-65,199-210,256-267,361-372`；
split runner 位于
`run_traveluav_closed_loop_split.sh:32-36,94-105,140-151,344-355`。
每个 `model_steps/*.json` 都记录完整 `move_timeout` 计算项。

真实单动作验证：

- 输出：`sim_eval_outputs/20260624_152500_dynamic_timeout_1step`
- 路径长度：`3.6477 m`
- 动态 timeout：`8.4715 s`
- AirSim Future 完成时间：`4.9359 s`
- Future 返回：`True`
- 终点误差：`0.4627 m`
- hover：第一次成功

因此固定 5 秒导致的过早终止已经消除。

### drivetrain 修正

旧代码把 HAD 计算出的绝对世界 yaw 传给
`airsim.DrivetrainType.ForwardOnly`。AirSim 的 ForwardOnly 路径控制还会把
路径航向叠加到 yaw，导致绝对 yaw 被二次变换。现在
`evaluate_traveluav_smoke.py:785-940` 默认使用
`MaxDegreeOfFreedom`，完整保留 HAD 输出的世界 yaw；仍可通过
`--drivetrain forward_only` 显式复现实验。

参数位置：

- Python：`evaluate_traveluav_smoke.py:1515-1524`
- 单场景 runner：`run_traveluav_closed_loop_eval.sh:77,210,267,372`
- split runner：`run_traveluav_closed_loop_split.sh:36,105,151,355`

### Future 和终点误差联合判定

AirSim 在部分动作中 Future 正常 resolve、返回 `False`，但无人机实际终点误差
只有约 `0.50 m`。现在 `evaluate_traveluav_smoke.py:843-906` 严格区分：

- Future 异常：终止并记录异常；
- 碰撞：立即终止；
- Future 正常 resolve 且终点误差不超过 `1 m`：接受完成，同时保留原始布尔值；
- Future 返回 `False` 且终点误差超限：timeout；
- Future 返回 `True` 但终点误差超限：stalled。

日志新增 `move_completion_basis`，明确动作是
`future_true_and_endpoint_within_tolerance` 还是
`future_resolved_and_endpoint_within_tolerance`。

## 完整轨迹实测

动态超时、ClockSpeed=1、MaxDegreeOfFreedom 和联合完成判定启用后：

| trajectory | 完成动作数 | 预测路径长度 | 最终距目标 | 终止原因 |
|---|---:|---:|---:|---|
| `02f80d28-...` | 10 | 44.27 m | 110.58 m | collision: Landscape_0 |
| `00cae84c-...` | 7 | 30.81 m | 160.36 m | collision: Landscape_0 |
| `03157986-...` | 21 | 103.89 m | 88.47 m | collision: Landscape_0 |
| `00dcf5f0-...` | 3 | 12.22 m | 297.00 m | collision: Landscape_0 |

其中 `03157986-...` 在修复前只运行 6 步左右；修复后连续完成 20 个动作，第
21 个动作才因真实地形碰撞停止。没有一条因固定 move timeout 终止。

### 为什么目前仍没有 SR 成功轨迹

这不是继续放大 timeout 能解决的问题：

1. checkpoint 自身记录的 `best_val_loss=0.7554013129798168`，不是接近零。
2. 用训练保存的 `02f80d28-...` step-0 原始图像离线推理，模型输出
   `[2.255,-0.141,-2.367,-2.516]`，标签为
   `[1.127,0.147,-4.665,-2.329]`；仅 `dz` 误差已经是 `2.298 m`。
3. 同一姿态的实时图像与训练原图差异明显。`00cae84c-...`：
   - 前视相关系数：`0.3566`
   - 下视相关系数：`0.1267`
   - 训练原图 step-0 推理 XYZ L2 误差：`0.143`
   - 实时图 step-0 输出从接近标签变成
     `[3.641,0.238,-1.805,-0.448]`
4. `02f80d28-...` 的专家前 8 步持续从 `z=-48.58 m` 爬升到约
   `z=-87.18 m`；实时模型到碰撞前只到约 `z=-53 m`，所以撞上
   `Landscape_0` 与动作误差一致。
5. 所有实测 stop probability 约为 `0.002`，远低于 `0.3`；即使之后到达目标
   邻域，stop head 是否能正确触发仍需单独校准。

因此当前可以确认“控制层不会再无故只跑几步”，但不能把一次真实碰撞或未到达
目标的 rollout 伪报为成功。下一阶段应优先解决实时图像与训练图像的渲染域差异，
并复核 stop head 的类别不平衡和阈值，而不是继续修改运动 timeout。

### 已撤回的无效实验

曾按 TravelUAV `assist.py` 的 7 m 下视规则尝试 DepthPerspective 辅助，但当前
场景接口持续返回 `1.0 m`，并导致无人机错误连续爬升。该代码和全部 CLI 参数已
从最终实现中完全撤回；最终 evaluator 不包含隐式安全干预。
