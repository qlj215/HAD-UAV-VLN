# HAD-TravelUAV 闭环仿真工作流程代码报告

本文只依据当前 laptopRTX3070 上的代码和配置进行说明，核心代码路径为：

- `/home/qlj/h3c_pro/HAD-UAV-VLN/engine/evaluate_traveluav_smoke.py`
- `/home/qlj/h3c_pro/HAD-UAV-VLN/scripts/run_traveluav_closed_loop_eval.sh`
- `/home/qlj/h3c_pro/HAD-UAV-VLN/engine/evaluate.py`
- `/home/qlj/h3c_pro/TravelUAV/airsim_plugin/AirVLNSimulatorServerTool.py`
- `/home/qlj/h3c_pro/TravelUAV/airsim_plugin/AirVLNSimulatorClientTool.py`

结论先行：

1. 每个未终止的 step 都会重新从 AirSim 读取当前图像、当前位置、当前姿态，再调用 HAD checkpoint 生成一次 `pred_action=[dx, dy, dz, dyaw]`。
2. HAD 代码会把这一步动作确定性地转换成 `next_world_position` 和最多 5 个 `world_waypoints`。
3. 但是 HAD 代码不会把无人机直接瞬移到 `next_world_position`。它调用 AirSim 的 `moveOnPath`，把 `path`、`velocity`、`yaw_mode`、`drivetrain`、`lookahead` 等参数交给 AirSim 执行。
4. 每步真实结束位置来自 AirSim 返回的状态，不一定等于计划的 `next_world_position`。
5. 当前 `collision=True` 包含三种情况：AirSim 明确碰撞、单步移动超时、单步移动卡住。后两种不一定有 `collision_step/collision_log_index/collision_object_name`。
6. 当前 engine 中 `--stop_on_collision` 只被 argparse 声明，真正的终止逻辑是 `if collision: break`，因此只要 `collision` 变量为 True，当前轨迹就会终止。

## 1. 入口脚本如何启动仿真

常用入口脚本是：

```bash
/home/qlj/h3c_pro/HAD-UAV-VLN/scripts/run_traveluav_closed_loop_eval.sh
```

这个脚本的用途在文件头部说明为“Run one TravelUAV closed-loop evaluation scene with HAD”，输出目录规则为：

```text
/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/<timestamp>_<run_name>
```

脚本会选择 Python 解释器：

1. 如果环境变量 `PYTHON` 存在，使用它。
2. 否则优先使用 `${REPO_ROOT}/.venv/bin/python`。
3. 否则尝试 `/root/miniconda3/envs/had/bin/python`。
4. 否则尝试 `/root/miniconda3/bin/python`。
5. 最后退回 `python3`。

代码证据：`scripts/run_traveluav_closed_loop_eval.sh:102-114`。

默认 checkpoint 和 vocab 的选择逻辑：

```text
DEFAULT_MODEL_DIR=${REPO_ROOT}/local_checkpoints/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15
DEFAULT_EXPERIMENT_DIR=/root/autodl-tmp/HAD_UAV_VLN_experiments/.../lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15
DEFAULT_TARGET_ALIGNED_VOCAB=/root/autodl-tmp/TravelUAVProcessedData_target_aligned/vocab.json
```

如果 `DEFAULT_EXPERIMENT_DIR/checkpoints/best_model.pth` 存在，则默认使用它；否则使用 `local_checkpoints/.../best_model.pth`。vocab 也类似，优先用 target-aligned vocab，否则用 local checkpoint 目录下的 `vocab.json`。

代码证据：`scripts/run_traveluav_closed_loop_eval.sh:115-131`。

脚本最终组装并执行：

```bash
python engine/evaluate_traveluav_smoke.py \
  --checkpoint ... \
  --vocab_path ... \
  --traveluav_root ... \
  --env_root ... \
  --raw_data_dir ... \
  --scene ... \
  --num_trajectories ... \
  --start_index ... \
  --max_steps ... \
  --success_threshold ... \
  --stop_threshold ... \
  --device ... \
  --server_ip ... \
  --server_port ... \
  --gpu_id ... \
  --scene_wait_s ... \
  --airsim_connect_timeout ... \
  --move_timeout_s ... \
  --velocity ... \
  --waypoint_count ... \
  --record_image_stride ... \
  --record_image_width ... \
  --record_image_format ... \
  --record_image_quality ... \
  --output_dir ... \
  --no-spawn_target
```

代码证据：`scripts/run_traveluav_closed_loop_eval.sh:279-307`。

如果指定：

- `--start-server`，脚本追加 `--start_server`。
- `--keep-server`，脚本追加 `--keep_server`。
- `--spawn-target`，脚本追加 `--spawn_target`。
- `--require-target-spawn`，脚本追加 `--require_target_spawn`。
- `--stop-on-collision`，脚本追加 `--stop_on_collision`。
- `--record-images`，脚本追加 `--record_images`。
- `--trajectory-ids`，脚本追加 `--trajectory_ids ...`。
- `--extra-arg`，脚本原样追加额外 engine 参数。

代码证据：`scripts/run_traveluav_closed_loop_eval.sh:309-332`。

## 2. engine 的参数入口

`engine/evaluate_traveluav_smoke.py` 的 `parse_args()` 定义了仿真和评估参数。

关键参数如下：

| 参数 | 默认值 | 代码含义 |
|---|---:|---|
| `--checkpoint` | 必填 | HAD checkpoint `.pth` |
| `--vocab_path` | 必填 | checkpoint 对应的 vocab |
| `--traveluav_root` | `/home/qlj/h3c_pro/TravelUAV` | TravelUAV 仓库根目录 |
| `--env_root` | `/home/qlj/TravelUAV_envs` | Unreal/AirSim 场景可执行文件根目录 |
| `--raw_data_dir` | `/home/qlj/datasets/TravelUAVData` | 原始 TravelUAVData 根目录 |
| `--scene` | `BrushifyCountryRoads` | 要评估的场景名 |
| `--num_trajectories` | `1` | 从 `start_index` 开始选取的轨迹数 |
| `--start_index` | `0` | 轨迹目录排序后的起始偏移 |
| `--trajectory_ids` | `None` | 指定精确轨迹目录名时使用 |
| `--output_dir` | `None` | 输出目录；为空时用 `sim_eval_outputs/<timestamp>` |
| `--device` | `auto` | `auto/cuda/cpu` |
| `--image_size` | `[224, 224]` | 送入模型前的图像 transform 尺寸 |
| `--max_inst_len` | `80` | 指令 token 最大长度 |
| `--uav_position_scale` | `100.0` | 位置输入归一化尺度 |
| `--success_threshold` | `20.0` | 成功距离阈值，单位按 AirSim 世界坐标米处理 |
| `--stop_threshold` | `0.3` | `stop_prob` 触发停止的阈值 |
| `--max_steps` | `20` | 单条轨迹最多闭环 step 数 |
| `--velocity` | `1.0` | 传给 AirSim `moveOnPath` 的速度 |
| `--waypoint_count` | `5` | 一步动作插值 waypoint 数量的输入参数 |
| `--move_timeout_s` | `5.0` | 单步 AirSim 移动超时时间 |
| `--stop_on_collision` | `False` | 目前只被 argparse 接收；engine 内没有按它做条件判断 |
| `--server_ip` | `127.0.0.1` | TravelUAV server IP |
| `--server_port` | `30000` | TravelUAV server port |
| `--gpu_id` | `0` | 启动场景时传给 TravelUAV server 的 GPU id |
| `--airsim_timeout` | `120.0` | AirSim RPC timeout |
| `--scene_wait_s` | `45.0` | 打开 Unreal 场景后等待 AirSim 的秒数 |
| `--start_server` | `False` | 是否由 evaluator 启动 TravelUAV server |
| `--server_wait_s` | `120.0` | 等待 TravelUAV server ping 成功的秒数 |
| `--keep_server` | `False` | 评估结束后是否保留 server 进程 |
| `--front_camera` | `FrontCamera` | 前视相机名 |
| `--down_camera` | `DownCamera` | 下视相机名 |
| `--record_images` | `False` | 是否保存模型视角 RGB 帧 |
| `--record_image_stride` | `1` | 每隔多少个模型 step 保存一次图 |
| `--record_image_width` | `384` | 保存图像的宽度；`<=0` 保持原尺寸 |
| `--record_image_format` | `jpg` | 保存格式：`jpg/jpeg/png/webp` |
| `--record_image_quality` | `80` | jpg/webp 质量 |
| `--spawn_target` | `False` | 是否尝试在 AirSim 中生成目标物体 |
| `--require_target_spawn` | `False` | 目标物体生成失败时是否直接报错 |

代码证据：`engine/evaluate_traveluav_smoke.py:891-936`。

## 3. 主函数整体执行顺序

`main()` 的顺序如下：

1. 解析参数，创建 `output_dir`。
2. 如果 `args.start_server=True`，启动 TravelUAV server，并等待 `ping` 成功。
3. 根据 `raw_data_dir/scene` 和 `trajectory_ids` 或 `num_trajectories/start_index` 选择轨迹。
4. 解析 device。
5. 加载 `WordVocabTokenizer(args.vocab_path)`。
6. 构建验证图像 transform：`get_val_transforms(tuple(args.image_size))`。
7. 从 checkpoint 构建模型：`build_model_from_checkpoint(args.checkpoint, device)`。
8. `model.eval()`，`model.to(device)`。
9. 打开 TravelUAV 场景并连接 AirSim。
10. 写出 `config.json`。
11. 对每条轨迹调用 `run_case(...)`。
12. 聚合全部轨迹结果，写出 `eval_trajectory.json`、`eval_overall.json`、`rollouts.jsonl`。
13. finally 中关闭场景；如果启动了 server 且没有 `--keep_server`，终止 server 进程。

代码证据：`engine/evaluate_traveluav_smoke.py:939-1028`。

## 4. TravelUAV server 和 AirSim 场景如何打开

如果传入 `--start_server`，`start_server(args)` 会启动：

```bash
python /home/qlj/h3c_pro/TravelUAV/airsim_plugin/AirVLNSimulatorServerTool.py \
  --port <server_port> \
  --root_path <env_root> \
  --gpus <gpu_id>
```

并设置：

```text
PYTHONPATH=<traveluav_root>
```

代码证据：`engine/evaluate_traveluav_smoke.py:249-264`。

`wait_for_socket()` 会循环创建 `msgpackrpc.Client`，调用 server 的 `ping`，直到成功或超过 `server_wait_s`。

代码证据：`engine/evaluate_traveluav_smoke.py:267-279`。

`open_scene(args)` 负责真正打开场景：

1. 连接 TravelUAV server：`msgpackrpc.Client(Address(args.server_ip, args.server_port))`。
2. 调用 `ping`。
3. 调用 `reopen_scenes(args.server_ip, [(args.scene, args.gpu_id)])`。
4. 从返回值取出 AirSim IP 和端口。
5. 等待 `scene_wait_s` 秒。
6. 创建 `airsim.MultirotorClient(ip=ip, port=port, timeout_value=args.airsim_timeout)`。
7. `confirmConnection()`。
8. `enableApiControl(True)`。
9. `armDisarm(True)`。
10. `simPause(True)`。

代码证据：`engine/evaluate_traveluav_smoke.py:282-300`。

TravelUAV 的 `AirVLNSimulatorServerTool.py` 负责 msgpackrpc server，`serve()` 会创建 `msgpackrpc.Server(EventHandler())` 并监听 `HOST, PORT`。

代码证据：`/home/qlj/h3c_pro/TravelUAV/airsim_plugin/AirVLNSimulatorServerTool.py:654-662`。

当前 TravelUAV settings 文件中场景 AirSim 以 multirotor 模式运行，车辆类型为 SimpleFlight：

```json
"SimMode": "Multirotor",
"ClockSpeed": 10,
"ViewMode": "NoDisplay",
"PhysiceEngineName": "ExternalPhysicsEngine",
"VehicleType": "SimpleFlight",
"LockStep": true
```

代码证据：`/home/qlj/h3c_pro/TravelUAV/airsim_plugin/settings/30001/settings.json`。

注意：TravelUAV 仓库中没有 `moveOnPath` 服务端实现；HAD 侧通过 AirSim Python client 发 RPC，实际路径跟随由 AirSim/Unreal 场景执行。能从代码严格确认的是“传给 AirSim 的参数”和“HAD 如何读取 AirSim 状态”，不能把 AirSim 内部控制方程写成 HAD 代码事实。

## 5. 轨迹数据如何加载成 TrajectoryCase

轨迹目录来源：

- 如果传入 `--trajectory_ids`，直接使用 `raw_data_dir/scene/<trajectory_id>`。
- 否则调用 `list_trajectory_dirs(raw_data_dir, scene)`，列出 `raw_data_dir/scene` 下包含以下三个文件的目录：
  - `merged_data.json`
  - `mark.json`
  - `object_description.json`

代码证据：`engine/evaluate_traveluav_smoke.py:164-175`、`221-246`。

每个有效目录通过 `load_case()` 加载：

1. 读取 `merged_data.json`、`mark.json`、`object_description.json`。
2. `trajectory = merged.get("trajectory") or []`。
3. `raw_states = merged.get("trajectory_raw") or merged.get("trajectory_raw_detailed") or []`。
4. 如果 `trajectory` 或 `raw_states` 长度小于 2，返回 `None`。
5. 指令优先来自 `merged_data.conversations[0].value`；若没有，则从 `object_description.json` 或 `mark.object_name` 兜底。
6. 起点世界坐标：
   - 优先 `mark.get("start")`
   - 否则 `raw_states[0].get("position")`
7. 目标世界坐标：`mark["target"]["position"]`。
8. 起点姿态：`raw_states[0].get("orientation", [0,0,0,1])`。
9. 起点四元数转欧拉角，再转 `start_rotation`。
10. `final_tdata = trajectory[-1]`。
11. 用 `final_tdata[0], final_tdata[1]` 构建 `target_basis` 和 `target_align_yaw`。
12. 专家轨迹世界坐标：`gt_positions = np.asarray([state["position"] for state in raw_states])`。
13. 专家终点：`gt_final_position = gt_positions[-1]`。

代码证据：`engine/evaluate_traveluav_smoke.py:178-218`。

`TrajectoryCase` 保存的字段包括：

```text
scene, traj_id, traj_dir, instruction, instruction_source,
start_position, start_orientation, target_position,
gt_positions, gt_final_position,
target_basis, target_align_yaw,
start_rotation, start_yaw, mark
```

代码证据：`engine/evaluate_traveluav_smoke.py:43-59`。

## 6. checkpoint 如何加载成模型

闭环仿真复用 `engine/evaluate.py` 中的：

```python
build_model_from_checkpoint(ckpt_path, device)
```

它的步骤：

1. `torch.load(ckpt_path, map_location=device, weights_only=True)`。
2. 从 checkpoint 中取 `config`。
3. 从 config 取 vision/language/height/fusion/policy/auxiliary/ablation/position 配置。
4. 如果 `position.enabled=True`，构建 `HADVLNModelwithPosition`。
5. 否则构建 `HADVLNModel`。
6. `model.load_state_dict(ckpt["model_state_dict"])`。
7. 打印 checkpoint epoch。

代码证据：`engine/evaluate.py:279-339`。

闭环仿真主函数随后执行：

```python
model.eval()
model.to(device)
```

代码证据：`engine/evaluate_traveluav_smoke.py:962-964`。

## 7. 每条轨迹开始时如何放置无人机

每条轨迹进入 `run_case()` 后，第一步是：

```python
reset_vehicle(client, case)
```

代码证据：`engine/evaluate_traveluav_smoke.py:672-683`。

`reset_vehicle()` 做：

1. `enableApiControl(True)`。
2. `armDisarm(True)`。
3. `simPause(True)`。
4. `simSetKinematics(airsim_kinematics(case.start_position, case.start_orientation), ignore_collision=True)`。
5. `simContinueForFrames(1)`。
6. `simPause(True)`。

代码证据：`engine/evaluate_traveluav_smoke.py:326-332`。

这里是当前闭环流程中明确直接设置无人机状态的位置。`airsim_kinematics()` 会把起点 position、orientation 写进 `airsim.KinematicsState`，并把线速度和角速度都置零。

代码证据：`engine/evaluate_traveluav_smoke.py:316-323`。

如果 `args.spawn_target=True`，会尝试根据 `case.mark["object_name"]` 在 `case.target_position` 处生成名为 `had_target_object` 的目标物体。如果失败且 `require=True`，抛异常；否则只打印 warning。

代码证据：`engine/evaluate_traveluav_smoke.py:335-369`。

## 8. 每个闭环 step 的完整执行流程

`run_case()` 初始化：

```python
observations = []
pred_positions = []
oracle_success = False
success = False
early_end = False
stop_step = None
collision = False
first_collision_step = None
first_collision_log_index = None
first_collision_object_name = None
```

然后读取初始 AirSim 状态：

```python
initial_position, initial_yaw, initial_payload = current_position_yaw(client)
observations.append({"sensors": {"state": initial_payload}})
pred_positions.append(initial_position)
```

代码证据：`engine/evaluate_traveluav_smoke.py:686-703`。

接下来进入：

```python
for step in range(args.max_steps):
```

代码证据：`engine/evaluate_traveluav_smoke.py:705`。

下面逐行描述每个 step。

### 8.1 读取当前世界状态

每个 step 首先调用：

```python
position, yaw, _ = current_position_yaw(client)
```

`current_position_yaw()` 内部调用 `state_payload()`。

`state_payload()` 从 AirSim 读取：

- `client.getMultirotorState()`
- `client.simGetCollisionInfo()`
- `kinematics_estimated.position`
- `linear_velocity`
- `linear_acceleration`
- `orientation`
- `angular_velocity`
- `angular_acceleration`
- `collision.has_collided`
- `collision.object_name`

代码证据：`engine/evaluate_traveluav_smoke.py:372-395`。

### 8.2 当前世界坐标转成模型需要的位置/yaw输入

当前世界位置先转为起点局部系：

```python
start_local = case.start_rotation.T @ (position - case.start_position)
```

然后再转成目标局部方位系：

```python
target_local_position = transform_point(start_local, case.target_basis)
```

其中：

```python
transform_point(point, basis_cols) = basis_cols.T @ point
```

代码证据：`engine/evaluate_traveluav_smoke.py:148-153`、`706-709`。

当前 yaw 转目标局部 yaw：

```python
current_target_yaw = wrap_angle_rad(wrap_angle_rad(yaw - case.start_yaw) - case.target_align_yaw)
```

高度输入为：

```python
altitude = abs(float(position[2]))
```

代码证据：`engine/evaluate_traveluav_smoke.py:706-710`。

### 8.3 从 AirSim 获取模型当前视角图片

调用：

```python
front_img, down_img = get_rgb_pair(client, args.front_camera, args.down_camera)
```

`get_rgb_pair()` 通过 `client.simGetImages()` 请求两个 `Scene` 图像：

- `args.front_camera`，默认 `FrontCamera`
- `args.down_camera`，默认 `DownCamera`

如果返回图片为空，会抛异常。

代码证据：`engine/evaluate_traveluav_smoke.py:398-411`。

如果 `args.record_images=True`，会保存这一步模型看到的前视/下视图像。保存路径相对轨迹目录为：

```text
images/model/front/<step>.<suffix>
images/model/down/<step>.<suffix>
```

保存间隔由 `record_image_stride` 控制，图片宽度由 `record_image_width` 控制，格式和质量由 `record_image_format/record_image_quality` 控制。

代码证据：`engine/evaluate_traveluav_smoke.py:539-568`。

### 8.4 构造 HAD 模型输入

`build_model_inputs()` 构造：

```python
tensors = {
  "front": transform(front_img).unsqueeze(0).to(device),
  "down": transform(down_img).unsqueeze(0).to(device),
  "inst": torch.tensor([tokenizer(instruction, max_inst_len)], dtype=torch.long, device=device),
  "alt": torch.tensor([altitude], dtype=torch.float32, device=device),
  "step_ids": torch.tensor([step_id], dtype=torch.long, device=device),
}
```

如果模型是 `HADVLNModelwithPosition`，还会加：

```python
yaw_feat = [sin(target_local_yaw), cos(target_local_yaw)]
position_feat = target_local_position / max(abs(position_scale), 1e-6)
tensors["target_yaw"] = yaw_feat
tensors["uav_position"] = position_feat
```

代码证据：`engine/evaluate_traveluav_smoke.py:414-441`。

### 8.5 调用 HAD checkpoint 生成动作

每个 step 调用：

```python
pred_action, stop_prob, pred_extra = predict_action(model, inputs)
```

`predict_action()` 内部：

- 如果是 `HADVLNModelwithPosition`，调用：

```python
model(front, down, inst, alt, target_yaw, uav_position, return_features=False, step_ids=step_ids)
```

- 否则调用：

```python
model(front, down, inst, alt, return_features=False, step_ids=step_ids)
```

然后取：

```python
action = outputs["pred_action"][0].detach().float().cpu().numpy()
```

如果存在 `stop_logit`，计算：

```python
stop_prob = sigmoid(stop_logit[0])
```

并记录 `gate_weight`、`stop_logit` 到 `pred_extra`。

代码证据：`engine/evaluate_traveluav_smoke.py:444-478`。

因此，每个未终止 step 都会重新调用 checkpoint 模型一次，输出当前 step 的 `pred_action=[dx,dy,dz,dyaw]`。

### 8.6 将模型动作转成世界系目标点和航路点

动作转换函数是：

```python
waypoints, target_yaw, transform_payload = waypoints_from_action(...)
```

代码证据：`engine/evaluate_traveluav_smoke.py:730-736`。

内部计算：

```python
target_delta = pred_action[:3]
start_delta = inverse_transform_delta(target_delta, case.target_basis)
world_delta = case.start_rotation @ start_delta
next_position = current_position + world_delta
```

其中：

```python
inverse_transform_delta(target_delta, basis_cols) = basis_cols @ target_delta
```

代码证据：`engine/evaluate_traveluav_smoke.py:481-491`、`152-153`。

所以空间变换公式是：

```text
delta_start = B_t * delta_target
delta_world = R0 * delta_start
p_world_next = p_world_current + delta_world
```

`dyaw` 的转换是：

```python
current_target_yaw = wrap_angle_rad(wrap_angle_rad(current_yaw - case.start_yaw) - case.target_align_yaw)
next_target_yaw = wrap_angle_rad(current_target_yaw + pred_action[3])
next_world_yaw = wrap_angle_rad(case.start_yaw + case.target_align_yaw + next_target_yaw)
```

代码证据：`engine/evaluate_traveluav_smoke.py:493-495`。

然后把当前点到 `next_position` 插值成 waypoint：

```python
count = max(int(waypoint_count), 2)
waypoints = [
    current_position + (next_position - current_position) * (idx / (count - 1))
    for idx in range(1, count)
]
while len(waypoints) < 5:
    waypoints.append(next_position.copy())
waypoints = waypoints[:5]
```

代码证据：`engine/evaluate_traveluav_smoke.py:497-504`。

重要细节：

- 如果 `waypoint_count=5`，会生成 4 个不同插值点：`1/4, 2/4, 3/4, 4/4`，其中最后一个是 `next_position`。
- 由于 `while len(waypoints) < 5`，会继续补 `next_position`，直到列表长度为 5。
- 因此实际传给 AirSim 的 `path` 长度通常是 5，但最后可能有重复的 `next_position`。

`transform_payload` 会把这些信息保存到 step json：

- `pred_action_target_local`
- `delta_target_local`
- `delta_start_local`
- `delta_world`
- `next_world_position`
- `next_world_yaw`
- `world_waypoints`
- `target_basis`
- `start_rotation`
- `target_align_yaw`
- `start_yaw`
- `formula`

代码证据：`engine/evaluate_traveluav_smoke.py:505-527`。

### 8.7 先检查 oracle_success

在真正移动前，代码会遍历本 step 计划的所有 waypoint：

```python
for waypoint in waypoints:
    if norm(waypoint - case.target_position) <= args.success_threshold:
        oracle_success = True
```

代码证据：`engine/evaluate_traveluav_smoke.py:737-739`。

这表示：只要模型计划路径上的任一 waypoint 距目标点不超过 `success_threshold`，就记录 `oracle_success=True`。它不要求模型已经显式输出 stop。

### 8.8 调用 AirSim 执行这一步路径

执行函数是：

```python
step_observations, step_collision = move_on_waypoints(
    client=client,
    waypoints=waypoints,
    target_yaw=target_yaw,
    velocity=args.velocity,
    timeout_s=args.move_timeout_s,
)
```

代码证据：`engine/evaluate_traveluav_smoke.py:741-747`。

`move_on_waypoints()` 做：

1. 把 numpy waypoint 转为 `airsim.Vector3r`。
2. 构造 `airsim.YawMode(is_rate=False, yaw_or_rate=degrees(target_yaw))`。
3. `enableApiControl(True)`。
4. `armDisarm(True)`。
5. `simPause(False)`。
6. 调用 AirSim：

```python
client.moveOnPathAsync(
    path=path,
    velocity=velocity,
    drivetrain=airsim.DrivetrainType.ForwardOnly,
    yaw_mode=yaw_mode,
    lookahead=3,
    adaptive_lookahead=1,
)
```

代码证据：`engine/evaluate_traveluav_smoke.py:579-598`。

AirSim Python package 中 `moveOnPathAsync()` 的实现是：

```python
return self.client.call_async(
    'moveOnPath',
    path,
    velocity,
    timeout_sec,
    drivetrain,
    yaw_mode,
    lookahead,
    adaptive_lookahead,
    vehicle_name
)
```

这说明 HAD 侧只是通过 RPC 把路径和控制参数交给 AirSim。实际飞行轨迹由 AirSim/Unreal 环境执行。HAD 代码本身没有把无人机瞬移到 `next_world_position`。

TravelUAV 自带 client 示例也使用同样参数风格：

```python
velocity = 1
drivetrain = airsim.DrivetrainType.ForwardOnly
yaw_mode = airsim.YawMode(is_rate=False)
lookahead = 3
adaptive_lookahead = 1
airsim_client.moveOnPathAsync(...)
```

代码证据：`/home/qlj/h3c_pro/TravelUAV/airsim_plugin/AirVLNSimulatorClientTool.py:294-315`。

### 8.9 AirSim 移动过程中如何判断是否完成当前 step

`move_on_waypoints()` 调用 `moveOnPathAsync()` 后，进入循环：

```python
while current_idx < len(path):
```

每 0.02 秒：

1. 读取 AirSim 状态。
2. 读取当前位置。
3. 检查 AirSim 是否报告碰撞。
4. 检查是否卡住。
5. 检查是否已经经过当前 waypoint。

代码证据：`engine/evaluate_traveluav_smoke.py:607-637`。

完成 waypoint 的判据不是“距离小于某个阈值”，而是：

```python
target = path[current_idx]
new_distance = norm(position - target)
if new_distance > last_distance:
    results.append({"sensors": {"state": payload}})
    current_idx += 1
    last_distance = inf
else:
    last_distance = new_distance
```

代码证据：`engine/evaluate_traveluav_smoke.py:626-636`。

含义是：当当前位置到当前 waypoint 的距离开始由小变大，代码认为无人机已经经过这个 waypoint，于是切换到下一个 waypoint。

因此，当前 step 完成条件是：

```text
current_idx >= len(path)
```

也就是所有 path waypoint 都被这个“距离开始变大”的判据推进过。

### 8.10 为什么 velocity 会影响结果

模型动作和插值 waypoint 是确定的，但 `velocity` 传给了 AirSim 的 `moveOnPath`，不是用于重新计算 `next_world_position`。

代码证据：

- `next_world_position` 在 `waypoints_from_action()` 中由 `pred_action` 和坐标变换确定：`engine/evaluate_traveluav_smoke.py:488-512`。
- `velocity` 只在 `move_on_waypoints()` 中传给 `client.moveOnPathAsync(...)`：`engine/evaluate_traveluav_smoke.py:591-598`。

这意味着：

```text
HAD 计划点: 由模型动作和矩阵变换确定
实际飞行: 由 AirSim moveOnPath 按 path + velocity + yaw/drivetrain/lookahead 执行
```

所以速度大小不会改变 `model_steps/*.json` 中记录的计划 `next_world_position`，但会影响 AirSim 状态随时间如何变化，从而影响：

- 当前 step 是否在 `move_timeout_s` 内走完所有 waypoint；
- 是否触发卡住判断；
- 是否在路径跟随过程中碰到场景碰撞体；
- 最终写入 `log/*.json` 和 `pred_positions` 的真实位置序列。

不能从当前 HAD 代码中推出 AirSim 内部的完整动力学方程，因为 TravelUAV 仓库没有 `moveOnPath` 服务端实现；这里严格能确认的是：`velocity` 被作为 RPC 参数传给 AirSim，而实际位置来自 AirSim 返回状态。

## 9. collision=True 的全部代码路径

当前 `collision=True` 有三类来源。

### 9.1 AirSim 明确报告碰撞

`state_payload()` 保存：

```python
"collision": {
    "has_collided": bool(collision_info.has_collided),
    "object_name": str(collision_info.object_name),
}
```

代码证据：`engine/evaluate_traveluav_smoke.py:372-380`。

`collision_from_payload()` 读取：

```python
collision_info = payload.get("collision") or {}
has_collided = bool(collision_info.get("has_collided", False))
object_name = collision_info.get("object_name")
```

代码证据：`engine/evaluate_traveluav_smoke.py:570-576`。

在 `move_on_waypoints()` 中：

```python
if has_collided:
    results.append({"sensors": {"state": payload}})
    collision = True
    break
```

代码证据：`engine/evaluate_traveluav_smoke.py:614-618`。

这种情况会在 `run_case()` 中记录：

```python
first_collision_step = step
first_collision_log_index = log_index
first_collision_object_name = object_name
```

代码证据：`engine/evaluate_traveluav_smoke.py:756-760`。

### 9.2 单步移动超过 move_timeout_s

`move_on_waypoints()` 中：

```python
if time.perf_counter() - start > timeout_s:
    collision = True
    break
```

代码证据：`engine/evaluate_traveluav_smoke.py:607-610`。

这种情况只是把局部变量 `collision` 设为 True，不一定会有 AirSim `has_collided=True`，因此可能没有 `collision_step/collision_log_index/collision_object_name`。

### 9.3 单步移动卡住或几乎不动

`move_on_waypoints()` 中：

```python
recent_positions.append(position)
if len(recent_positions) > 20:
    moved = norm(recent_positions[-1] - recent_positions[-20])
    if moved < 0.1:
        collision = True
        break
```

代码证据：`engine/evaluate_traveluav_smoke.py:619-624`。

循环每次 `time.sleep(0.02)`，所以这是约 20 次采样窗口内的位移判断。代码没有把这类情况命名为 `stuck`，而是直接设为 `collision=True`。

### 9.4 collision 如何传回 run_case 并终止轨迹

`move_on_waypoints()` 返回：

```python
return results, collision
```

代码证据：`engine/evaluate_traveluav_smoke.py:638-641`。

`run_case()` 接收后：

```python
collision = collision or step_collision
```

代码证据：`engine/evaluate_traveluav_smoke.py:741-748`。

每个 step 写 `model_steps/<step>.json` 时保存：

```python
"collision": collision,
"collision_object_name": first_collision_object_name,
```

代码证据：`engine/evaluate_traveluav_smoke.py:768-795`。

然后：

```python
if collision:
    break
```

代码证据：`engine/evaluate_traveluav_smoke.py:807-808`。

重要：当前代码中 `--stop_on_collision` 只在 `parse_args()` 中声明，且 wrapper 脚本会在用户指定时追加；但是 `run_case()` 中的终止逻辑没有判断 `args.stop_on_collision`。因此当前 engine 实际行为是：只要 `collision=True`，轨迹立即终止。

代码证据：`engine/evaluate_traveluav_smoke.py:918`、`scripts/run_traveluav_closed_loop_eval.sh:321-322`、`engine/evaluate_traveluav_smoke.py:807-808`。

## 10. stop/success/oracle_success 的逻辑

每步模型除了 `pred_action`，还可能输出 `stop_logit`。代码会计算：

```python
stop_prob = sigmoid(stop_logit[0])
```

若没有 `stop_logit`，`stop_prob=0.0`。

代码证据：`engine/evaluate_traveluav_smoke.py:469-478`。

每步移动后：

```python
distance_to_target = norm(final_position - case.target_position)
stopped = stop_prob >= args.stop_threshold
```

代码证据：`engine/evaluate_traveluav_smoke.py:765-767`。

如果 `stopped=True` 且 `distance_to_target <= success_threshold` 且还没有 `early_end`：

```python
success = True
stop_step = step
break
```

否则如果 `stopped=True` 但没有达到成功距离：

```python
early_end = True
if stop_step is None:
    stop_step = step
```

代码证据：`engine/evaluate_traveluav_smoke.py:797-804`。

如果 `early_end=True` 且此前/当前路径触发过 `oracle_success=True`：

```python
if early_end and oracle_success:
    break
```

代码证据：`engine/evaluate_traveluav_smoke.py:805-806`。

`status` 最终定义为：

```python
status = "success" if success else ("oracle" if oracle_success else "fail")
```

代码证据：`engine/evaluate_traveluav_smoke.py:820`。

## 11. 每条轨迹输出哪些文件

每条轨迹先写到：

```text
trajectories/running_<scene>_<traj_id>
```

结束后按 status 改名为：

```text
trajectories/<status>_<scene>_<traj_id>
```

代码证据：`engine/evaluate_traveluav_smoke.py:696-699`、`820-854`。

每条轨迹目录包含：

### 11.1 `model_steps/*.json`

每个闭环 step 写一个文件：

```text
model_steps/000000.json
model_steps/000001.json
...
```

字段包括：

- `step`
- `pred_action`
- `stop_prob`
- `stopped`
- `distance_to_target`
- `collision`
- `collision_object_name`
- `world_position`
- `world_yaw`
- `world_delta`
- `next_world_position`
- `next_world_yaw`
- `world_waypoints`
- `start_local_delta`
- `target_local_delta`
- `target_local_position`
- `target_local_yaw`
- `target_basis`
- `start_rotation`
- `target_align_yaw`
- `start_yaw`
- `frame_transform`
- `image_paths`
- `altitude`
- `gate_weight`
- `stop_logit`

代码证据：`engine/evaluate_traveluav_smoke.py:768-795`。

### 11.2 `log/*.json`

`save_rollout_logs()` 把 `observations` 写成：

```text
log/000000.json
log/000001.json
...
```

每个 log 是：

```json
{
  "sensors": {
    "state": {
      ...
    }
  }
}
```

代码证据：`engine/evaluate_traveluav_smoke.py:657-661`。

这些 log 对应 AirSim 实际返回的状态，不等价于模型计划 waypoint。由于每个 step 内可能经过多个 waypoint，也可能提前碰撞/超时，所以 `log` 数量不等于 `model_steps` 数量。

### 11.3 `ori_info.json`

内容为：

```json
{
  "ori_traj_dir": "...",
  "scene": "...",
  "trajectory_id": "..."
}
```

代码证据：`engine/evaluate_traveluav_smoke.py:662-669`。

### 11.4 `summary.json`

每条轨迹写：

- `scene`
- `trajectory_id`
- `status`
- `success`
- `oracle_success`
- `early_end`
- `collision`
- `collision_step`
- `collision_log_index`
- `collision_object_name`
- `stop_step`
- `num_steps`
- `instruction`
- `instruction_source`
- `start_position_world`
- `target_position_world`
- `start_to_target_distance`
- `start_to_gt_final_distance`
- `final_position`
- `target_position`
- `gt_final_position`
- `final_distance_to_target`
- `ne`
- `pred_path_length`
- `gt_path_length_minus_threshold`
- `spl`
- `output_dir`

代码证据：`engine/evaluate_traveluav_smoke.py:824-853`。

### 11.5 `images/model/...`

只有指定 `--record_images` 时才保存。否则 `image_paths={}`。

代码证据：`engine/evaluate_traveluav_smoke.py:539-568`。

## 12. 指标如何计算

每条轨迹结束后：

```python
pred_points = np.asarray(pred_positions)
final_position = pred_points[-1]
ne = norm(final_position - case.gt_final_position)
gt_length = max(path_length(case.gt_positions) - success_threshold, 0.0)
pred_length = path_length(pred_points)
start_to_target_distance = norm(case.start_position - case.target_position)
start_to_gt_final_distance = norm(case.start_position - case.gt_final_position)
spl = 0.0
if success and gt_length > 0:
    spl = gt_length / max(gt_length, pred_length, 1e-8)
```

代码证据：`engine/evaluate_traveluav_smoke.py:810-819`。

注意：

- `ne` 是最终实际位置到专家终点 `gt_final_position` 的距离。
- `final_distance_to_target` 是最终实际位置到 `target_position` 的距离。
- `pred_path_length` 来自 AirSim 实际返回位置序列 `pred_positions`，不是模型计划 waypoint 的长度。
- `spl` 只有 `success=True` 时才可能非零。

聚合指标：

```python
sr = success_count / total * 100
osr = oracle_count / total * 100
ne = mean(item["ne"])
spl = mean(item["spl"]) * 100
collision_count = sum(item["collision"])
early_end_count = sum(item["early_end"])
mean_final_distance_to_target = mean(...)
mean_pred_path_length = mean(...)
mean_gt_path_length_minus_threshold = mean(...)
```

代码证据：`engine/evaluate_traveluav_smoke.py:859-884`。

输出：

```text
eval_trajectory.json
eval_overall.json
rollouts.jsonl
```

代码证据：`engine/evaluate_traveluav_smoke.py:1010-1016`。

`engine/metrics.py` 里明确说明离线 JSONL 不能计算真实轨迹指标；没有 simulator 时 trajectory metrics 返回 null。当前真实 NE/SR/OSR/SPL 是 `evaluate_traveluav_smoke.py` 在在线 AirSim 仿真中计算的，不是离线 `engine/metrics.py` 估算出来的。

代码证据：`engine/metrics.py:10-13`、`201-229`。

## 13. 为什么“下一步点确定”但“实际位置不确定为该点”

严格按代码可以分成两种坐标：

### 13.1 计划坐标

保存在 `model_steps/*.json`：

- `world_position`
- `world_delta`
- `next_world_position`
- `world_waypoints`

这些由当前 AirSim 状态、模型动作和矩阵变换确定。

代码证据：`engine/evaluate_traveluav_smoke.py:481-528`、`768-795`。

### 13.2 实际坐标

保存在：

- `log/*.json` 的 `sensors.state.position`
- `summary.final_position`
- 用于指标的 `pred_positions`

这些来自 AirSim：

```python
payload = state_payload(client)
position = payload["position"]
pred_positions.append(pos)
```

代码证据：`engine/evaluate_traveluav_smoke.py:372-388`、`612-613`、`750-755`。

如果没有碰撞、没有超时、没有卡住，且 AirSim 成功按路径执行，实际位置通常会沿 waypoint 前进。但代码不保证最后状态精确等于 `next_world_position`，因为：

1. `moveOnPathAsync` 是 AirSim RPC，不是直接赋值。
2. 完成 waypoint 的判据是“距离开始变大”，不是距离为 0。
3. 发生 AirSim 碰撞时，循环会提前 break。
4. 超过 `move_timeout_s` 时，循环会提前 break。
5. 20 次采样窗口内移动小于 0.1m 时，循环会提前 break。

代码证据：`engine/evaluate_traveluav_smoke.py:591-641`。

## 14. 可修改的仿真参数

下面这些参数可以通过 `scripts/run_traveluav_closed_loop_eval.sh` 直接改，或通过 `--extra-arg` 传给 engine。

### 14.1 模型和数据参数

| 参数 | 入口 | 影响 |
|---|---|---|
| `--checkpoint` | wrapper/engine | 使用哪个 HAD checkpoint |
| `--vocab` / `--vocab_path` | wrapper/engine | 指令 tokenizer vocab |
| `--raw-data-dir` / `--raw_data_dir` | wrapper/engine | TravelUAVData 根目录 |
| `--scene` | wrapper/engine | 评估哪个场景 |
| `--num-trajectories` / `--num_trajectories` | wrapper/engine | 评估多少条轨迹 |
| `--trajectory-ids` / `--trajectory_ids` | wrapper/engine | 指定精确轨迹 |
| `--start-index` / `--start_index` | wrapper/engine | 从排序后的第几条开始 |
| `--image_size` | engine only | 输入模型的图像尺寸 |
| `--max_inst_len` | engine only | 指令 token 最大长度 |
| `--uav_position_scale` | engine only | 位置输入归一化尺度 |

### 14.2 AirSim/TravelUAV 连接参数

| 参数 | 入口 | 影响 |
|---|---|---|
| `--traveluav-root` / `--traveluav_root` | wrapper/engine | TravelUAV repo 根目录 |
| `--env-root` / `--env_root` | wrapper/engine | Unreal 场景可执行文件根目录 |
| `--start-server` / `--start_server` | wrapper/engine | 是否由 evaluator 启动 server |
| `--keep-server` / `--keep_server` | wrapper/engine | 结束后是否保留 server |
| `--server-ip` / `--server_ip` | wrapper/engine | TravelUAV msgpackrpc server IP |
| `--server-port` / `--server_port` | wrapper/engine | TravelUAV msgpackrpc server port |
| `--gpu-id` / `--gpu_id` | wrapper/engine | 场景启动用 GPU id |
| `--scene-wait-s` / `--scene_wait_s` | wrapper/engine | 打开场景后等待时间 |
| `--airsim-connect-timeout` / `--airsim_timeout` | wrapper/engine | AirSim client timeout |
| `--server_wait_s` | engine only | 等 TravelUAV server ping 的时间 |

### 14.3 闭环控制参数

| 参数 | 入口 | 影响 |
|---|---|---|
| `--max-steps` / `--max_steps` | wrapper/engine | 每条轨迹最多模型 step 数 |
| `--velocity` | wrapper/engine | 传给 AirSim `moveOnPath` 的速度 |
| `--waypoint-count` / `--waypoint_count` | wrapper/engine | 一步动作插值 waypoint 数量输入 |
| `--move-timeout-s` / `--move_timeout_s` | wrapper/engine | 单个模型动作允许 AirSim 移动的最长时间 |
| `--success-threshold` / `--success_threshold` | wrapper/engine | 成功/OSR 判定距离阈值 |
| `--stop-threshold` / `--stop_threshold` | wrapper/engine | 模型 stop 概率阈值 |
| `--stop-on-collision` / `--stop_on_collision` | wrapper/engine | 当前 engine 中不控制终止逻辑；解析存在但未参与判断 |
| `--front_camera` | engine only | 前视相机名 |
| `--down_camera` | engine only | 下视相机名 |

### 14.4 图像记录参数

| 参数 | 入口 | 影响 |
|---|---|---|
| `--record-images` / `--record_images` | wrapper/engine | 是否保存模型每步图像 |
| `--record-image-stride` / `--record_image_stride` | wrapper/engine | 每隔多少 step 保存一次 |
| `--record-image-width` / `--record_image_width` | wrapper/engine | 保存图像宽度 |
| `--record-image-format` / `--record_image_format` | wrapper/engine | jpg/jpeg/png/webp |
| `--record-image-quality` / `--record_image_quality` | wrapper/engine | jpg/webp 质量 |

### 14.5 目标物体参数

| 参数 | 入口 | 影响 |
|---|---|---|
| `--spawn-target` / `--spawn_target` | wrapper/engine | 尝试在目标位置生成目标物体 |
| `--require-target-spawn` / `--require_target_spawn` | wrapper/engine | 生成失败是否报错 |

## 15. 建议调参时优先关注的代码事实

如果目标是分析当前大量 `collision=True`：

1. 先区分 AirSim 真碰撞和 timeout/stuck：
   - 真碰撞通常有 `collision_step`、`collision_log_index`、`collision_object_name`。
   - timeout/stuck 可能只有 `collision=True`，但上述三个字段为 null。
2. 如果怀疑单步移动时间不够，直接相关参数是：
   - `--velocity`
   - `--move-timeout-s`
   - `--waypoint-count`
3. 如果怀疑动作太大或路径太贴地，当前代码没有动作裁剪、安全高度、地形避障；模型输出经矩阵变换后直接作为 AirSim 路径。
4. 如果希望 `--stop-on-collision` 真正控制是否碰撞即停，需要改 `run_case()` 中 `if collision: break` 的逻辑；当前代码不使用该参数。
5. 如果要可视化模型看到的图像，必须打开 `--record-images`；否则 `model_steps/*.json` 中 `image_paths` 为空。

## 16. 一次 step 的完整伪代码

下面伪代码严格对应当前实现：

```text
reset_vehicle(case.start_position, case.start_orientation)
initial_state = AirSim.get_state()
pred_positions = [initial_state.position]
observations = [initial_state]

for step in range(max_steps):
    state = AirSim.get_state()
    position = state.position
    yaw = euler(state.orientation).yaw

    start_local = start_rotation.T @ (position - case.start_position)
    target_local_position = target_basis.T @ start_local
    target_local_yaw = wrap(wrap(yaw - start_yaw) - target_align_yaw)
    altitude = abs(position.z)

    front_img, down_img = AirSim.get_images(front_camera, down_camera)
    if record_images:
        save front/down images for this step

    inputs = {
        front image tensor,
        down image tensor,
        tokenized instruction,
        altitude,
        step_id,
        optionally sin/cos(target_local_yaw),
        optionally target_local_position / uav_position_scale
    }

    outputs = HAD_model(inputs)
    pred_action = outputs["pred_action"][0]  # [dx, dy, dz, dyaw]
    stop_prob = sigmoid(outputs["stop_logit"][0]) if exists else 0

    target_delta = pred_action[:3]
    start_delta = target_basis @ target_delta
    world_delta = start_rotation @ start_delta
    next_world_position = current_position + world_delta
    next_world_yaw = start_yaw + target_align_yaw + next_target_yaw

    waypoints = interpolate(current_position -> next_world_position)
    pad waypoints to length 5 with next_world_position

    if any waypoint within success_threshold of target_position:
        oracle_success = True

    AirSim.moveOnPath(path=waypoints, velocity=velocity, yaw_mode=next_world_yaw)

    while not all waypoints passed:
        if elapsed_time > move_timeout_s:
            collision = True
            break
        sleep(0.02)
        state = AirSim.get_state()
        if state.collision.has_collided:
            collision = True
            append state to observations
            break
        if recent movement over 20 samples < 0.1:
            collision = True
            break
        if distance_to_current_waypoint starts increasing:
            append state to observations
            advance waypoint index

    pause AirSim
    append observed AirSim states to logs and pred_positions

    final_position = last AirSim observed position
    distance_to_target = norm(final_position - target_position)
    stopped = stop_prob >= stop_threshold
    write model_steps/<step>.json

    if stopped and distance_to_target <= success_threshold and not early_end:
        success = True
        break
    if stopped:
        early_end = True
        stop_step = step if unset
    if early_end and oracle_success:
        break
    if collision:
        break

compute NE/SR/OSR/SPL-related fields from actual AirSim pred_positions
write log/*.json, ori_info.json, summary.json
rename running directory to success/oracle/fail directory
```

## 17. 当前报告的边界

本文没有假设 AirSim 内部动力学方程。当前可见代码只能证明：

1. HAD 如何生成动作。
2. HAD 如何把动作转换为世界系 waypoint。
3. HAD 如何调用 AirSim `moveOnPath`。
4. HAD 如何轮询 AirSim 状态。
5. HAD 如何设置 collision、success、oracle_success、NE、SR、OSR、SPL。

AirSim 内部 `moveOnPath` 如何具体控制 SimpleFlight 多旋翼，需要 AirSim/Unreal 服务端源码才能进一步逐行证明；该实现不在当前 TravelUAV Python 仓库中。
