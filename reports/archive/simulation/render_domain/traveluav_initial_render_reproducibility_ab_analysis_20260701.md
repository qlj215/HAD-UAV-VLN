# TravelUAV 初始渲染与环境一致性简明分析

生成/修订时间：2026-07-01
项目目录：`/home/qlj/h3c_pro/HAD-UAV-VLN`
主 A/B 实验：`sim_eval_outputs/20260701_192237_initial_view_ab_val_seen_BrushifyCountryRoads_10traj`
旧 96 条批量对照：`sim_eval_outputs/20260701_144300_val_seen_model_teleport`
渲染 warmup 探针：`sim_eval_outputs/20260701_203934_initial_render_warmup_probe_BrushifyCountryRoads_4e405584`
Time of Day 扫描：`sim_eval_outputs/20260701_212844_timeofday_sweep_BrushifyCountryRoads_4e405584`

## 1. 当前结论

目前问题已经不主要是 instruction 或起点 pose。已有实验说明：

1. `--split-metadata-path` 正常传入，A/B 和旧 96 条对照均使用 `split_jsonl` 指令。
2. 起点世界坐标、姿态、yaw、速度在 A/B 中可以做到完全一致。
3. 但模型看到的首帧 RGB 图像不一致，并且和 TravelUAV 原始训练图像差异很大。当前已有保存图片的 10 条 A/B 首帧样本中，front 视角 10/10 明显偏离 raw，down 视角 10/10 也不一致。
4. 单纯等待渲染帧、或者调用 AirSim `simSetTimeOfDay` 扫 8-16 点，都没有把当前仿真画面恢复到训练数据的白天视觉状态。

更准确的判断是：

**当前闭环仿真使用的 BrushifyCountryRoads 环境，其启动后的光照/天空/曝光状态，与 TravelUAV 原始数据采集时的视觉状态不一致。**

这比“首帧渲染未稳定”更严重。首帧不稳定仍然存在，但它不是唯一原因；至少对第一条轨迹而言，即使没有上一条轨迹残留，当前仿真画面也已经明显偏暗。

## 2. A/B 实验要点

A/B 设置：

| 分组 | 含义 |
|---|---|
| A_continuous | 打开一次 `BrushifyCountryRoads`，连续跑 10 条轨迹 |
| B_restart | 每条轨迹单独打开/关闭一次 `BrushifyCountryRoads` |

10 条轨迹的初始图像比较：

| camera | sha256 完全一致 | mean MAE | mean changed pixels |
|---|---:|---:|---:|
| front | 0/10 | 18.05 | 53.78% |
| down | 0/10 | 1.11 | 12.93% |

A/B 初始位姿：

```text
step0_world_diff_m mean = 0.0
step0_world_diff_m max  = 0.0
A/B yaw diff = 0.0
```

A/B 首步动作已经不同：

```text
step0_action_l2 mean = 0.688
step0_action_l2 max  = 2.629
```

完整轨迹层面：

| 指标 | A_continuous | B_restart |
|---|---:|---:|
| n | 10 | 10 |
| success_count | 0 | 0 |
| oracle_success_count | 1 | 1 |
| collision_count | 6 | 6 |
| mean NE | 405.87 | 410.32 |
| mean steps | 92.1 | 90.9 |
| mean final_position_diff(A/B) | 10.63 m | - |

结论：A/B 的宏观指标方向接近，但不是严格可复现；首帧视觉差异会通过模型第一步动作放大成轨迹差异。

## 3. 第一条轨迹的关键发现

重点轨迹：

```text
4e405584-8c33-41cd-9b5f-f3ab290df648
```

A/B 第一条都是各自实验开始后的第一条，按理不应有“上一条轨迹残留”。核验结果：

```text
A position = [-462.93499755859375, -289.8550109863281, -46.058998107910156]
B position = [-462.93499755859375, -289.8550109863281, -46.058998107910156]
position diff = 0.0 m
orientation angle diff = 0.0 deg
yaw diff = 0.0
linear_velocity diff = 0.0
angular_velocity diff = 0.0
```

但图像和 raw 训练首帧的亮度差异很大：

| 图像 | mean brightness |
|---|---:|
| raw `frontcamera/000000.png` | 81.32 |
| A step0 front | 14.73 |
| B step0 front | 17.63 |

像素 MAE：

```text
A vs B   MAE = 4.19
A vs raw MAE = 66.75
B vs raw MAE = 64.94
```

这说明第一条轨迹的问题不是 A/B 之间谁错位，而是 **A/B 当前仿真首帧都明显不像 TravelUAV 原始训练图像**。视觉上看，raw 是白天，当前仿真更像低曝光/夜间。


## 4. 差异是普遍存在还是少数轨迹

当前能直接做像素级对比的是 A/B 实验中保存了首帧图片的 10 条轨迹。旧 96 条批量目录没有保存 RGB 图片，所以不能直接断言 96 条或全数据集全部都有同样差异。

已统计文件：

```text
sim_eval_outputs/20260701_192237_initial_view_ab_val_seen_BrushifyCountryRoads_10traj/raw_vs_sim_initial_image_stats.json
```

这 10 条的结论很明确：**不是少数轨迹异常，而是当前已有样本中普遍存在视觉域差异。**

| 组别 | camera | n | mean MAE to raw | min/max MAE | raw mean | sim mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| A_continuous | front | 10 | 49.53 | 34.15 / 72.94 | 71.54 | 30.30 | 10/10 明显偏暗 |
| B_restart | front | 10 | 60.68 | 44.71 / 74.21 | 71.54 | 13.02 | 10/10 明显偏暗，且比 A 更暗 |
| A_continuous | down | 10 | 44.07 | 9.22 / 78.83 | 44.53 | 1.12 | 10/10 不一致，多数接近全黑 |
| B_restart | down | 10 | 44.51 | 8.91 / 78.86 | 44.53 | 0.02 | 10/10 不一致，几乎全黑 |

按阈值看：

```text
A front: 10/10 MAE > 30
B front: 10/10 MAE > 30
A down : 10/10 MAE >= 1, 9/10 MAE > 10
B down : 10/10 MAE >= 1, 9/10 MAE > 10
```

这说明当前问题不只是某一两条轨迹或某一个视角。front 的主要问题是仿真画面明显偏暗；down 的问题更像是当前 AirSim 截图几乎没有有效地面纹理/亮度，和原始 `downcamera/000000.png` 也存在明显 domain gap。

但严谨地说，这个结论目前只覆盖这 10 条保存了首帧的样本。要判断 96 条或整个 `val_seen/BrushifyCountryRoads` 是否全部如此，需要再做一个轻量批量首帧采样脚本：不跑模型、不跑完整轨迹，只对每条轨迹 reset 到起点，保存 front/down，并和 raw `000000.png` 计算 MAE。

## 5. Warmup 实验结果

实验脚本：

```text
scripts/simulation/experiments/capture_initial_render_warmup_frames.py
```

对同一条轨迹，在 reset 后依次截图：

```text
frame 0
+10 frames
+30 frames
+60 frames
+120 frames
```

累计帧数与结果：

| label | total frames | front mean | front MAE to raw | z |
|---|---:|---:|---:|---:|
| frame_000_total_000 | 0 | 14.71 | 66.77 | -46.059 |
| after_inc_010_total_010 | 10 | 14.81 | 66.68 | -46.036 |
| after_inc_030_total_040 | 40 | 14.89 | 66.59 | -45.930 |
| after_inc_060_total_100 | 100 | 14.89 | 66.58 | -45.410 |
| after_inc_120_total_220 | 220 | 14.78 | 66.64 | -43.611 |

结论：单纯推进渲染/仿真帧没有让画面变亮，也没有接近 raw。并且如果不重新钉住 pose，物理状态会漂移，z 从 `-46.059` 到 `-43.611`。

因此，warmup 可以作为稳定渲染的辅助手段，但它不能解决当前“整体光照状态不一致”的主问题。

## 6. Time of Day 扫描结果

测试调用：

```python
client.simSetTimeOfDay(
    True,
    start_datetime="YYYY-MM-DD HH:MM:SS",
    is_start_datetime_dst=False,
    celestial_clock_speed=1,
    update_interval_secs=1,
    move_sun=True,
)
```

候选时间：

```text
2018-06-21 08:00:00
2018-06-21 10:00:00
2018-06-21 12:00:00
2018-06-21 14:00:00
2018-06-21 16:00:00
```

结果：

| datetime | front mean | front MAE to raw |
|---|---:|---:|
| no TimeOfDay | 14.767 | 66.719 |
| 2018-06-21 08:00:00 | 15.031 | 66.472 |
| 2018-06-21 10:00:00 | 15.284 | 66.231 |
| 2018-06-21 12:00:00 | 15.396 | 66.125 |
| 2018-06-21 14:00:00 | 15.447 | 66.071 |
| 2018-06-21 16:00:00 | 15.461 | 66.052 |

最接近的是 `2018-06-21 16:00:00`，但仍然很差：raw 亮度约 `81`，当前仍只有 `15` 左右。

结论：对当前 `BrushifyCountryRoads` 环境，AirSim `simSetTimeOfDay` 只有极小影响，没有把太阳/天空/光照切回训练数据的白天状态。可能原因包括：

1. 该打包环境的 sky/sun/lighting 没有正确接入 AirSim Time of Day API。
2. 训练数据采集用的环境包或默认关卡光照，与当前 `/home/qlj/TravelUAV_envs/BrushifyCountryRoads` 不一致。
3. 需要控制的是 UE 场景内部的特定 Sky/DirectionalLight/曝光蓝图，而不是 AirSim 通用 Time of Day API。

## 7. 代码证据

当前 TravelUAV 代码中没有发现显式调用：

```text
simSetTimeOfDay
TimeOfDay
start_datetime
simSetWeatherParameter
```

TravelUAV server 生成 AirSim settings 时主要设置了：

```text
ClockSpeed
相机位置/FOV/分辨率
AutoExposureMinBrightness = 0.03
AutoExposureMaxBrightness = 1
```

但没有固定：

```text
TimeOfDay
OriginGeopoint
太阳角
天气
场景光照配置
```

原项目 `setPoses` 也只是：

```text
simSetKinematics(...)
simContinueForFrames(1)
simPause(True)
```

没有设置太阳时间。

## 8. 速度结论

A/B 时间效率：

| 分组 | 总墙钟时间 | 平均每条 |
|---|---:|---:|
| A_continuous | 681 s | 68.1 s |
| B_restart | 1333 s | 133.3 s |

B 每条重开约为 A 的 `1.96x`。由于 B 也没有解决视觉一致性问题，不建议把“每条轨迹重开场景”作为默认评估方式。

## 9. 下一步方向

### 8.1 先确认环境包是否一致

优先级最高。需要确认当前使用的：

```text
/home/qlj/TravelUAV_envs/BrushifyCountryRoads
```

是否就是 TravelUAV 原始数据采集使用的同版本环境。建议：

1. 重新核对 HuggingFace `TravelUAV_env` 中 BrushifyCountryRoads 的文件版本、大小、更新时间。
2. 如果有多个 BrushifyCountryRoads/extra_envs 版本，逐个启动同一 pose 截图，与 raw `frontcamera/000000.png` 比较 MAE。
3. 检查当前环境是否打开了错误的关卡/默认 map/lighting scenario。

### 8.2 找 UE 场景内部的光照控制，而不是只依赖 AirSim TimeOfDay

`simSetTimeOfDay` 基本无效，说明可能要从 UE 关卡内部入手：

1. 检查打包环境是否有可用 console command 或 blueprint 参数控制 Sky/DirectionalLight。
2. 尝试 UE console command 调整曝光/后处理，例如固定 auto exposure、禁用 eye adaptation、调整 gamma。
3. 检查 AirSim settings 是否可加入固定曝光参数，例如让 `AutoExposureMinBrightness` 和 `AutoExposureMaxBrightness` 相同，但这只能处理曝光，不一定能处理太阳/天空。

### 8.3 做更大范围的视觉校准

现在只验证了一条轨迹。建议选 5-10 条不同位置的轨迹，比较：

```text
raw front/down 000000
当前仿真 step0 front/down
不同 TimeOfDay
不同曝光设置
不同环境包版本
```

用统一指标排序：

```text
front MAE to raw
front mean brightness gap
down MAE to raw
首步 action L2
```

只有当多个轨迹都接近 raw，才能认为环境视觉状态基本对齐。

### 8.4 如果无法复现原始视觉状态，需要重定评估协议

如果最终确认当前环境无法恢复到 raw 采集光照，那么有两条路线：

1. **仿真评估路线**：接受当前环境视觉域，重新生成/校准一套当前环境下的图像数据，模型训练和闭环评估都用同一视觉域。
2. **数据复现实验路线**：寻找或恢复原始采集环境和光照配置，使当前 AirSim 截图接近 TravelUAV raw 图像，再进行闭环评估。

对现有 HAD 模型，优先推荐第二条；否则模型训练图像和闭环仿真图像存在明显 domain gap，NE/SR/OSR/SPL 的解释会不稳定。

## 10. 简短判断

当前最重要的结论是：

**起点 pose 已经对齐，但视觉环境没有对齐。A/B 首帧差异只是表象，更根本的问题是当前 BrushifyCountryRoads 启动后的光照/天空/曝光状态与 TravelUAV 原始训练图像不一致。当前已保存首帧的 10 条样本里，front/down 都不是个别异常，而是普遍偏离 raw。**

短期不建议继续扩大完整闭环评估规模。应先把“同一 pose 截图接近 raw 训练首帧”这件事解决，再谈大规模 NE/SR/OSR/SPL。
