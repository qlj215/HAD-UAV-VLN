"""
transforms.py
=============
HAD-UAV-VLN 图像预处理与数据增强。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  设计原则
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  视觉语言导航 (VLN) 任务中，图像的空间方向与动作标签强耦合：
  - 水平翻转 → "左转"的像素模式变成"右转", 动作标签失效 ❌
  - 旋转      → 方向角 (yaw) 的视觉线索被篡改            ❌
  - 透视扭曲  → 俯仰/高度感知被打乱                      ❌

  允许的安全增强 (不改变空间语义):
  - 颜色抖动   → 模拟不同光照 / 天气 / 时段               ✓
  - 随机裁剪   → 模拟不同飞行高度导致的视野缩放 (等比)    ✓
  - 高斯模糊   → 模拟运动模糊 / 相机失焦                  ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from datasets.transforms import get_train_transforms, get_val_transforms

  train_tf = get_train_transforms(input_size=(224, 224))
  val_tf   = get_val_transforms(input_size=(224, 224))
"""

from typing import Tuple

from torchvision import transforms as T


def get_train_transforms(
    input_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    color_jitter: bool = True,
    gaussian_blur: bool = False,
) -> T.Compose:
    """获取训练时的图像变换流水线。

    仅包含方向安全的增强 ——
    不会执行翻转、旋转、透视变换，保持空间语义与动作标签一致。

    Args:
        input_size:     目标图像尺寸 (H, W)
        mean:           ImageNet 均值
        std:            ImageNet 标准差
        color_jitter:   是否启用颜色抖动 (默认 True)
        gaussian_blur:  是否启用高斯模糊 (默认 False, 小数据集建议 True)

    Returns:
        torchvision.transforms.Compose
    """
    ops = []

    # 等比缩放 + 裁剪: 模拟飞行高度变化导致的视野缩放
    # ratio=(1.0, 1.0) 保持宽高比不变, 不会扭曲空间结构
    ops.append(
        T.RandomResizedCrop(input_size, scale=(0.85, 1.0), ratio=(1.0, 1.0))
    )

    # 颜色抖动: 模拟光照/天气变化, 不改变空间方向
    if color_jitter:
        ops.append(
            T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
        )

    # 高斯模糊: 模拟运动模糊 / 相机失焦 (可选)
    if gaussian_blur:
        ops.append(T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)))

    # 注意: 不包含 RandomHorizontalFlip / RandomRotation
    # 原因: VLN 任务的 action 与空间方向绑定, 翻转/旋转会破坏这种对应关系

    ops.append(T.ToTensor())
    ops.append(T.Normalize(mean=mean, std=std))

    return T.Compose(ops)


def get_val_transforms(
    input_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """获取验证/测试时的图像变换流水线 (无数据增强)。

    Args:
        input_size:  目标图像尺寸 (H, W)
        mean:        ImageNet 均值
        std:         ImageNet 标准差

    Returns:
        torchvision.transforms.Compose
    """
    return T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
