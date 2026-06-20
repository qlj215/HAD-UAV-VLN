"""
transforms.py
=============
HAD-UAV-VLN 图像预处理。

VLN 动作回归任务中，图像空间几何与动作标签强绑定。训练默认不再使用
RandomResizedCrop、ColorJitter、旋转或翻转等随机增强，只做确定性的尺寸变换
和 ImageNet normalization，以匹配 torchvision 预训练 ResNet 输入分布。
"""

from typing import Tuple

from torchvision import transforms as T


def get_train_transforms(
    input_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    color_jitter: bool = False,
    gaussian_blur: bool = False,
) -> T.Compose:
    """训练图像变换。

    `color_jitter` 和 `gaussian_blur` 参数保留为兼容旧调用，但默认关闭；
    本函数不会做随机裁剪，避免改变目标相对位置和动作标签之间的对应关系。
    """
    ops = [T.Resize(input_size)]

    if color_jitter:
        ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    if gaussian_blur:
        ops.append(T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)))

    ops.append(T.ToTensor())
    ops.append(T.Normalize(mean=mean, std=std))
    return T.Compose(ops)


def get_val_transforms(
    input_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """验证/测试图像变换。"""
    return T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
