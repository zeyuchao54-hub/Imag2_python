import os
import time
import logging
from functools import wraps
from pathlib import Path
from typing import Union, List, Optional

import numpy as np


def diagonal_of(obj) -> float:
    """
    返回点云/点集的轴对齐包围盒 (AABB) 对角线长度，作为"场景物理尺度"的参照。

    接受 open3d.geometry.PointCloud、Point3D 容器、或 (N,3) numpy 数组。
    空输入返回 0.0 (不抛异常，便于调用方退化处理)。

    用途: 模块内的距离类阈值原本是硬编码的绝对值 (如 0.05 / 0.02 / 50.0)，
    隐含假设"点云尺度 ≈ 1 个单位"。这些数值在当前数据集上凑巧合适，
    但换一台扫描仪或换一种规格的零件就会集体失效。改为以对角线的比例表达后，
    阈值随数据自适应。
    """
    if obj is None:
        return 0.0

    if isinstance(obj, np.ndarray):
        pts = obj
    else:
        # open3d 点云: 优先走 hasattr 判断，避免强依赖 open3d 类型
        points_attr = getattr(obj, "points", None)
        if points_attr is None:
            return 0.0
        pts = np.asarray(points_attr)

    if pts.size == 0 or pts.ndim != 2 or pts.shape[0] == 0:
        return 0.0

    extent = np.ptp(pts, axis=0)
    if not np.all(np.isfinite(extent)):
        return 0.0
    return float(np.linalg.norm(extent))


def resolve_threshold(explicit: Optional[float],
                      reference_diagonal: float,
                      ratio: float,
                      floor: float = 0.0,
                      name: str = "threshold") -> float:
    """
    解析一个距离类阈值: 显式传入的绝对值优先，否则按 reference_diagonal × ratio 推导。

    :param explicit: 调用方显式给定的绝对值; None 表示按比例自适应
    :param reference_diagonal: 参照对角线长度
    :param ratio: 相对于参照对角线的比例
    :param floor: 下限，防止退化点云 (对角线≈0) 产生 0 阈值
    :param name: 阈值名，仅用于异常信息
    """
    if explicit is not None:
        if not np.isfinite(explicit) or explicit <= 0:
            raise ValueError(f"{name} 必须为正有限值，收到: {explicit!r}")
        return float(explicit)

    if not np.isfinite(reference_diagonal) or reference_diagonal <= 0:
        return float(floor)
    return max(float(reference_diagonal) * float(ratio), float(floor))


def time_it(func):
    """
    工业级性能监控装饰器 (Decorator)
    职责: 自动监控并记录任何被打上该标签的函数的执行耗时。
    用法: 在目标函数上方加上 @time_it 即可。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取与主系统同源的 Logger，保持日志格式统一
        logger = logging.getLogger("PointToCAD_System.Timer")

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time
        # 使用 DEBUG 级别记录，避免日常运行刷屏，但在排错时极为有用
        logger.debug(f"函数 [{func.__name__}] 执行完毕 -> 耗时: {elapsed_time:.4f} 秒")

        return result

    return wrapper


def validate_input_file(file_path: Union[str, Path], allowed_exts: Optional[List[str]] = None) -> bool:
    """
    工业级防御性文件校验器
    职责: 在读取 3D 数据前，彻底排查路径错误、空文件、非法格式等产线常见异常。

    :param file_path: 输入文件路径
    :param allowed_exts: 允许的后缀名列表，如 ['.ply', '.pcd']
    :return: 校验通过返回 True，否则抛出明确的异常
    """
    path_obj = Path(file_path)

    # 1. 检查是否存在
    if not path_obj.exists():
        raise FileNotFoundError(f"严重错误: 找不到目标文件 '{path_obj.absolute()}'")

    # 2. 检查是否为文件 (防止用户传了个文件夹路径进来)
    if not path_obj.is_file():
        raise IsADirectoryError(f"输入路径必须是文件，但传入了一个目录: '{path_obj.absolute()}'")

    # 3. 检查文件大小 (防止 0KB 的损坏文件导致 Open3D 底层 C++ 崩溃)
    if path_obj.stat().st_size == 0:
        raise ValueError(f"文件大小为 0 字节，可能已损坏: '{path_obj.absolute()}'")

    # 4. 检查文件后缀格式
    if allowed_exts is not None:
        # 将输入后缀转小写，方便统一比对 (如 .PLY 视为 .ply)
        ext = path_obj.suffix.lower()
        if ext not in allowed_exts:
            raise TypeError(
                f"不支持的文件格式 '{ext}'。当前系统仅支持: {', '.join(allowed_exts)}"
            )

    return True


def safe_mkdir(dir_path: Union[str, Path]) -> Path:
    """
    安全创建目录
    避免因为权限问题或多线程竞争导致 os.makedirs 报错
    """
    path_obj = Path(dir_path)
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    except PermissionError:
        logger = logging.getLogger("PointToCAD_System.Utils")
        logger.error(f"权限不足，无法创建目录: {path_obj.absolute()}")
        raise