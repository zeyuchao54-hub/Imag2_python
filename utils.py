import os
import time
import logging
from functools import wraps
from pathlib import Path
from typing import Union, List, Optional


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