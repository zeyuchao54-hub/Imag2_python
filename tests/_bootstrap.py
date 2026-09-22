"""
测试引导: 必须在任何模块 import open3d 之前执行。

两个职责:
1) 固定 OpenMP 线程数。Open3D 的 RANSAC 在多线程下有竞态 (详见 main.py 顶部注释),
   会使"同种子复现"类测试偶发失败。unittest discover 按字母序导入**所有**测试模块,
   若某个不含该设置的模块先把 open3d 拉起来，后面再 setdefault 就无效了。
   因此每个测试文件都必须先 import 本模块，再 import open3d。
2) 把项目根目录加入 sys.path，使测试可从任意工作目录运行。
"""

import os

# 必须在 import open3d 之前设置 (OpenMP 运行时在共享库首次加载时读取该变量)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
