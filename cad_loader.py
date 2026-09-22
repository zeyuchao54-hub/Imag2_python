"""
cad_loader.py
工业级 CAD 模型加载与表面采样模块
职责: 读取 STL -> 构建水密/非水密 Mesh -> 按面积加权表面采样 -> 估计法向 -> 输出 Open3D 点云
"""

import logging
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from utils import validate_input_file

#: get_vertices() 中合并重复 STL 顶点的容差 (mm)。
#: STL 按三角面重复存储顶点，同一几何顶点在相邻面中各存一份。
_VERTEX_MERGE_TOLERANCE_MM = 0.01


class CADLoader:
    """
    将 STL 三角网格模型转换为可用于 ICP 配准的稠密表面点云。

    设计要点:
    - 使用 trimesh 读取 STL，避免自己实现 parser
    - 采样时考虑三角形面积，避免简单取 vertices 导致的密度不均
    - 法向直接来自 mesh face normal，保证与 CAD 理想表面一致
    - 输出为 open3d.geometry.PointCloud，便于复用现有 pipeline 中的 Open3D 工具
    """

    def __init__(
        self,
        num_points: int = 50000,
        use_even_sampling: bool = True,
        seed: Optional[int] = 42,
    ):
        """
        :param num_points: 目标采样点数，默认 50k。实际点数可能略有偏差。
        :param use_even_sampling: 是否使用 sample_surface_even 获得更均匀分布。
                                  若失败则退回到 sample_surface。
        :param seed: 随机种子，保证可复现；设为 None 则不固定。
        """
        self.logger = logging.getLogger("PointToCAD_System.CADLoader")
        self.num_points = int(num_points)
        self.use_even_sampling = bool(use_even_sampling)
        self.seed = seed
        self._mesh: Optional[trimesh.Trimesh] = None

        if self.num_points <= 0:
            raise ValueError(f"num_points 必须为正整数，收到: {num_points}")

    def load(self, stl_path: Union[str, Path]) -> o3d.geometry.PointCloud:
        """
        主入口: 读取 STL 文件并返回带法向的 CAD 表面点云。

        :param stl_path: STL 文件路径
        :return: open3d.geometry.PointCloud (含 normals)
        :raises: FileNotFoundError, TypeError, ValueError
        """
        stl_path = Path(stl_path)
        validate_input_file(stl_path, allowed_exts=[".stl"])

        self.logger.info(f"开始加载 STL 模型: {stl_path.absolute()}")

        # 1. 加载 mesh
        mesh = self._load_mesh(stl_path)
        self._mesh = mesh

        # 2. 表面采样
        points, normals = self._sample_surface(mesh)

        if len(points) == 0:
            raise ValueError("STL 表面采样结果为空，请检查模型是否有效。")

        # 3. 构造 Open3D 点云并赋值法向
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        self.logger.info(
            f"CAD 点云生成完成: {len(pcd.points)} 点 | "
            f"包围盒: {pcd.get_axis_aligned_bounding_box().get_extent()}"
        )
        return pcd

    def _load_mesh(self, stl_path: Path) -> trimesh.Trimesh:
        """使用 trimesh 加载 STL 并做基本校验。"""
        mesh = trimesh.load(str(stl_path), force="mesh")

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"读取 {stl_path} 后未得到 Trimesh 对象，实际类型: {type(mesh)}")

        if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            raise ValueError(f"STL 模型为空: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

        self.logger.info(
            f"STL 加载成功: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, "
            f"面积={mesh.area:.2f}, 水密={mesh.is_watertight}"
        )
        return mesh

    def _sample_surface(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        对 mesh 进行面积加权的表面采样，并返回采样点坐标与法向。

        :return: (points, normals)，均为 Nx3 numpy 数组
        """
        # 固定随机种子以获得可复现结果。
        # 通过 trimesh 的 seed 参数传入 (局部生效)，不再调用 np.random.seed()
        # 污染全局 RNG —— 那会连带打乱后续所有 numpy 随机消费方的取值序列。
        try:
            if self.use_even_sampling:
                self.logger.debug(f"使用 sample_surface_even 采样 {self.num_points} 点...")
                points, face_indices = trimesh.sample.sample_surface_even(
                    mesh, self.num_points, seed=self.seed
                )
            else:
                raise RuntimeError("Fallback to sample_surface")
        except Exception as e:
            self.logger.warning(
                f"sample_surface_even 失败或禁用，退回到 sample_surface: {e}"
            )
            points, face_indices = trimesh.sample.sample_surface(
                mesh, self.num_points, seed=self.seed
            )

        # 法向直接取对应 face 的法向
        face_normals = mesh.face_normals
        normals = face_normals[face_indices]

        # 归一化并确保法向朝外/一致
        normals = self._normalize_normals(normals)

        self.logger.info(
            f"表面采样完成: {len(points)} 点 | 平均点间距估算={self._estimate_average_spacing(points):.4f}"
        )
        return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)

    def get_vertices(self) -> np.ndarray:
        """
        返回原始 STL 网格顶点（去重后），用于几何公差中的名义顶点检测。

        STL 按三角面存储顶点，同一个几何顶点会在相邻面中重复出现多次，
        因此这里按容差 0.01mm 合并。旧实现为 O(n²) 的 Python 双重循环
        (对每个顶点线性扫描已有顶点)，实测 2000 个顶点需 4.5s，
        5 万顶点的真实 STL 需十几分钟。
        现改为 cKDTree.query_pairs + 并查集，复杂度 O(n log n)。
        """
        if self._mesh is None:
            return np.zeros((0, 3))

        vertices = np.asarray(self._mesh.vertices, dtype=float)
        if len(vertices) == 0:
            return vertices

        # 退化情况: 单点或退化包围盒直接返回，避免 KDTree 异常
        if len(vertices) == 1 or not np.all(np.isfinite(vertices)):
            return vertices
        if float(np.linalg.norm(np.ptp(vertices, axis=0))) <= 0.0:
            return vertices

        tree = cKDTree(vertices)
        pairs = tree.query_pairs(_VERTEX_MERGE_TOLERANCE_MM, output_type="ndarray")

        # 并查集: 合并所有距离 <= 容差的顶点对
        parent = np.arange(len(vertices))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]  # 路径压缩
                a = parent[a]
            return a

        for i, j in pairs:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj

        roots = np.array([find(i) for i in range(len(vertices))])
        # 每个等价类取首次出现的顶点作为代表 (与原实现的"保留先到者"一致)
        _, representative_idx = np.unique(roots, return_index=True)
        representative_idx.sort()
        return vertices[representative_idx]

    @staticmethod
    def _normalize_normals(normals: np.ndarray) -> np.ndarray:
        """将法向归一化为单位向量。"""
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        # 避免除以零
        norms = np.where(norms < 1e-12, 1.0, norms)
        return normals / norms

    @staticmethod
    def _estimate_average_spacing(points: np.ndarray) -> float:
        """通过点云包围盒体积与点数估算平均点间距。"""
        if len(points) < 2:
            return 0.0
        bbox = np.max(points, axis=0) - np.min(points, axis=0)
        volume = np.prod(bbox)
        if volume <= 0:
            return 0.0
        # 近似认为点均匀分布在表面上，这里用体积开三次方作为数量级参考
        return float(volume ** (1.0 / 3.0) / (len(points) ** (1.0 / 2.0)))
