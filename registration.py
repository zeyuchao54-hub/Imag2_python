"""
registration.py
工业级点云配准模块
职责: CAD 点云与扫描点云的初始粗配准 + coarse-to-fine Point-to-Plane ICP
"""

import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# 单次 ICP 的轻量结果容器 (字段命名对齐 Open3D RegistrationResult)
ICPStepResult = namedtuple("ICPStepResult", ["transformation", "fitness", "inlier_rmse"])


@dataclass
class RegistrationResult:
    """
    封装 ICP 配准结果，便于序列化到 JSON 与下游偏差分析使用。
    """
    success: bool
    fitness: float
    inlier_rmse: float
    transformation: np.ndarray          # 4x4, source -> target
    rotation: np.ndarray                # 3x3
    translation: np.ndarray             # (3,)
    coarse_transformation: np.ndarray   # 4x4, 仅 PCA 粗配准结果
    num_source_points: int
    num_target_points: int
    message: str = ""
    estimated_scale: float = 1.0        # Auto-Scale (Sim3) 估计的全局比例尺, 未启用时为 1.0

    def to_dict(self) -> dict:
        """转换为可 JSON 序列化的字典。"""
        return {
            "success": self.success,
            "fitness": round(float(self.fitness), 6),
            "inlier_rmse": round(float(self.inlier_rmse), 6),
            "transformation": self.transformation.tolist(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "coarse_transformation": self.coarse_transformation.tolist(),
            "num_source_points": self.num_source_points,
            "num_target_points": self.num_target_points,
            "message": self.message,
            "estimated_scale": round(float(self.estimated_scale), 6),
        }


class ICPRegistrar:
    """
    基于 Open3D Point-to-Plane ICP 的粗到精配准器。

    设计要点:
    - source: CAD 点云 (将被变换)
    - target: 扫描点云 (作为参考)
    - 初始位姿: PCA 主轴对齐 + 轴方向歧义搜索
    - 粗 ICP: 大 voxel + 大 correspondence distance，快速拉近
    - 精 ICP: 小 voxel + 小 correspondence distance，精细对齐
    """

    def __init__(
        self,
        coarse_voxel_size: float = 2.0,
        fine_voxel_size: float = 0.5,
        coarse_max_correspondence_distance: float = 10.0,
        fine_max_correspondence_distance: float = 1.0,
        coarse_max_iterations: int = 50,
        fine_max_iterations: int = 50,
        relative_fitness: float = 1e-6,
        relative_rmse: float = 1e-6,
        normal_radius_factor: float = 2.0,
        normal_max_nn: int = 30,
        init_method: str = "plane",
        plane_distance_threshold: float = 1.0,
        plane_max_planes: int = 10,
        plane_min_points: int = 500,
        plane_perp_angle_thresh: float = 15.0,
        fpfh_voxel_size: float = 2.0,
        fpfh_normal_radius_factor: float = 2.0,
        fpfh_feature_radius_factor: float = 5.0,
        fpfh_distance_threshold_factor: float = 1.5,
        fpfh_ransac_n: int = 3,
        fpfh_max_iteration: int = 100000,
        fpfh_confidence: float = 0.999,
        fpfh_edge_length_threshold: float = 0.9,
        datum_weight: float = 10.0,
        datum_dist_thresh: float = 1.5,
        trim_fraction: float = 1.0,
    ):
        """
        :param coarse_voxel_size: 粗配准阶段的 voxel downsample 尺寸 (单位与点云一致，默认 mm)
        :param fine_voxel_size: 精配准阶段的 voxel downsample 尺寸
        :param coarse_max_correspondence_distance: 粗 ICP 最大对应点距离
        :param fine_max_correspondence_distance: 精 ICP 最大对应点距离
        :param coarse_max_iterations: 粗 ICP 最大迭代次数
        :param fine_max_iterations: 精 ICP 最大迭代次数
        :param relative_fitness: 收敛条件 - 相对 fitness 变化阈值
        :param relative_rmse: 收敛条件 - 相对 RMSE 变化阈值
        :param normal_radius_factor: 估计法向时的搜索半径 = voxel_size * factor
        :param normal_max_nn: 估计法向时的最大邻域点数
        :param init_method: 初始位姿方法，'pca' / 'plane' / 'fpfh'。推荐 box-like 零件使用 'plane'，自由曲面使用 'fpfh'。
        :param plane_distance_threshold: 平面检测距离阈值 (仅 init_method='plane')
        :param plane_max_planes: 最大检测平面数 (仅 init_method='plane')
        :param plane_min_points: 平面最小点数 (仅 init_method='plane')
        :param plane_perp_angle_thresh: 判定两平面垂直的角度容差 (仅 init_method='plane')
        :param fpfh_voxel_size: FPFH 特征提取时的降采样体素尺寸 (仅 init_method='fpfh')
        :param fpfh_normal_radius_factor: FPFH 法向估计半径 = fpfh_voxel_size * factor
        :param fpfh_feature_radius_factor: FPFH 特征半径 = fpfh_voxel_size * factor
        :param fpfh_distance_threshold_factor: RANSAC 距离阈值 = fpfh_voxel_size * factor
        :param fpfh_ransac_n: RANSAC 每次采样点数 (3 或 4)
        :param fpfh_max_iteration: RANSAC 最大迭代次数
        :param fpfh_confidence: RANSAC 置信度
        :param fpfh_edge_length_threshold: 边长一致性校验阈值
        :param datum_weight: Datum-Weighted ICP 中基准区域对应点的权重 (默认 10.0; 1.0 等价于标准 ICP)
        :param datum_dist_thresh: 判定 target 点归属 datum 平面的距离阈值 mm (默认 1.5)
        :param trim_fraction: Trimmed ICP 保留的对应点比例 (默认 1.0; 缺失面/遮挡扫描建议 0.9)
        """
        self.logger = logging.getLogger("PointToCAD_System.Registration")

        self.coarse_voxel_size = float(coarse_voxel_size)
        self.fine_voxel_size = float(fine_voxel_size)
        self.coarse_max_corr = float(coarse_max_correspondence_distance)
        self.fine_max_corr = float(fine_max_correspondence_distance)
        self.coarse_max_iter = int(coarse_max_iterations)
        self.fine_max_iter = int(fine_max_iterations)
        self.relative_fitness = float(relative_fitness)
        self.relative_rmse = float(relative_rmse)
        self.normal_radius_factor = float(normal_radius_factor)
        self.normal_max_nn = int(normal_max_nn)

        self.init_method = init_method.lower()
        if self.init_method not in ("pca", "plane", "fpfh"):
            raise ValueError(f"不支持的 init_method: {init_method}，仅支持 'pca' / 'plane' / 'fpfh'")

        self.plane_distance_threshold = float(plane_distance_threshold)
        self.plane_max_planes = int(plane_max_planes)
        self.plane_min_points = int(plane_min_points)
        self.plane_perp_angle_thresh = float(plane_perp_angle_thresh)

        self.fpfh_voxel_size = float(fpfh_voxel_size)
        self.fpfh_normal_radius_factor = float(fpfh_normal_radius_factor)
        self.fpfh_feature_radius_factor = float(fpfh_feature_radius_factor)
        self.fpfh_distance_threshold_factor = float(fpfh_distance_threshold_factor)
        self.fpfh_ransac_n = int(fpfh_ransac_n)
        self.fpfh_max_iteration = int(fpfh_max_iteration)
        self.fpfh_confidence = float(fpfh_confidence)
        self.fpfh_edge_length_threshold = float(fpfh_edge_length_threshold)
        self.datum_weight = float(datum_weight)
        self.datum_dist_thresh = float(datum_dist_thresh)
        self.trim_fraction = float(trim_fraction)

        if self.coarse_voxel_size <= 0 or self.fine_voxel_size <= 0:
            raise ValueError("voxel_size 必须大于 0")
        if self.fpfh_voxel_size <= 0:
            raise ValueError("fpfh_voxel_size 必须大于 0")
        if self.datum_weight <= 0:
            raise ValueError("datum_weight 必须大于 0")
        if not (0.0 < self.trim_fraction <= 1.0):
            raise ValueError("trim_fraction 必须在 (0, 1] 区间")

    def register(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        datum_planes: Optional[List[Tuple[np.ndarray, float]]] = None,
    ) -> RegistrationResult:
        """
        执行完整配准流程。

        :param source: CAD 点云 (将被变换到 target 坐标系)
        :param target: 扫描点云 (参考)
        :param datum_planes: 可选，基准平面列表 [(normal, d), ...] (与 target 同坐标系、同单位)。
                             提供且 datum_weight > 1 时，Fine ICP 切换为 Datum-Weighted 模式。
        :return: RegistrationResult
        """
        self.logger.info("开始 CAD → Scan 配准...")

        if source.is_empty() or target.is_empty():
            return self._fail_result("source 或 target 点云为空")

        self.logger.info(
            f"输入: source={len(source.points)} 点, target={len(target.points)} 点"
        )

        # 1. 粗配准 (统一分发与回退策略)
        t_coarse, method_name = self._initial_alignment(source, target)

        self.logger.info(
            f"{method_name} 粗配准变换矩阵已计算。det(R)={np.linalg.det(t_coarse[:3, :3]):.4f}"
        )

        # 2. 粗 ICP
        source_coarse = source.voxel_down_sample(self.coarse_voxel_size)
        target_coarse = target.voxel_down_sample(self.coarse_voxel_size)
        self._ensure_normals(source_coarse, self.coarse_voxel_size)
        self._ensure_normals(target_coarse, self.coarse_voxel_size)

        self.logger.info(
            f"粗 ICP: voxel={self.coarse_voxel_size}mm, max_corr={self.coarse_max_corr}mm, "
            f"points={len(source_coarse.points)}/{len(target_coarse.points)}"
        )
        result_coarse = self._run_icp(
            source=source_coarse,
            target=target_coarse,
            init_transform=t_coarse,
            max_correspondence_distance=self.coarse_max_corr,
            max_iterations=self.coarse_max_iter,
        )
        self.logger.info(
            f"粗 ICP 结果: fitness={result_coarse.fitness:.4f}, rmse={result_coarse.inlier_rmse:.4f}"
        )

        # 3. 精 ICP (提供 datum_planes 时切换为 Datum-Weighted 模式)
        source_fine = source.voxel_down_sample(self.fine_voxel_size)
        target_fine = target.voxel_down_sample(self.fine_voxel_size)
        self._ensure_normals(source_fine, self.fine_voxel_size)
        self._ensure_normals(target_fine, self.fine_voxel_size)

        datum_mask = self._compute_datum_mask(np.asarray(target_fine.points), datum_planes)
        use_weighted = (
            datum_mask is not None
            and int(datum_mask.sum()) >= 50
            and self.datum_weight > 1.0
        )

        if use_weighted:
            fine_method = f"Datum-Weighted (w={self.datum_weight:g})"
            self.logger.info(
                f"精 ICP [{fine_method}]: voxel={self.fine_voxel_size}mm, max_corr={self.fine_max_corr}mm, "
                f"points={len(source_fine.points)}/{len(target_fine.points)}, "
                f"datum 点={int(datum_mask.sum())}/{len(target_fine.points)}"
            )
            result_fine = self._run_weighted_icp(
                source=source_fine,
                target=target_fine,
                init_transform=result_coarse.transformation,
                max_correspondence_distance=self.fine_max_corr,
                max_iterations=self.fine_max_iter,
                datum_mask=datum_mask,
            )
        else:
            if datum_planes and self.datum_weight > 1.0:
                self.logger.warning("datum 区域内的 target 点不足 (<50)，Fine ICP 退化为标准 Point-to-Plane")
            fine_method = "Standard Point-to-Plane"
            self.logger.info(
                f"精 ICP [{fine_method}]: voxel={self.fine_voxel_size}mm, max_corr={self.fine_max_corr}mm, "
                f"points={len(source_fine.points)}/{len(target_fine.points)}"
            )
            result_fine = self._run_icp(
                source=source_fine,
                target=target_fine,
                init_transform=result_coarse.transformation,
                max_correspondence_distance=self.fine_max_corr,
                max_iterations=self.fine_max_iter,
            )
        self.logger.info(
            f"精 ICP 结果 ({fine_method}): fitness={result_fine.fitness:.4f}, rmse={result_fine.inlier_rmse:.4f}"
        )

        # 4. 封装最终结果 (使用全分辨率 source 的变换)
        transform = result_fine.transformation
        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        return RegistrationResult(
            success=True,
            fitness=result_fine.fitness,
            inlier_rmse=result_fine.inlier_rmse,
            transformation=transform,
            rotation=rotation,
            translation=translation,
            coarse_transformation=t_coarse,
            num_source_points=len(source.points),
            num_target_points=len(target.points),
            message=f"ICP 配准成功 (fine: {fine_method})",
        )

    def _run_icp(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        init_transform: np.ndarray,
        max_correspondence_distance: float,
        max_iterations: int,
    ) -> o3d.pipelines.registration.RegistrationResult:
        """执行一次 Open3D Point-to-Plane ICP。"""
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=self.relative_fitness,
            relative_rmse=self.relative_rmse,
            max_iteration=max_iterations,
        )
        return o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )

    def _ensure_normals(self, pcd: o3d.geometry.PointCloud, voxel_size: float):
        """若点云没有法向，则按 voxel_size 估算搜索半径并估计法向。"""
        if pcd.has_normals():
            return

        radius = voxel_size * self.normal_radius_factor
        self.logger.debug(f"估计法向: radius={radius:.4f}, max_nn={self.normal_max_nn}")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=self.normal_max_nn
            )
        )
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    def _initial_alignment(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> Tuple[np.ndarray, str]:
        """
        统一的初始位姿分发 (register 与 estimate_scale_factor 共用，避免策略漂移)。
        统一回退策略: 选定方法失败 → PCA → 单位矩阵。
        :return: (transform, method_name)
        """
        if self.init_method == "plane":
            self.logger.info("使用基于 RANSAC 平面的初始位姿估计...")
            t_init = self._plane_based_initial_alignment(source, target)
            method_name = "Plane-based"
        elif self.init_method == "fpfh":
            self.logger.info("使用 FPFH + RANSAC 全局配准估计初始位姿...")
            t_init = self._fpfh_initial_alignment(source, target)
            method_name = "FPFH-RANSAC"
        else:
            self.logger.info("使用 PCA 初始位姿估计...")
            t_init = self._pca_initial_alignment(source, target)
            method_name = "PCA"

        if t_init is None and self.init_method != "pca":
            self.logger.warning(f"{method_name} 初始位姿估计失败，回退到 PCA")
            t_init = self._pca_initial_alignment(source, target)
            method_name = "PCA (fallback)"

        if t_init is None:
            self.logger.warning(f"{method_name} 初始位姿估计失败，退回到单位矩阵")
            t_init = np.eye(4)
            method_name = "Identity (fallback)"

        return t_init, method_name

    def _pca_initial_alignment(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> np.ndarray:
        """
        基于 PCA 主成分分析计算 source → target 的粗变换。

        算法:
        1. 分别计算 source/target 的质心，平移到原点
        2. 计算协方差矩阵与特征向量 (按特征值降序排列)
        3. 枚举 source 特征向量的合法符号翻转 (保持右手系)
        4. 用 fitness 评分选择最佳初始位姿
        """
        source_pts = np.asarray(source.points)
        target_pts = np.asarray(target.points)

        src_center = np.mean(source_pts, axis=0)
        tgt_center = np.mean(target_pts, axis=0)

        src_eigvec = self._compute_sorted_eigenvectors(source_pts - src_center)
        tgt_eigvec = self._compute_sorted_eigenvectors(target_pts - tgt_center)

        best_fitness = -1.0
        best_transform = np.eye(4)

        # 枚举 source 主轴的符号翻转，只保留 det(R)=+1 的组合
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                sz = sx * sy  # 保证右手系
                signs = np.diag([sx, sy, sz])
                src_oriented = src_eigvec @ signs

                R = tgt_eigvec @ src_oriented.T
                if not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
                    continue

                t = tgt_center - R @ src_center
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t

                # 用 voxel 降采样后的点云快速评估该初始位姿
                fitness = self._evaluate_initial_transform(
                    source, target, T, voxel_size=self.coarse_voxel_size
                )
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_transform = T

        self.logger.info(
            f"PCA 最佳候选位姿 fitness={best_fitness:.4f} (基于 coarse voxel 评估)"
        )
        return best_transform

    @staticmethod
    def _compute_sorted_eigenvectors(points: np.ndarray) -> np.ndarray:
        """计算点云的协方差矩阵特征向量，并按特征值从大到小排列。"""
        cov = np.cov(points.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 按特征值降序
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        return eigvecs

    def _evaluate_initial_transform(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        transform: np.ndarray,
        voxel_size: float,
    ) -> float:
        """用 Open3D 的 evaluate_registration 快速评估一个初始位姿的 fitness。"""
        try:
            source_down = source.voxel_down_sample(voxel_size)
            target_down = target.voxel_down_sample(voxel_size)
            self._ensure_normals(target_down, voxel_size)
            result = o3d.pipelines.registration.evaluate_registration(
                source_down,
                target_down,
                max_correspondence_distance=self.coarse_max_corr,
                transformation=transform,
            )
            return float(result.fitness)
        except Exception as e:
            self.logger.debug(f"初始位姿评估失败: {e}")
            return -1.0

    def _plane_based_initial_alignment(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> Optional[np.ndarray]:
        """
        基于 RANSAC 平面检测的粗配准。

        适用场景: box-like 工业零件（立方体、棱柱等），其主轴可由 3 个互相垂直的平面确定。
        算法:
        1. 在 source/target 中分别迭代检测主平面
        2. 找出 3 个互相垂直的平面组成角点框架
        3. 枚举 source ↔ target 平面的对应关系与法向符号
        4. 通过角点/质心对齐构造候选变换，用 fitness 评分选出最佳
        """
        self.logger.info("正在从 source/target 中提取 RANSAC 平面...")
        source_planes = self._extract_planes_for_alignment(source)
        target_planes = self._extract_planes_for_alignment(target)

        self.logger.info(f"Source 平面数: {len(source_planes)}, Target 平面数: {len(target_planes)}")

        if len(source_planes) < 3 or len(target_planes) < 3:
            self.logger.warning("平面数量不足 3 个，无法执行基于平面的粗配准")
            return None

        source_triplets = self._find_perpendicular_triplets(source_planes)
        target_triplets = self._find_perpendicular_triplets(target_planes)

        if not source_triplets or not target_triplets:
            self.logger.warning("未找到互相垂直的三平面组合")
            return None

        self.logger.info(
            f"找到 Source 三平面组: {len(source_triplets)}, Target 三平面组: {len(target_triplets)}"
        )

        best_fitness = -1.0
        best_transform = np.eye(4)
        eval_distance = max(self.coarse_max_corr, 5.0)

        # 以 source 的第一组三平面为基准（通常是最显著的角点）
        src_trip = source_triplets[0]
        src_frame, src_signs = self._build_frame_from_planes(source_planes, src_trip)
        src_corner = self._solve_plane_intersection(
            [source_planes[i] for i in src_trip]
        )
        src_center = np.asarray(source.points).mean(axis=0)

        for tgt_trip in target_triplets:
            tgt_frame_base, _ = self._build_frame_from_planes(target_planes, tgt_trip)
            tgt_corner = self._solve_plane_intersection(
                [target_planes[i] for i in tgt_trip]
            )
            tgt_center = np.asarray(target.points).mean(axis=0)

            # 枚举 target 法向符号的 8 种组合，只保留右手系
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        signs = np.diag([sx, sy, sz])
                        tgt_frame = tgt_frame_base @ signs
                        if not np.isclose(np.linalg.det(tgt_frame), 1.0, atol=1e-3):
                            continue

                        # 枚举三轴对应关系 (source axis i -> target axis perm[i])
                        for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                            # 构造旋转: source 的坐标轴重新排列后对齐到 target
                            R = tgt_frame[:, perm] @ src_frame.T
                            if not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
                                continue

                            # 候选 1: 对齐角点
                            t_corner = tgt_corner - R @ src_corner
                            T_corner = np.eye(4)
                            T_corner[:3, :3] = R
                            T_corner[:3, 3] = t_corner
                            fitness_corner = self._evaluate_initial_transform(
                                source, target, T_corner, voxel_size=self.coarse_voxel_size
                            )

                            # 候选 2: 对齐质心
                            t_center = tgt_center - R @ src_center
                            T_center = np.eye(4)
                            T_center[:3, :3] = R
                            T_center[:3, 3] = t_center
                            fitness_center = self._evaluate_initial_transform(
                                source, target, T_center, voxel_size=self.coarse_voxel_size
                            )

                            for fitness, T, mode in [
                                (fitness_corner, T_corner, "corner"),
                                (fitness_center, T_center, "center"),
                            ]:
                                if fitness > best_fitness:
                                    best_fitness = fitness
                                    best_transform = T
                                    self.logger.debug(
                                        f"新的最佳候选: fitness={fitness:.4f}, "
                                        f"perm={perm}, signs=({sx},{sy},{sz}), mode={mode}"
                                    )

        self.logger.info(f"Plane-based 最佳候选 fitness={best_fitness:.4f}")
        return best_transform if best_fitness > 0 else None

    def _extract_planes_for_alignment(self, pcd: o3d.geometry.PointCloud) -> List[dict]:
        """迭代 RANSAC 提取平面，返回归一化后的平面列表。"""
        planes = []
        current = pcd
        total_points = len(pcd.points)

        for _ in range(self.plane_max_planes):
            if len(current.points) < self.plane_min_points:
                break

            model, inliers = current.segment_plane(
                distance_threshold=self.plane_distance_threshold,
                ransac_n=3,
                num_iterations=1000,
            )

            if len(inliers) < self.plane_min_points:
                break

            inlier_cloud = current.select_by_index(inliers)
            pts = np.asarray(inlier_cloud.points)
            center = pts.mean(axis=0)

            # 归一化平面方程，使法向为单位向量
            model = np.array(model, dtype=np.float64)
            norm_len = np.linalg.norm(model[:3])
            if norm_len < 1e-12:
                continue
            model = model / norm_len

            planes.append({
                "model": model,
                "normal": model[:3],
                "d": model[3],
                "center": center,
                "inlier_count": len(inliers),
            })

            current = current.select_by_index(inliers, invert=True)

        # 按内点数从大到小排序，保证显著平面优先
        planes.sort(key=lambda p: p["inlier_count"], reverse=True)
        return planes

    @staticmethod
    def _find_perpendicular_triplets(planes: List[dict]) -> List[Tuple[int, int, int]]:
        """找出所有互相垂直的三平面组合。"""
        triplets = []
        n = len(planes)
        thresh = np.cos(np.radians(90.0 - 15.0))  # 75°~105° 视为垂直

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    ni, nj, nk = planes[i]["normal"], planes[j]["normal"], planes[k]["normal"]
                    if (abs(np.dot(ni, nj)) < thresh and
                        abs(np.dot(nj, nk)) < thresh and
                        abs(np.dot(nk, ni)) < thresh):
                        triplets.append((i, j, k))
        return triplets

    @staticmethod
    def _build_frame_from_planes(
        planes: List[dict], triplet: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        由三平面法向构建右手正交标架。
        返回 (frame, signs): frame 的列为 3 个正交单位轴；signs 为使用的符号对角阵。
        """
        n1, n2, n3 = [planes[i]["normal"] for i in triplet]

        # 默认尝试右手系；如果叉积反向，翻转 n3
        if np.dot(np.cross(n1, n2), n3) < 0:
            n3 = -n3

        frame = np.column_stack([n1, n2, n3])
        # 正交化 (处理 RANSAC 噪声)
        u, _, vh = np.linalg.svd(frame)
        frame = u @ vh
        return frame, np.eye(3)

    @staticmethod
    def _solve_plane_intersection(planes: List[dict]) -> np.ndarray:
        """求解三平面交点。"""
        A = np.array([p["normal"] for p in planes])
        b = -np.array([p["d"] for p in planes])
        try:
            if abs(np.linalg.det(A)) < 1e-8:
                return np.zeros(3)
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros(3)

    def _fpfh_initial_alignment(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> Optional[np.ndarray]:
        """
        基于 FPFH (Fast Point Feature Histograms) + RANSAC 的全局配准。

        适用场景: 自由曲面、圆角、斜面零件，或 box-like 零件但平面检测不稳定时。
        算法:
        1. 对 source/target 降采样并估计法向
        2. 计算 FPFH 特征 (33 维局部几何描述子)
        3. 在特征空间找最近邻对应点
        4. 用 RANSAC + 边长/距离校验估计最佳刚体变换
        """
        voxel_size = self.fpfh_voxel_size
        self.logger.info(
            f"FPFH 预处理: voxel={voxel_size}, normal_radius={voxel_size * self.fpfh_normal_radius_factor}, "
            f"feature_radius={voxel_size * self.fpfh_feature_radius_factor}"
        )

        source_down, source_fpfh = self._preprocess_for_fpfh(source, voxel_size)
        target_down, target_fpfh = self._preprocess_for_fpfh(target, voxel_size)

        self.logger.info(
            f"FPFH 特征: source={len(source_down.points)} 点, target={len(target_down.points)} 点"
        )

        distance_threshold = voxel_size * self.fpfh_distance_threshold_factor
        self.logger.info(
            f"FPFH RANSAC: distance_threshold={distance_threshold:.3f}, ransac_n={self.fpfh_ransac_n}"
        )

        try:
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                True,  # mutual_filter
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                self.fpfh_ransac_n,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        self.fpfh_edge_length_threshold
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(
                    max_iteration=self.fpfh_max_iteration,
                    confidence=self.fpfh_confidence,
                ),
            )

            self.logger.info(
                f"FPFH-RANSAC 结果: fitness={result.fitness:.4f}, "
                f"inlier_rmse={result.inlier_rmse:.4f}, correspondences={len(result.correspondence_set)}"
            )

            # 如果 fitness 过低，认为不可靠，返回 None 以触发 fallback
            if result.fitness < 1e-6 or not np.isfinite(result.transformation).all():
                self.logger.warning("FPFH-RANSAC 结果不可靠")
                return None

            return np.asarray(result.transformation, dtype=np.float64)

        except Exception as e:
            self.logger.warning(f"FPFH-RANSAC 失败: {e}")
            return None

    def _preprocess_for_fpfh(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float,
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """
        为 FPFH 全局配准预处理点云：降采样、估计法向、计算 FPFH 特征。
        """
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * self.fpfh_normal_radius_factor
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=self.normal_max_nn
            )
        )
        pcd_down.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, 0.0])
        )

        radius_feature = voxel_size * self.fpfh_feature_radius_factor
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=100
            ),
        )

        return pcd_down, pcd_fpfh

    # ------------------------------------------------------------------
    # Datum-Weighted ICP (Fine 阶段, 手动加权迭代实现)
    # ------------------------------------------------------------------
    def _compute_datum_mask(
        self,
        target_points: np.ndarray,
        datum_planes: Optional[List[Tuple[np.ndarray, float]]],
    ) -> Optional[np.ndarray]:
        """
        判定 target 点云中每个点是否归属任一 datum 平面 (距离 <= datum_dist_thresh)。
        :return: bool 掩码数组；无 datum 平面时返回 None
        """
        if not datum_planes:
            return None

        d_min = np.full(len(target_points), np.inf)
        for normal, d in datum_planes:
            n = np.asarray(normal, dtype=np.float64)
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n = n / norm
            dists = np.abs(target_points @ n + d)
            d_min = np.minimum(d_min, dists)

        if not np.isfinite(d_min).any():
            return None
        return d_min <= self.datum_dist_thresh

    def _run_weighted_icp(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        init_transform: np.ndarray,
        max_correspondence_distance: float,
        max_iterations: int,
        datum_mask: np.ndarray,
    ) -> ICPStepResult:
        """
        Datum-Weighted Point-to-Plane ICP (论文核心算法, 含可选 Trimmed)。

        目标函数: E(R, t) = Σ w_i · [ (R·p_i + t - q_j)ᵀ·n_j ]²
        其中 w_i = datum_weight 当对应 target 点落在 datum 区域, 否则为 1。
        每轮对微小增量 (ω, v) 构建加权 6×6 最小二乘系统, 并用指数映射更新位姿,
        保证 det(R) = +1 (无反射畸变)。
        """
        src = np.asarray(source.points)
        tgt = np.asarray(target.points)
        tgt_n = np.asarray(target.normals)
        tree = cKDTree(tgt)
        w_all = np.where(datum_mask, self.datum_weight, 1.0)

        T = np.array(init_transform, dtype=np.float64)
        prev_loss = None
        n_iters = 0

        for it in range(max_iterations):
            n_iters = it + 1
            R = T[:3, :3]
            t = T[:3, 3]
            src_tf = src @ R.T + t

            # 最近邻对应
            dists, idx = tree.query(src_tf)
            inl = dists <= max_correspondence_distance
            n_in = int(inl.sum())
            if n_in < 6:
                self.logger.warning(f"加权 ICP 第 {it + 1} 轮内点不足 ({n_in})，提前终止")
                break

            idx_in = idx[inl]
            d_in = dists[inl]
            src_in = src_tf[inl]
            w_in = w_all[idx_in]

            # Trimmed ICP: 仅保留点对点距离最小的对应 (缺失面/飞点防护)
            if self.trim_fraction < 1.0:
                k = max(6, int(n_in * self.trim_fraction))
                order = np.argsort(d_in)[:k]
                idx_in, d_in, src_in, w_in = idx_in[order], d_in[order], src_in[order], w_in[order]

            q_j = tgt[idx_in]
            n_j = tgt_n[idx_in]
            # point-to-plane 残差
            r = np.einsum("ij,ij->i", src_in - q_j, n_j)

            # 加权线性系统 A·[ω, v]ᵀ = b
            A = np.hstack([np.cross(src_in, n_j), n_j])
            sw = np.sqrt(w_in)
            Aw = A * sw[:, None]
            bw = -r * sw
            AtA = Aw.T @ Aw
            Atb = Aw.T @ bw
            try:
                x = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(AtA, Atb, rcond=None)[0]

            omega, v = x[:3], x[3:]
            R_delta = Rotation.from_rotvec(omega).as_matrix()
            T_new = np.eye(4)
            T_new[:3, :3] = R_delta @ R
            T_new[:3, 3] = R_delta @ t + v

            loss = float(np.mean((sw * r) ** 2))
            step = float(np.linalg.norm(omega) + np.linalg.norm(v))
            self.logger.debug(
                f"加权 ICP iter {it + 1}: loss={loss:.6f}, |step|={step:.2e}, inliers={len(idx_in)}"
            )
            T = T_new
            if prev_loss is not None:
                rel = abs(prev_loss - loss) / max(prev_loss, 1e-12)
                if rel < self.relative_rmse and step < 1e-6:
                    break
            prev_loss = loss

        # 最终指标 (与 Open3D 定义一致: fitness=内点率, rmse=点对点距离 RMSE)
        src_tf = src @ T[:3, :3].T + T[:3, 3]
        dists, idx = tree.query(src_tf)
        inl = dists <= max_correspondence_distance
        n_in = int(inl.sum())
        fitness = float(n_in) / len(src) if len(src) > 0 else 0.0
        rmse = float(np.sqrt(np.mean(dists[inl] ** 2))) if n_in > 0 else float("inf")
        datum_share = float(np.mean(datum_mask[idx[inl]])) if n_in > 0 else 0.0
        self.logger.info(
            f"加权 ICP 收敛: iters={n_iters}, fitness={fitness:.4f}, rmse={rmse:.4f}, "
            f"datum 对应占比={datum_share * 100:.1f}%"
        )
        return ICPStepResult(transformation=T, fitness=fitness, inlier_rmse=rmse)

    # ------------------------------------------------------------------
    # Auto-Scale: Sim3 (相似变换) 全局比例尺估计
    # ------------------------------------------------------------------
    @staticmethod
    def weighted_umeyama(
        src: np.ndarray,
        tgt: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        带权重的 Umeyama 相似变换闭式解: min Σ w_i · ||s·R·p_i + t - q_i||²。
        :param src: (N,3) 源点 (虚拟单位)
        :param tgt: (N,3) 对应目标点 (物理单位)
        :param weights: (N,) 可选逐点权重
        :return: (s, R, t)，满足 q ≈ s·R·p + t，且 det(R) = +1
        """
        n = len(src)
        if weights is None:
            w = np.full(n, 1.0 / n)
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = w / max(w.sum(), 1e-12)

        mu_s = (w[:, None] * src).sum(axis=0)
        mu_t = (w[:, None] * tgt).sum(axis=0)
        X = src - mu_s
        Y = tgt - mu_t

        cov = (Y * w[:, None]).T @ X
        U, S, Vt = np.linalg.svd(cov)
        D = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            D[2, 2] = -1.0
        R = U @ D @ Vt

        var_src = float((w[:, None] * X * X).sum())
        if var_src < 1e-12:
            return 1.0, np.eye(3), mu_t - mu_s
        s = float(np.trace(np.diag(S) @ D) / var_src)
        t = mu_t - s * (R @ mu_s)
        return s, R, t

    def estimate_scale_factor(
        self,
        scan_pcd: o3d.geometry.PointCloud,
        cad_pcd: o3d.geometry.PointCloud,
        max_iterations: int = 30,
        gate_ratio: float = 0.12,
        min_correspondences: int = 100,
    ) -> float:
        """
        通过 Sim3 (相似变换) ICP 自动估计 scan(虚拟单位) → CAD(物理单位) 的全局比例尺,
        取代人工交互式标定，实现全自动尺度标定。

        流程: bbox 对角线初值 s0 → 初始刚性位姿 (与 register 共用统一分发) →
              迭代最近邻 + Weighted Umeyama 闭式解, 同时优化 s / R / t。

        :return: scale_factor [mm/虚拟单位]
        """
        self.logger.info("[Auto-Scale] 开始 Sim3 全局比例尺估计 (scan 虚拟单位 → CAD 物理单位)...")

        scan_diag = float(np.linalg.norm(scan_pcd.get_axis_aligned_bounding_box().get_extent()))
        cad_diag = float(np.linalg.norm(cad_pcd.get_axis_aligned_bounding_box().get_extent()))
        if scan_diag < 1e-12 or cad_diag < 1e-12:
            raise ValueError("scan 或 cad 点云包围盒退化，无法估计比例尺")
        s0 = cad_diag / scan_diag
        self.logger.info(f"[Auto-Scale] bbox 对角线初值 s0 = {s0:.4f} (cad={cad_diag:.2f}mm, scan={scan_diag:.4f})")

        # 1. 用 s0 预缩放 scan 并降采样 (此后在"近似 mm"空间中操作)
        scan_scaled = o3d.geometry.PointCloud(scan_pcd)
        scan_scaled.scale(s0, center=(0.0, 0.0, 0.0))
        scan_d = scan_scaled.voxel_down_sample(self.coarse_voxel_size)
        cad_d = cad_pcd.voxel_down_sample(self.coarse_voxel_size)
        self._ensure_normals(scan_d, self.coarse_voxel_size)
        self._ensure_normals(cad_d, self.coarse_voxel_size)

        # 2. 初始刚性位姿 (与 register 共用统一分发与回退策略)
        t_init, init_name = self._initial_alignment(scan_d, cad_d)
        self.logger.info(f"[Auto-Scale] 初始位姿方法: {init_name}")

        # 3. 迭代最近邻 + Weighted Umeyama, 同时优化 scale / R / t
        src_pts = np.asarray(scan_d.points)
        tgt_pts = np.asarray(cad_d.points)
        tree = cKDTree(tgt_pts)
        gate = max(self.coarse_max_corr, gate_ratio * cad_diag)

        T = np.array(t_init, dtype=np.float64)
        s_rel = 1.0
        for it in range(max_iterations):
            src_tf = src_pts @ T[:3, :3].T + T[:3, 3]
            dists, idx = tree.query(src_tf)
            inl = dists <= gate
            n_in = int(inl.sum())
            if n_in < min_correspondences:
                self.logger.warning(f"[Auto-Scale] 第 {it + 1} 轮对应点不足 ({n_in})，提前终止")
                break
            s_new, R_new, t_new = self.weighted_umeyama(src_pts[inl], tgt_pts[idx[inl]])
            if s_new <= 0:
                # 非法尺度 (反射): 保持 T 不变并终止, 由下方回退逻辑处理
                self.logger.warning(f"[Auto-Scale] 第 {it + 1} 轮尺度非正 (s={s_new:.4f})，提前终止迭代")
                s_rel = s_new
                break
            ds = abs(s_new - s_rel)
            dt = float(np.linalg.norm(t_new - T[:3, 3]))
            T = np.eye(4)
            T[:3, :3] = s_new * R_new
            T[:3, 3] = t_new
            s_rel = s_new
            self.logger.debug(f"[Auto-Scale] iter {it + 1}: s_rel={s_rel:.6f}, inliers={n_in}")
            if ds < 1e-7 and dt < 1e-4 * cad_diag:
                break

        if s_rel <= 0:
            self.logger.warning(f"[Auto-Scale] 收敛到非法尺度 s_rel={s_rel:.4f}，回退到 bbox 初值 s0")
            s_rel = 1.0
        if abs(s_rel - 1.0) > 0.3:
            self.logger.warning(
                f"[Auto-Scale] 尺度相对 bbox 初值修正幅度过大 (s_rel={s_rel:.4f})，"
                f"建议核对初始位姿或改用 --scale_factor"
            )

        scale_factor = s0 * s_rel
        self.logger.info(
            f"[Auto-Scale] 完成: scale_factor = {scale_factor:.6f} mm/虚拟单位 "
            f"(s0={s0:.4f} × 收敛修正 {s_rel:.6f})"
        )
        return scale_factor

    def _fail_result(self, message: str) -> RegistrationResult:
        """生成一个失败的 RegistrationResult。"""
        self.logger.error(message)
        return RegistrationResult(
            success=False,
            fitness=0.0,
            inlier_rmse=float("inf"),
            transformation=np.eye(4),
            rotation=np.eye(3),
            translation=np.zeros(3),
            coarse_transformation=np.eye(4),
            num_source_points=0,
            num_target_points=0,
            message=message,
        )
