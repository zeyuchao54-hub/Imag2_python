import os
import logging
import open3d as o3d
import numpy as np


class PointCloudPreprocessor:
    """
    工业级点云预处理模块
    职责: 加载点云 -> 自适应降采样 -> 统计去噪 -> 法向量估计
    """

    def __init__(self, target_points=50000, nb_neighbors=30, std_ratio=2.0):
        """
        :param target_points: 期望降采样后的目标点数（用于自适应计算体素大小）
        :param nb_neighbors: 统计滤波时的邻域点数
        :param std_ratio: 统计滤波的标准差倍数阈值
        """
        self.logger = logging.getLogger("PointToCAD_System.Preprocessor")
        self.target_points = target_points
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def process(self, file_path) -> o3d.geometry.PointCloud:
        """
        执行完整的预处理流水线
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到点云文件: {file_path}")

        # 1. 加载文件
        pcd = self._load_point_cloud(file_path)

        # 2. 动态体素降采样
        pcd_down = self._adaptive_downsample(pcd)

        # 3. 统计滤波去噪
        pcd_clean = self._remove_noise(pcd_down)

        # 4. 法向量估计 (对后续 RANSAC 精度和 CAD 重建至关重要)
        pcd_ready = self._estimate_normals(pcd_clean)

        self.logger.info(f"预处理完毕: 最终剩余 {len(pcd_ready.points)} 个高质量点。")
        return pcd_ready

    def _load_point_cloud(self, file_path) -> o3d.geometry.PointCloud:
        """读取点云并进行初步校验"""
        self.logger.info(f"读取原始文件: {file_path}")
        pcd = o3d.io.read_point_cloud(file_path)

        original_count = len(pcd.points)
        if original_count == 0:
            raise ValueError(f"读取失败，文件可能已损坏或为空: {file_path}")

        self.logger.debug(f"原始点云加载成功，点数: {original_count}")
        return pcd

    def _adaptive_downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        自适应体素降采样
        传统做法是写死 voxel_size=0.01，但这会导致处理大尺寸(米级)和小尺寸(毫米级)零件时崩溃。
        这里的算法通过包围盒动态推算合适的体素尺寸。
        """
        original_count = len(pcd.points)

        # 如果原始点数少于目标点数，跳过降采样
        if original_count <= self.target_points:
            self.logger.info(f"点数 ({original_count}) 低于阈值，跳过降采样。")
            return pcd

        # 获取点云的物理尺寸 (Bounding Box)
        bbox = pcd.get_axis_aligned_bounding_box()
        max_extent = np.max(bbox.get_extent())

        # 启发式推算体素大小: 假设点云均匀分布在立方体表面
        # voxel_size ≈ max_extent / sqrt(target_points)
        estimated_voxel_size = max_extent / np.sqrt(self.target_points)

        self.logger.debug(f"自动推算体素尺寸: {estimated_voxel_size:.4f}")

        pcd_down = pcd.voxel_down_sample(voxel_size=estimated_voxel_size)

        self.logger.info(f"降采样完成: {original_count} -> {len(pcd_down.points)}")
        return pcd_down

    def _remove_noise(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """使用统计滤波剔除空间中的飞点和游离噪点"""
        self.logger.debug(f"开始统计滤波去噪 (neighbors={self.nb_neighbors}, std_ratio={self.std_ratio})...")

        pcd_clean, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors,
            std_ratio=self.std_ratio
        )

        noise_count = len(pcd.points) - len(pcd_clean.points)
        self.logger.info(f"去噪完成: 剔除了 {noise_count} 个离群噪点。")

        return pcd_clean

    def _estimate_normals(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        估计法向量并统一定向。
        有了法向量，后续在 CAD 重建时才能知道平面的“正面”和“反面”。
        """
        self.logger.debug("开始计算点云法向量...")

        # 搜索半径设定为平均点间距的几倍
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # 统一法线朝向 (假设相机/扫描仪从外部观测，将法向指向外部)
        # 这里默认以原点为参考，如果扫描仪在其他位置，可修改 camera_location
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        self.logger.info("法向量计算及定向完成。")
        return pcd