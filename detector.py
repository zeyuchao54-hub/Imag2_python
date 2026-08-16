import logging
import open3d as o3d
import numpy as np


class RansacDetector:
    """
    工业级 RANSAC 特征检测模块
    职责: 从预处理后的点云中，通过迭代 RANSAC 算法提取多个主平面
    """

    def __init__(self, max_planes=6, min_ratio=0.05, num_iterations=2000, distance_threshold=None):
        """
        :param max_planes: 尝试提取的最大平面数量
        :param min_ratio: 终止条件 - 当剩余点数少于初始点数的此比例时自动停止
        :param num_iterations: 每次 RANSAC 随机采样的最大迭代次数
        :param distance_threshold: 点到平面的距离判定容差 (若为 None 则自适应计算)
        """
        self.logger = logging.getLogger("PointToCAD_System.Detector")
        self.max_planes = max_planes
        self.min_ratio = min_ratio
        self.num_iterations = num_iterations
        self.distance_threshold = distance_threshold

    def detect(self, pcd: o3d.geometry.PointCloud):
        """
        执行多平面提取算法

        :param pcd: 输入的预处理点云
        :return: (raw_planes, rest_pcd)
                 - raw_planes: 包含每个平面详细信息的字典列表
                 - rest_pcd: 剔除所有提取出的平面后，剩余的点云 (残余噪点/杂质)
        """
        if len(pcd.points) == 0:
            self.logger.warning("输入的点云为空，无法提取平面！")
            return [], pcd

        current_pcd = pcd
        total_initial_points = len(pcd.points)
        min_points_threshold = int(total_initial_points * self.min_ratio)

        # 计算自适应的 RANSAC 距离容差
        dist_thresh = self._compute_distance_threshold(current_pcd)

        raw_planes = []
        self.logger.info(
            f"开始 RANSAC 平面提取 (输入点数: {total_initial_points}, 距离阈值: {dist_thresh:.5f})"
        )

        for i in range(self.max_planes):
            current_points_count = len(current_pcd.points)

            # 🟢 终止条件 1: 剩余点数不足总量的设定比例
            if current_points_count < min_points_threshold:
                self.logger.info(
                    f"剩余点数 ({current_points_count}) 低于最小比例限制 ({min_points_threshold})，停止提取。"
                )
                break

            # -------------------------------------------------------------
            # 执行单次 RANSAC 平面拟合
            # -------------------------------------------------------------
            plane_model, inliers = current_pcd.segment_plane(
                distance_threshold=dist_thresh,
                ransac_n=3,
                num_iterations=self.num_iterations
            )

            # 🟢 终止条件 2: 拟合出的平面点数太少 (离群碎片噪点)
            if len(inliers) < 30:
                self.logger.info(f"第 {i+1} 次 RANSAC 提取到的点数稀疏 ({len(inliers)} 点)，终止提取。")
                break

            # 从当前点云中分离出内点 (平面) 和外点 (剩余部分)
            inlier_cloud = current_pcd.select_by_index(inliers)
            outlier_cloud = current_pcd.select_by_index(inliers, invert=True)

            [a, b, c, d] = plane_model
            self.logger.info(
                f"  -> 面 {i+1}: 方程 [{a:7.4f}x + {b:7.4f}y + {c:7.4f}z + {d:7.4f} = 0] | 内点数: {len(inliers)}"
            )

            # 封装标准化数据包
            raw_planes.append({
                "id": i + 1,
                "model": plane_model,  # 平面方程系数 [a, b, c, d]
                "cloud": inlier_cloud,  # 该平面的点云对象
                "inlier_count": len(inliers)
            })

            # 更新当前点云为外点，进入下一轮剥离
            current_pcd = outlier_cloud

        self.logger.info(f"RANSAC 提取完成，共成功捕获 {len(raw_planes)} 个原始平面。")
        return raw_planes, current_pcd

    def _compute_distance_threshold(self, pcd: o3d.geometry.PointCloud) -> float:
        """根据点云的物理对角线尺寸，自适应计算 RANSAC 距离容差"""
        if self.distance_threshold is not None:
            return self.distance_threshold

        bbox = pcd.get_axis_aligned_bounding_box()
        diag_len = np.linalg.norm(bbox.get_extent())

        # 经验公式: 容差取物理包围盒对角线长度的 0.8%
        adaptive_dist = diag_len * 0.008
        self.logger.debug(f"动态推算 RANSAC 容差: {adaptive_dist:.5f}")
        return adaptive_dist